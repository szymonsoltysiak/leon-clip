#!/usr/bin/env python3
"""Naive vs. chained retrieval over the combination space (same validation pool).

Same checkpoint / split / 1024-cell validation pool as the original probe.
Sweeps:
  * naive   R@D            for every D in 1..D_MAX   (direct query->target)
  * chained R@A@B          for the full A x B grid    (A,B in {1,2,3,4,5,10})
in BOTH directions:
  image -> graph -> text   (direct baseline: image -> text)
  text  -> graph -> image  (direct baseline: text  -> image)

Lets us compare each chained R@A@B against naive R@(A*B) (matched nominal budget)
AND against naive R@D at the chain's actual mean unique-candidate count.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from triple_encoder import EmbeddingStore, H3TripleDataset
from triple_encoder.benchmarks.runner import _build_model_from_checkpoint

ALL_MODALITIES = ("graph", "text", "image")
A_VALUES = [1, 2, 3, 4, 5, 10]
B_VALUES = [1, 2, 3, 4, 5, 10]
D_MAX = 100


def split_indices(length: int, val_split: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(length))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = int(round(length * val_split)) if val_split > 0 else 0
    val_count = min(max(0, val_count), max(0, length - 1))
    return indices[val_count:], indices[:val_count]


@torch.no_grad()
def encode_pool(dataset, pool_indices, model, device, batch_size=256):
    items = [dataset[i] for i in pool_indices]
    out = {m: [] for m in ALL_MODALITIES}
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        batch = {f"{m}_embedding": torch.stack([r[f"{m}_embedding"] for r in chunk]).to(device) for m in ALL_MODALITIES}
        pres = {f"{m}_present": torch.stack([r[f"{m}_present"] for r in chunk]).to(device) for m in ALL_MODALITIES}
        res = model(
            graph_embedding=batch["graph_embedding"], text_embedding=batch["text_embedding"],
            image_embedding=batch["image_embedding"], graph_present=pres["graph_present"],
            text_present=pres["text_present"], image_present=pres["image_present"],
        )
        for m in ALL_MODALITIES:
            out[m].append(res[m].detach())
    feats = {m: torch.cat(out[m], dim=0) for m in ALL_MODALITIES}
    present = {m: torch.stack([r[f"{m}_present"] for r in items]).to(device) for m in ALL_MODALITIES}
    return feats, present


def naive_rank(sim: torch.Tensor) -> torch.Tensor:
    """1-indexed rank of the diagonal positive for each row of a [N,N] sim matrix."""
    n = sim.shape[0]
    diag = sim.diag().view(-1, 1)
    return (sim > diag).sum(dim=1) + 1  # ties -> diagonal ranked after strictly-greater


def naive_recall_curve(rank: torch.Tensor, d_max: int) -> dict[int, float]:
    return {d: float((rank <= d).float().mean().item()) for d in range(1, d_max + 1)}


def chained_recall(sim_first: torch.Tensor, sim_second: torch.Tensor, a: int, b: int) -> tuple[float, float]:
    n = sim_first.shape[0]
    a, b = min(a, n), min(b, n)
    inter = torch.topk(sim_first, a, dim=1).indices            # [N, a]
    top_b = torch.topk(sim_second, b, dim=1).indices           # [N, b]
    cand = top_b[inter].reshape(n, a * b)                      # [N, a*b]
    idx = torch.arange(n, device=cand.device).view(-1, 1)
    hit = (cand == idx).any(dim=1).float().mean().item()
    sc, _ = cand.sort(dim=1)
    uniq = ((sc[:, 1:] != sc[:, :-1]).sum(dim=1) + 1).float().mean().item()
    return float(hit), float(uniq)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/Base_Best_Dim4096/tri_clip_epoch_10.pt")
    ap.add_argument("--store-db", default="data/embeddings/embeddings.sqlite")
    ap.add_argument("--target-resolutions", default="7,8,9")
    ap.add_argument("--ancestor-res", type=int, default=7)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=67)
    ap.add_argument("--max-eval-samples", type=int, default=1024)
    ap.add_argument("--embedding-sample-strategy", default="mean")
    ap.add_argument("--out-json", default="outputs/chained_retrieval/comparison_grid.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_resolutions = tuple(sorted({int(x) for x in args.target_resolutions.split(",") if x.strip()}))

    store = EmbeddingStore(args.store_db, read_only=True, initialize_schema=False)
    try:
        dataset = H3TripleDataset(
            store=store, target_resolutions=target_resolutions, ancestor_resolutions=(args.ancestor_res,),
            require_modalities=("text", "image", "graph"), embedding_sample_strategy=args.embedding_sample_strategy,
        )
        print(f"[dataset] {len(dataset)} cells")
        _, val_indices = split_indices(len(dataset), args.val_split, args.seed)
        pool = val_indices[: args.max_eval_samples]
        print(f"[pool] val set = {len(val_indices)}; using first {len(pool)} (training eval pool)")
        model = _build_model_from_checkpoint(Path(args.checkpoint), dataset, device=device, use_ema=True)
        feats, present = encode_pool(dataset, pool, model, device)
    finally:
        store.close()

    for m in ALL_MODALITIES:
        assert bool(present[m].all()), f"{m} not fully present"
    feat = {"graph": feats["graph"], "text": feats["text"], "image": feats["image"]}
    n = feat["graph"].shape[0]
    d_max = min(D_MAX, n)

    sim = {
        ("image", "text"): feat["image"] @ feat["text"].T,
        ("text", "image"): feat["text"] @ feat["image"].T,
        ("image", "graph"): feat["image"] @ feat["graph"].T,
        ("graph", "text"): feat["graph"] @ feat["text"].T,
        ("text", "graph"): feat["text"] @ feat["graph"].T,
        ("graph", "image"): feat["graph"] @ feat["image"].T,
    }

    # naive recall curves (R@D for D=1..d_max) for all directions
    naive_curve = {f"{s}->{t}": naive_recall_curve(naive_rank(m), d_max) for (s, t), m in sim.items()}

    chains = {
        "image->graph->text": (("image", "graph"), ("graph", "text"), "image->text"),
        "text->graph->image": (("text", "graph"), ("graph", "image"), "text->image"),
    }

    chained: dict[str, dict] = {}
    for name, (first, second, _b) in chains.items():
        chained[name] = {}
        for a in A_VALUES:
            for b in B_VALUES:
                recall, ncand = chained_recall(sim[first], sim[second], a, b)
                chained[name][f"{a}x{b}"] = {
                    "a": a, "b": b, "budget": a * b, "recall": recall, "mean_unique_candidates": ncand,
                }

    # ---------- report ----------
    def pct(x):
        return f"{100 * x:5.1f}"

    print(f"\n=============== POOL = {n} cells ===============")
    print("\n--- NAIVE R@D (selected D) ---")
    sel_d = [1, 2, 3, 4, 5, 9, 10, 15, 20, 25, 50]
    print(f"{'direction':<16}" + "".join(f"{f'@{d}':>7}" for d in sel_d))
    for d in ["image->text", "text->image", "image->graph", "graph->text", "text->graph", "graph->image"]:
        print(f"{d:<16}" + "".join(f"{pct(naive_curve[d][dd]):>7}" for dd in sel_d))

    for name, (first, second, baseline) in chains.items():
        print(f"\n--- CHAINED {name}  : R@A@B recall grid (%)   [direct baseline {baseline}] ---")
        print(f"{'A\\B':<5}" + "".join(f"{b:>8}" for b in B_VALUES))
        for a in A_VALUES:
            print(f"{a:<5}" + "".join(f"{pct(chained[name][f'{a}x{b}']['recall']):>8}" for b in B_VALUES))

        print(f"\n--- CHAINED {name} vs naive R@(A*B) matched budget : delta (chained - naive) in points ---")
        print(f"{'A\\B':<5}" + "".join(f"{b:>8}" for b in B_VALUES))
        for a in A_VALUES:
            row = ""
            for b in B_VALUES:
                c = chained[name][f"{a}x{b}"]
                budget = min(c["budget"], d_max)
                delta = 100 * (c["recall"] - naive_curve[baseline][budget])
                row += f"{delta:>+8.1f}"
            print(f"{a:<5}" + row)

    report = {
        "checkpoint": args.checkpoint, "pool_size": n, "val_set_size": len(val_indices),
        "config": {
            "seed": args.seed, "val_split": args.val_split, "max_eval_samples": args.max_eval_samples,
            "target_resolutions": list(target_resolutions),
            "embedding_sample_strategy": args.embedding_sample_strategy, "use_ema": True,
            "A_values": A_VALUES, "B_values": B_VALUES, "d_max": d_max,
        },
        "naive_curve": naive_curve,
        "chained": chained,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
