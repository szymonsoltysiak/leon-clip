#!/usr/bin/env python3
"""Compare naive cross-modal retrieval vs. chained (multi-hop) retrieval.

Reproduces the exact validation pool used for the Base_Best_Dim4096 checkpoint
(seed=67, val_split=0.1, first max_eval_samples cells, EMA weights) and contrasts:

  * Naive   Image -> Text          (direct sim ranking)
  * Naive   Text  -> Image
  * Chained Image -> Graph -> Text  (top-a graph hops, top-b text each, union)
  * Chained Text  -> Graph -> Image

For chained retrieval we report R@a@b: take the top-`a` results of the first hop,
run a second search from each of those `a` items keeping top-`b`, union the final
candidates, and count a hit if the diagonal positive (same H3 cell) is in the union.
The fair naive baseline for R@a@b is naive R@(a*b) since both inspect that many
final candidates.
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


def split_indices(length: int, val_split: float, seed: int) -> tuple[list[int], list[int]]:
    """Identical to train.split_indices."""
    indices = list(range(length))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = int(round(length * val_split)) if val_split > 0 else 0
    val_count = min(max(0, val_count), max(0, length - 1))
    return indices[val_count:], indices[:val_count]


@torch.no_grad()
def encode_pool(dataset, pool_indices, model, device, batch_size=256):
    """Encode every pool cell into the shared space for all three modalities."""
    items = [dataset[i] for i in pool_indices]
    out = {m: [] for m in ALL_MODALITIES}
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        batch = {
            f"{m}_embedding": torch.stack([row[f"{m}_embedding"] for row in chunk]).to(device)
            for m in ALL_MODALITIES
        }
        present = {
            f"{m}_present": torch.stack([row[f"{m}_present"] for row in chunk]).to(device)
            for m in ALL_MODALITIES
        }
        res = model(
            graph_embedding=batch["graph_embedding"],
            text_embedding=batch["text_embedding"],
            image_embedding=batch["image_embedding"],
            graph_present=present["graph_present"],
            text_present=present["text_present"],
            image_present=present["image_present"],
        )
        for m in ALL_MODALITIES:
            out[m].append(res[m].detach())
    feats = {m: torch.cat(out[m], dim=0) for m in ALL_MODALITIES}
    present_all = {
        m: torch.stack([row[f"{m}_present"] for row in items]).to(device) for m in ALL_MODALITIES
    }
    return feats, present_all


def naive_recall(sim: torch.Tensor, ks: list[int]) -> dict[int, float]:
    """Diagonal-positive recall@k for a [N,N] similarity matrix."""
    n = sim.shape[0]
    order = torch.argsort(sim, dim=1, descending=True)
    targets = torch.arange(n, device=sim.device).view(-1, 1)
    # rank of the diagonal positive per row (1-indexed)
    rank = (order == targets).float().argmax(dim=1) + 1
    return {k: float((rank <= k).float().mean().item()) for k in ks}


def chained_recall(
    sim_first: torch.Tensor,  # query -> intermediate  [N, N]
    sim_second: torch.Tensor,  # intermediate -> target [N, N]
    a: int,
    b: int,
) -> tuple[float, float]:
    """R@a@b plus mean unique-candidate count.

    For query i: take top-a intermediate indices from sim_first[i]; from each,
    take top-b target indices from sim_second; union them; hit if i is in the union.
    """
    n = sim_first.shape[0]
    a = min(a, n)
    b = min(b, n)
    top_inter = torch.topk(sim_first, a, dim=1).indices  # [N, a]
    top_targets_per_inter = torch.topk(sim_second, b, dim=1).indices  # [N, b]

    hits = 0
    cand_counts = 0
    for i in range(n):
        cands: set[int] = set()
        for inter in top_inter[i].tolist():
            cands.update(top_targets_per_inter[inter].tolist())
        cand_counts += len(cands)
        if i in cands:
            hits += 1
    return hits / n, cand_counts / n


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
    ap.add_argument("--out-json", default="outputs/chained_retrieval/comparison.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_resolutions = tuple(sorted({int(x) for x in args.target_resolutions.split(",") if x.strip()}))

    store = EmbeddingStore(args.store_db, read_only=True, initialize_schema=False)
    try:
        dataset = H3TripleDataset(
            store=store,
            target_resolutions=target_resolutions,
            ancestor_resolutions=(args.ancestor_res,),
            require_modalities=("text", "image", "graph"),
            embedding_sample_strategy=args.embedding_sample_strategy,
        )
        print(f"[dataset] {len(dataset)} cells (require text+image+graph)")

        _, val_indices = split_indices(len(dataset), args.val_split, args.seed)
        pool = val_indices[: args.max_eval_samples]
        print(f"[pool] val set = {len(val_indices)} cells; using first {len(pool)} (== training eval pool)")

        model = _build_model_from_checkpoint(Path(args.checkpoint), dataset, device=device, use_ema=True)
        feats, present = encode_pool(dataset, pool, model, device)
    finally:
        store.close()

    # sanity: all modalities present for every pool cell
    for m in ALL_MODALITIES:
        assert bool(present[m].all()), f"{m} not fully present in pool"

    g, t, im = feats["graph"], feats["text"], feats["image"]
    n = g.shape[0]

    # similarity matrices (cosine; embeddings already L2-normalized)
    sim = {
        ("image", "text"): im @ t.T,
        ("text", "image"): t @ im.T,
        ("image", "graph"): im @ g.T,
        ("graph", "text"): g @ t.T,
        ("text", "graph"): t @ g.T,
        ("graph", "image"): g @ im.T,
    }

    ks = [1, 2, 5, 10]
    naive = {f"{s}->{tg}": naive_recall(sim[(s, tg)], ks) for (s, tg) in sim}

    # chained grid: (a, b) -> budget a*b
    grid = [(1, 5), (2, 5), (1, 10), (2, 2), (5, 2), (10, 1), (3, 3), (5, 5)]
    chains = {
        "image->graph->text": (("image", "graph"), ("graph", "text")),
        "text->graph->image": (("text", "graph"), ("graph", "image")),
    }
    chained: dict[str, dict] = {}
    for name, (first, second) in chains.items():
        chained[name] = {}
        for (a, b) in grid:
            recall, ncand = chained_recall(sim[first], sim[second], a, b)
            chained[name][f"R@{a}@{b}"] = {"recall": recall, "budget": a * b, "mean_unique_candidates": ncand}

    # ---- pretty print ----
    def pct(x):
        return f"{100 * x:5.1f}%"

    print("\n================ NAIVE CROSS-MODAL RETRIEVAL (diagonal positive) ================")
    print(f"{'direction':<18}" + "".join(f"  R@{k:<5}" for k in ks))
    for d in ["image->text", "text->image", "image->graph", "graph->text", "text->graph", "graph->image"]:
        print(f"{d:<18}" + "".join(f"  {pct(naive[d][k])}" for k in ks))

    print("\n================ CHAINED vs NAIVE (matched candidate budget) ================")
    for name, (first, second) in chains.items():
        first_dir = f"{first[0]}->{first[1]}"
        # the naive baseline retrieves directly from query modality -> final target modality
        query_mod, target_mod = first[0], second[1]
        naive_dir = f"{query_mod}->{target_mod}"
        print(f"\n--- {name}   (naive baseline: {naive_dir}) ---")
        print(f"{'metric':<10}{'budget':>8}{'chained':>12}{'naive R@budget':>18}{'delta':>10}{'mean#cand':>12}")
        for (a, b) in grid:
            key = f"R@{a}@{b}"
            c = chained[name][key]
            budget = c["budget"]
            naive_b = naive[naive_dir].get(budget)
            naive_str = pct(naive_b) if naive_b is not None else "  n/a "
            delta = (c["recall"] - naive_b) if naive_b is not None else None
            delta_str = f"{100*delta:+5.1f}" if delta is not None else "  n/a"
            print(f"{key:<10}{budget:>8}{pct(c['recall']):>12}{naive_str:>18}{delta_str:>10}{c['mean_unique_candidates']:>12.1f}")

    report = {
        "checkpoint": args.checkpoint,
        "pool_size": n,
        "val_set_size": len(val_indices),
        "config": {
            "seed": args.seed,
            "val_split": args.val_split,
            "max_eval_samples": args.max_eval_samples,
            "target_resolutions": list(target_resolutions),
            "embedding_sample_strategy": args.embedding_sample_strategy,
            "use_ema": True,
        },
        "naive": naive,
        "chained": chained,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
