#!/usr/bin/env python3
"""Run example retrievals over modality pairs from a trained TriModalCLIP checkpoint.

For each requested direction (e.g. ``graph -> text``), the script samples a small set
of query H3 cells from a retrieval pool, projects every modality into the shared CLIP
space with the loaded checkpoint, and records the top-k retrieved H3 cells along with
their cosine similarity scores. Output is JSONL plus a flat CSV under
``outputs/retrieval_examples/`` so it can be inspected directly or fed to a later
visualization step.

Each JSONL record has the shape:

    {
      "query_modality": "graph",
      "target_modality": "text",
      "query_h3": "8928308280fffff",
      "query_h3_resolution": 9,
      "query_ancestor_res7": "872830828ffffff",
      "retrieved": [
        {"rank": 1, "h3": "...", "h3_resolution": 9, "similarity": 0.83,
         "is_self_cell": true, "shared_ancestor_res7": true},
        ...
      ]
    }

Example:

    python retrieve_examples.py \
        --checkpoint checkpoints/Base_Best_Dim4096/tri_clip_epoch_10.pt \
        --target-resolutions 9 --pool-size 2000 --num-queries 8 --top-k 10
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import h3
import numpy as np
import torch

from triple_encoder import EmbeddingStore, H3TripleDataset
from triple_encoder.benchmarks.runner import (
    _build_model_from_checkpoint,
    _collect_post_projection_features,
)


MODALITIES = ("graph", "text", "image")
ALL_DIRECTIONS = tuple((src, tgt) for src in MODALITIES for tgt in MODALITIES if src != tgt)


def parse_directions(raw: str) -> list[tuple[str, str]]:
    if raw.strip().lower() == "all":
        return list(ALL_DIRECTIONS)
    pairs: list[tuple[str, str]] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        for sep in ("->", "_to_", "-", ">"):
            if sep in token:
                src, tgt = (part.strip() for part in token.split(sep, 1))
                break
        else:
            raise ValueError(f"Cannot parse direction: {token!r}")
        if src not in MODALITIES or tgt not in MODALITIES or src == tgt:
            raise ValueError(f"Unsupported direction: {token!r}")
        pairs.append((src, tgt))
    return pairs


def parse_resolutions(raw: str) -> tuple[int, ...]:
    values = sorted({int(token.strip()) for token in raw.split(",") if token.strip()})
    if not values:
        raise ValueError("--target-resolutions cannot be empty")
    return tuple(values)


def parse_modality_set(raw: str) -> tuple[str, ...]:
    items = sorted({token.strip().lower() for token in raw.split(",") if token.strip()})
    invalid = [item for item in items if item not in MODALITIES]
    if invalid:
        raise ValueError(f"Unsupported modality values: {invalid}")
    return tuple(items)


def ancestor_at(cell: str, resolution: int) -> str | None:
    try:
        cell_res = int(h3.get_resolution(cell))
    except Exception:
        return None
    if resolution > cell_res:
        return None
    if resolution == cell_res:
        return cell
    try:
        return str(h3.cell_to_parent(cell, resolution))
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embeddings-root", default="data")
    parser.add_argument("--store-db", default="data/embeddings/embeddings.sqlite")
    parser.add_argument("--rebuild-store", action="store_true")
    parser.add_argument("--target-resolutions", default="9")
    parser.add_argument("--directions", default="all",
                        help="Comma-separated 'src->tgt' pairs (e.g. 'graph->text,text->image') or 'all'")
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=2000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ancestor-res", type=int, default=7)
    parser.add_argument("--require-modalities", default="text,image,graph",
                        help="Pool restricted to cells with all of these modalities present; pass empty to disable")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--disable-ema", action="store_true")
    parser.add_argument("--out-dir", default="outputs/retrieval_examples")
    parser.add_argument("--tag", default="", help="Optional suffix for output filenames")
    args = parser.parse_args()

    if args.disable_ema:
        args.use_ema = False

    rng = random.Random(args.seed)
    directions = parse_directions(args.directions)
    target_resolutions = parse_resolutions(args.target_resolutions)
    require_modalities = parse_modality_set(args.require_modalities) if args.require_modalities.strip() else ()
    device = torch.device(args.device)

    store_path = Path(args.store_db)
    use_read_only = store_path.exists() and not args.rebuild_store
    with EmbeddingStore(store_path, read_only=use_read_only, initialize_schema=not use_read_only) as store:
        if args.rebuild_store:
            store.build_from_embeddings_root(args.embeddings_root, rebuild=True)
        elif not store_path.exists():
            store.build_from_embeddings_root(args.embeddings_root, rebuild=False)

        dataset = H3TripleDataset(
            store=store,
            target_resolutions=target_resolutions,
            require_modalities=require_modalities,
        )
        if len(dataset) == 0:
            raise RuntimeError(
                f"Dataset is empty under require_modalities={require_modalities!r}; nothing to retrieve over"
            )

        h3_pool = list(dataset.h3_ids)
        rng.shuffle(h3_pool)
        h3_pool = h3_pool[: args.pool_size]
        print(f"[pool] {len(h3_pool)} cells across resolutions={target_resolutions}")

        model = _build_model_from_checkpoint(
            Path(args.checkpoint), dataset, device=device, use_ema=args.use_ema,
        )

        feature_matrices: dict[str, torch.Tensor] = {}
        keep_per_modality: dict[str, list[int]] = {}
        for modality in MODALITIES:
            features, keep = _collect_post_projection_features(
                dataset,
                h3_pool,
                [modality],
                model=model,
                device=device,
                batch_size=256,
                progress_desc=f"Encoding pool ({modality})",
            )
            if features.size == 0:
                raise RuntimeError(f"Pool produced zero features for modality={modality}")
            feature_matrices[modality] = features
            keep_per_modality[modality] = keep

        common_indices = sorted(
            set(keep_per_modality["graph"])
            & set(keep_per_modality["text"])
            & set(keep_per_modality["image"])
        )
        if not common_indices:
            raise RuntimeError("No cells survived feature collection across all modalities")
        h3_pool_aligned = [h3_pool[idx] for idx in common_indices]

        tensors: dict[str, torch.Tensor] = {}
        for modality in MODALITIES:
            keep_to_row = {pool_idx: row_idx for row_idx, pool_idx in enumerate(keep_per_modality[modality])}
            rows = [keep_to_row[idx] for idx in common_indices]
            tensors[modality] = torch.from_numpy(feature_matrices[modality][rows]).to(device)

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{args.tag}" if args.tag else ""
        jsonl_path = out_dir / f"retrievals{suffix}.jsonl"
        csv_path = out_dir / f"retrievals{suffix}.csv"
        if jsonl_path.exists():
            jsonl_path.unlink()

        total_records = 0
        with jsonl_path.open("w", encoding="utf-8") as jsonl_handle, \
                csv_path.open("w", newline="", encoding="utf-8") as csv_handle:
            writer = csv.writer(csv_handle)
            writer.writerow([
                "query_modality", "target_modality",
                "query_h3", "query_h3_resolution", "query_ancestor_res7",
                "rank", "retrieved_h3", "retrieved_h3_resolution",
                "similarity", "is_self_cell", "shared_ancestor_res7",
            ])

            for src, tgt in directions:
                src_features = tensors[src]
                tgt_features = tensors[tgt]

                positions = list(range(len(h3_pool_aligned)))
                rng.shuffle(positions)
                query_positions = positions[: args.num_queries]

                with torch.no_grad():
                    sim = src_features[query_positions] @ tgt_features.T
                top_k = min(args.top_k, sim.shape[1])
                top_sims, top_idx = torch.topk(sim, top_k, dim=1)
                top_sims_np = top_sims.cpu().numpy()
                top_idx_np = top_idx.cpu().numpy()

                for row, pool_pos in enumerate(query_positions):
                    query_h3 = h3_pool_aligned[pool_pos]
                    query_ancestor = ancestor_at(query_h3, args.ancestor_res)
                    query_res = int(h3.get_resolution(query_h3))
                    retrieved: list[dict] = []
                    for rank, (tgt_pos, score) in enumerate(
                        zip(top_idx_np[row], top_sims_np[row]), start=1,
                    ):
                        retrieved_h3 = h3_pool_aligned[int(tgt_pos)]
                        retrieved_res = int(h3.get_resolution(retrieved_h3))
                        retrieved_ancestor = ancestor_at(retrieved_h3, args.ancestor_res)
                        shared_ancestor = bool(
                            query_ancestor is not None and retrieved_ancestor == query_ancestor
                        )
                        retrieved.append({
                            "rank": rank,
                            "h3": retrieved_h3,
                            "h3_resolution": retrieved_res,
                            "similarity": float(score),
                            "is_self_cell": bool(retrieved_h3 == query_h3),
                            "shared_ancestor_res7": shared_ancestor,
                        })
                        writer.writerow([
                            src, tgt,
                            query_h3, query_res, query_ancestor or "",
                            rank, retrieved_h3, retrieved_res,
                            float(score), int(retrieved_h3 == query_h3), int(shared_ancestor),
                        ])

                    record = {
                        "query_modality": src,
                        "target_modality": tgt,
                        "query_h3": query_h3,
                        "query_h3_resolution": query_res,
                        "query_ancestor_res7": query_ancestor,
                        "retrieved": retrieved,
                    }
                    jsonl_handle.write(json.dumps(record) + "\n")
                    total_records += 1

                print(f"[{src:>5} -> {tgt:<5}] {len(query_positions)} queries x top-{top_k}")

        print(f"\nWrote {total_records} retrieval records")
        print(f"  JSONL: {jsonl_path}")
        print(f"  CSV:   {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
