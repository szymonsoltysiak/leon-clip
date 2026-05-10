"""CLI for embedding alignment analysis.

Usage examples:
    python eval_alignment.py --modality graph --stage pre --plot
    python eval_alignment.py --modality graph --stage post --checkpoint checkpoints/final_sota.pt --plot
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import h3

from alignment.geometry import covariance_eigstats, uniformity_gaussian, haversine_distance_matrix, geographic_correlation
from alignment.visualize import plot_spectrum, geographic_density_and_trend
from triple_encoder import EmbeddingStore, H3TripleDataset
from triple_encoder.benchmarks.runner import (
    _build_model_from_checkpoint,
    _collect_post_projection_features,
    _collect_pre_projection_features,
)


def ensure_out_dir():
    out = os.path.join("outputs", "embeddings_eval")
    os.makedirs(out, exist_ok=True)
    return out


def _h3_coords(h3_ids):
    coords = []
    for h3_id in h3_ids:
        lat, lon = h3.cell_to_latlng(str(h3_id))
        coords.append((lat, lon))
    return np.asarray(coords, dtype=np.float32)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["pre", "post"], default="post")
    p.add_argument("--modality", choices=["image", "text", "graph", "all"], default="graph")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--embeddings-root", default="data")
    p.add_argument("--store-db", default="data/embeddings/embeddings.sqlite")
    p.add_argument("--rebuild-store", action="store_true")
    p.add_argument("--checkpoint", type=str, default="", help="Path to TriModalCLIP checkpoint required for post stage")
    p.add_argument("--max-samples", type=int, default=30000)
    p.add_argument("--sample-strategy", choices=["first", "random"], default="random")
    args = p.parse_args()

    outdir = ensure_out_dir()

    store_path = Path(args.store_db)
    use_read_only = store_path.exists() and not args.rebuild_store
    with EmbeddingStore(store_path, read_only=use_read_only, initialize_schema=not use_read_only) as store:
        if args.rebuild_store:
            store.build_from_embeddings_root(args.embeddings_root, rebuild=True)
        elif not store_path.exists():
            store.build_from_embeddings_root(args.embeddings_root, rebuild=False)

        selected_modalities = ["graph", "text", "image"] if args.modality == "all" else [args.modality]
        dataset = H3TripleDataset(store=store, active_modalities=tuple(selected_modalities))
        if len(dataset) == 0:
            raise RuntimeError(f"No H3 samples found under {args.embeddings_root}")

        presence = dataset._exact_presence_map
        h3_values = [
            h for h in dataset.h3_ids
            if all(presence[m][h] for m in selected_modalities)
        ]
        if not h3_values:
            raise RuntimeError(
                f"No H3 cells have all of {selected_modalities} present; nothing to evaluate"
            )
        print(
            f"Filtered to {len(h3_values)} cells with {'+'.join(selected_modalities)} present "
            f"(of {len(dataset.h3_ids)})"
        )
        if len(h3_values) > args.max_samples:
            if args.sample_strategy == "random":
                rng = random.Random(42)
                h3_values = rng.sample(h3_values, args.max_samples)
            else:
                h3_values = h3_values[: args.max_samples]

        model = None
        if args.stage == "post":
            if not args.checkpoint:
                raise ValueError("--checkpoint is required for --stage post")
            model = _build_model_from_checkpoint(Path(args.checkpoint), dataset, device=torch.device("cpu"), use_ema=True)

        data = {}
        if args.modality == "all":
            # treat all selected modalities as one concatenated group
            if args.stage == "pre":
                collected, _ = _collect_pre_projection_features(
                    dataset,
                    h3_values,
                    selected_modalities,
                    progress_desc="Loading pre embeddings (all)",
                )
            else:
                collected, _ = _collect_post_projection_features(
                    dataset,
                    h3_values,
                    selected_modalities,
                    model=model,
                    device=torch.device("cpu"),
                    batch_size=256,
                    progress_desc="Loading post embeddings (all)",
                )
            data["all"] = collected
        else:
            for modality in selected_modalities:
                if args.stage == "pre":
                    collected, _ = _collect_pre_projection_features(
                        dataset,
                        h3_values,
                        [modality],
                        progress_desc=f"Loading pre embeddings ({modality})",
                    )
                else:
                    collected, _ = _collect_post_projection_features(
                        dataset,
                        h3_values,
                        [modality],
                        model=model,
                        device=torch.device("cpu"),
                        batch_size=256,
                        progress_desc=f"Loading post embeddings ({modality})",
                    )
                data[modality] = collected

    # For each modality compute covariance/eigstats and uniformity
    summary = {}
    for m, X in data.items():
        if X is None:
            print(f"[skip] modality {m} has no data")
            continue
        X = np.asarray(X)
        stats = covariance_eigstats(X)
        unif = uniformity_gaussian(X)
        summary[m] = {"stats": stats, "uniformity": unif}
        uniformity_text = ", ".join(f"tau={tau:g}:{value:.4f}" for tau, value in unif.items())
        print(
            f"Modality={m}: isotropy={stats['isotropy']:.3e}, PR={stats['participation_ratio']:.2f}, "
            f"uniformity=({uniformity_text})"
        )
        if args.plot:
            fname = f"{m}_{args.stage}_projection_spectrum.png"
            plot_spectrum(stats["eigenvalues"], stats, os.path.join(outdir, fname))

    for target_key, embeddings in data.items():
        if embeddings is not None:
            target_h3_ids = h3_values[: len(embeddings)]
            coords = _h3_coords(target_h3_ids)
            geo = geographic_correlation(embeddings, coords)
            summary.setdefault(target_key, {})["geo_corr"] = geo
            print(f"Geo correlation ({target_key}): pearson={geo['pearson']:.3f}, spearman={geo['spearman']:.3f}")
            if args.plot:
                D = haversine_distance_matrix(coords)
                normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
                S = normed @ normed.T
                iu = np.triu_indices(D.shape[0], k=1)
                geographic_density_and_trend(D[iu], S[iu], os.path.join(outdir, f"{target_key}_{args.stage}_geo_density_trend.png"))

    metrics_path = os.path.join(outdir, f"{args.modality}_{args.stage}_alignment_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, indent=2, sort_keys=True)
    print(f"Saved metrics to {metrics_path}")

    print("Done. Artifacts saved to outputs/retrieval/ when --plot is used.")


if __name__ == "__main__":
    main()
