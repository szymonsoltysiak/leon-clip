from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping

import h3
import torch

from .losses import build_positive_mask


def _safe_is_valid_cell(cell: str) -> bool:
    try:
        return bool(h3.is_valid_cell(cell))
    except Exception:
        return False


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in km."""
    earth_radius_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earth_radius_km * c


def _compute_direction_metrics(
    sim: torch.Tensor,
    positive_mask: torch.Tensor,
    query_valid: torch.Tensor,
    target_valid: torch.Tensor,
    h3_ids: list[str],
) -> Dict[str, float]:
    n = int(sim.shape[0])
    valid_target = target_valid.view(1, n).bool()
    row_pos = positive_mask.bool() & valid_target
    row_valid = query_valid.bool() & row_pos.any(dim=1)

    if not row_valid.any():
        return {
            "recall@1": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "median_rank": 0.0,
            "mrr": 0.0,
            "queries": 0.0,
            "mge@1_km": 0.0,
            "mge@5_km": 0.0,
            "mge@10_km": 0.0,
        }

    sim_masked = sim.masked_fill(~valid_target, float("-inf"))
    valid_rows_idx = torch.where(row_valid)[0]
    sim_valid = sim_masked[valid_rows_idx]
    pos_valid = row_pos[valid_rows_idx]

    order = torch.argsort(sim_valid, dim=1, descending=True)
    ranked_pos = torch.gather(pos_valid, dim=1, index=order)

    # First positive rank per row (1-indexed). Row validity already guarantees at least one positive.
    first_rank_zero_idx = ranked_pos.to(dtype=torch.int64).argmax(dim=1)
    has_any = ranked_pos.any(dim=1)

    mge_1_list: list[float] = []
    mge_5_list: list[float] = []
    mge_10_list: list[float] = []

    for row_idx in range(len(valid_rows_idx)):
        max_k = min(10, int(order.shape[1]))
        top_k_indices = order[row_idx, :max_k].tolist()
        pos_target_indices = torch.where(pos_valid[row_idx])[0].tolist()

        if not pos_target_indices:
            continue

        dists_for_query: list[float] = []
        for pred_idx in top_k_indices:
            pred_cell = h3_ids[pred_idx]
            if not _safe_is_valid_cell(pred_cell):
                dists_for_query.append(float("inf"))
                continue

            try:
                pred_lat, pred_lon = h3.cell_to_latlng(pred_cell)
                min_dist_to_gt = float("inf")

                for tgt_idx in pos_target_indices:
                    tgt_cell = h3_ids[tgt_idx]
                    if not _safe_is_valid_cell(tgt_cell):
                        continue
                    tgt_lat, tgt_lon = h3.cell_to_latlng(tgt_cell)
                    dist_km = _haversine_km(pred_lat, pred_lon, tgt_lat, tgt_lon)
                    min_dist_to_gt = min(min_dist_to_gt, dist_km)

                dists_for_query.append(min_dist_to_gt)
            except Exception:
                dists_for_query.append(float("inf"))

        if len(dists_for_query) >= 1:
            best_1 = min(dists_for_query[:1])
            if best_1 != float("inf"):
                mge_1_list.append(best_1)
        if len(dists_for_query) >= 5:
            best_5 = min(dists_for_query[:5])
            if best_5 != float("inf"):
                mge_5_list.append(best_5)
        if len(dists_for_query) >= 10:
            best_10 = min(dists_for_query[:10])
            if best_10 != float("inf"):
                mge_10_list.append(best_10)

    def safe_mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    if not has_any.any():
        return {
            "recall@1": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "median_rank": 0.0,
            "mrr": 0.0,
            "queries": 0.0,
            "mge@1_km": 0.0,
            "mge@5_km": 0.0,
            "mge@10_km": 0.0,
        }

    first_rank = (first_rank_zero_idx[has_any] + 1).to(dtype=torch.float32)
    rr_tensor = 1.0 / first_rank
    return {
        "recall@1": float((first_rank <= 1).to(dtype=torch.float32).mean().item()),
        "recall@5": float((first_rank <= 5).to(dtype=torch.float32).mean().item()),
        "recall@10": float((first_rank <= 10).to(dtype=torch.float32).mean().item()),
        "median_rank": float(first_rank.median().item()),
        "mrr": float(rr_tensor.mean().item()),
        "queries": float(first_rank.numel()),
        "mge@1_km": safe_mean(mge_1_list),
        "mge@5_km": safe_mean(mge_5_list),
        "mge@10_km": safe_mean(mge_10_list),
    }


def compute_retrieval_metrics(
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    source_present: torch.Tensor,
    target_present: torch.Tensor,
    h3_ids: list[str],
    *,
    loss_mode: str,
    allow_cross_resolution_positives: bool,
    cross_resolution_mode: str,
    cross_resolution_ancestor_res: int,
    precomputed_ancestors: Mapping[int, Iterable[str | None]] | None = None,
) -> Dict[str, float]:
    sim = source_embeddings @ target_embeddings.T
    positive_mask = build_positive_mask(
        h3_ids,
        loss_mode=loss_mode,
        allow_cross_resolution_positives=allow_cross_resolution_positives,
        cross_resolution_mode=cross_resolution_mode,
        cross_resolution_ancestor_res=cross_resolution_ancestor_res,
        precomputed_ancestors=precomputed_ancestors,
    ).to(device=sim.device)

    return _compute_direction_metrics(
        sim,
        positive_mask,
        source_present.bool(),
        target_present.bool(),
        h3_ids,
    )


def evaluate_all_pairs(
    embeddings: Dict[str, torch.Tensor],
    presence: Dict[str, torch.Tensor],
    h3_ids: list[str],
    *,
    positive_cfg: Dict[str, object],
    precomputed_ancestors: Mapping[int, Iterable[str | None]] | None = None,
) -> Dict[str, float]:
    pairs = [
        ("graph", "text"),
        ("graph", "image"),
        ("text", "image"),
    ]

    metrics: Dict[str, float] = {}
    for src, tgt in pairs:
        forward = compute_retrieval_metrics(
            embeddings[src],
            embeddings[tgt],
            presence[src],
            presence[tgt],
            h3_ids,
            loss_mode=str(positive_cfg["loss_mode"]),
            allow_cross_resolution_positives=bool(positive_cfg["allow_cross_resolution_positives"]),
            cross_resolution_mode=str(positive_cfg["cross_resolution_mode"]),
            cross_resolution_ancestor_res=int(positive_cfg["cross_resolution_ancestor_res"]),
            precomputed_ancestors=precomputed_ancestors,
        )
        backward = compute_retrieval_metrics(
            embeddings[tgt],
            embeddings[src],
            presence[tgt],
            presence[src],
            h3_ids,
            loss_mode=str(positive_cfg["loss_mode"]),
            allow_cross_resolution_positives=bool(positive_cfg["allow_cross_resolution_positives"]),
            cross_resolution_mode=str(positive_cfg["cross_resolution_mode"]),
            cross_resolution_ancestor_res=int(positive_cfg["cross_resolution_ancestor_res"]),
            precomputed_ancestors=precomputed_ancestors,
        )

        for name, value in forward.items():
            metrics[f"retrieval/{src}_to_{tgt}/{name}"] = value
        for name, value in backward.items():
            metrics[f"retrieval/{tgt}_to_{src}/{name}"] = value

    return metrics
