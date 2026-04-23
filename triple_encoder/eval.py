from __future__ import annotations

from typing import Dict, Iterable, Mapping

import torch

from .losses import build_positive_mask


def _compute_direction_metrics(
    sim: torch.Tensor,
    positive_mask: torch.Tensor,
    query_valid: torch.Tensor,
    target_valid: torch.Tensor,
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
    if not has_any.any():
        return {
            "recall@1": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "median_rank": 0.0,
            "mrr": 0.0,
            "queries": 0.0,
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

    return _compute_direction_metrics(sim, positive_mask, source_present.bool(), target_present.bool())


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
