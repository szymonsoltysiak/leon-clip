from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

import h3
import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    """Relative weights for the three pairwise contrastive objectives."""

    graph_text: float = 1.0
    graph_image: float = 0.3
    text_image: float = 0.3


@dataclass
class PositiveMaskConfig:
    loss_mode: str = "multi_positive"
    allow_cross_resolution_positives: bool = True
    cross_resolution_mode: str = "shared_ancestor"
    cross_resolution_ancestor_res: int = 7


@dataclass
class HardNegativeConfig:
    mode: str = "shared_parent"
    weight: float = 1.5
    radius: int = 1
    parent_res_delta: int = 1
    enable: bool = True


@dataclass
class PresenceNormalizationConfig:
    enabled: bool = True
    mode: str = "mean_valid"


def _safe_is_valid_cell(cell: str) -> bool:
    try:
        return bool(h3.is_valid_cell(cell))
    except Exception:
        return False


def _ancestor_at_resolution(cell: str, resolution: int) -> str | None:
    if not _safe_is_valid_cell(cell):
        return None
    cell_res = int(h3.get_resolution(cell))
    if resolution > cell_res:
        return None
    if resolution == cell_res:
        return cell
    return str(h3.cell_to_parent(cell, resolution))


def build_positive_mask(
    h3_ids: list[str],
    *,
    loss_mode: str,
    allow_cross_resolution_positives: bool,
    cross_resolution_mode: str,
    cross_resolution_ancestor_res: int,
    precomputed_ancestors: Mapping[int, Iterable[str | None]] | None = None,
) -> torch.Tensor:
    """Build an [N, N] positive mask for in-batch retrieval objectives."""
    n = len(h3_ids)
    mask = torch.eye(n, dtype=torch.bool)

    if loss_mode == "single_positive":
        return mask
    if loss_mode != "multi_positive":
        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

    same_anchor = torch.zeros((n, n), dtype=torch.bool)
    for i in range(n):
        for j in range(i, n):
            is_pos = h3_ids[i] == h3_ids[j]
            same_anchor[i, j] = is_pos
            same_anchor[j, i] = is_pos
    mask = mask | same_anchor

    if not allow_cross_resolution_positives:
        return mask

    if cross_resolution_mode == "parent_child":
        for i in range(n):
            for j in range(i + 1, n):
                if not (_safe_is_valid_cell(h3_ids[i]) and _safe_is_valid_cell(h3_ids[j])):
                    continue
                a = h3_ids[i]
                b = h3_ids[j]
                ra = int(h3.get_resolution(a))
                rb = int(h3.get_resolution(b))
                if ra == rb:
                    continue
                if ra > rb:
                    parent = h3.cell_to_parent(a, rb)
                    is_pos = parent == b
                else:
                    parent = h3.cell_to_parent(b, ra)
                    is_pos = parent == a
                if is_pos:
                    mask[i, j] = True
                    mask[j, i] = True
    elif cross_resolution_mode == "shared_ancestor":
        ancestor_values: list[str | None]
        if precomputed_ancestors is not None and cross_resolution_ancestor_res in precomputed_ancestors:
            ancestor_values = [item for item in precomputed_ancestors[cross_resolution_ancestor_res]]
        else:
            ancestor_values = [_ancestor_at_resolution(cell, cross_resolution_ancestor_res) for cell in h3_ids]
        for i in range(n):
            for j in range(i + 1, n):
                ai = ancestor_values[i]
                aj = ancestor_values[j]
                if ai is not None and ai == aj:
                    mask[i, j] = True
                    mask[j, i] = True
    elif cross_resolution_mode == "exact":
        pass
    else:
        raise ValueError(f"Unsupported cross_resolution_mode: {cross_resolution_mode}")

    return mask


def build_hard_negative_mask(
    h3_ids: list[str],
    *,
    mode: str,
    radius: int,
    parent_res_delta: int,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    """Build [N, N] mask selecting negatives to upweight in denominator."""
    n = len(h3_ids)
    hard = torch.zeros((n, n), dtype=torch.bool)

    if mode == "none":
        return hard

    for i in range(n):
        a = h3_ids[i]
        if not _safe_is_valid_cell(a):
            continue
        res_a = int(h3.get_resolution(a))
        for j in range(i + 1, n):
            b = h3_ids[j]
            if not _safe_is_valid_cell(b):
                continue
            if positive_mask[i, j]:
                continue

            res_b = int(h3.get_resolution(b))
            same_res = res_a == res_b
            is_hard = False

            if mode in {"same_resolution", "combined"}:
                is_hard = is_hard or (same_res and a != b)

            if mode in {"shared_parent", "combined"}:
                if same_res and res_a >= parent_res_delta and res_a - parent_res_delta >= 0:
                    pa = h3.cell_to_parent(a, res_a - parent_res_delta)
                    pb = h3.cell_to_parent(b, res_b - parent_res_delta)
                    is_hard = is_hard or (pa == pb and a != b)

            if mode in {"k_ring", "combined"} and same_res:
                try:
                    distance = h3.grid_distance(a, b)
                    is_hard = is_hard or (distance is not None and distance <= radius)
                except Exception:
                    pass

            if is_hard:
                hard[i, j] = True
                hard[j, i] = True

    return hard


def multi_positive_clip_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    row_valid_mask: torch.Tensor,
    negative_weights: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, int, int]:
    """Compute row-wise multi-positive InfoNCE with masked log-sum-exp.

    Returns loss, number of valid anchors, and number of positive pairs used.
    """
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("logits must be square [N, N]")

    n = logits.shape[0]
    valid_rows = row_valid_mask.bool().clone()
    pos = positive_mask.bool().clone()

    # A row contributes only if it has at least one positive among valid columns.
    valid_col = row_valid_mask.bool().view(1, n)
    valid_pos = pos & valid_col
    has_positive = valid_pos.any(dim=1)
    valid_rows = valid_rows & has_positive

    if not valid_rows.any():
        return logits.sum() * 0.0, 0, 0

    if negative_weights is None:
        neg_weights = torch.ones_like(logits)
    else:
        neg_weights = negative_weights.to(dtype=logits.dtype)

    # Positive terms keep unit weight in the denominator.
    den_weights = torch.where(pos, torch.ones_like(neg_weights), neg_weights)
    den_weights = torch.clamp(den_weights, min=eps)
    log_den_weights = torch.log(den_weights)

    pos_logits = logits.masked_fill(~valid_pos, float("-inf"))
    weighted_logits = logits + log_den_weights
    weighted_logits = weighted_logits.masked_fill(~valid_col, float("-inf"))

    log_num = torch.logsumexp(pos_logits, dim=1)
    log_den = torch.logsumexp(weighted_logits, dim=1)
    per_row = -(log_num - log_den)
    used = per_row[valid_rows]

    finite = torch.isfinite(used)
    if not finite.any():
        return logits.sum() * 0.0, 0, 0
    used = used[finite]

    pos_count = int((valid_pos & valid_rows.view(-1, 1)).sum().item())
    return used.mean(), int(used.shape[0]), pos_count


class TriModalContrastiveLoss:
    """Symmetric pairwise contrastive loss over the three modality pairs."""

    def __init__(
        self,
        weights: LossWeights,
        positive_cfg: PositiveMaskConfig | None = None,
        hard_negative_cfg: HardNegativeConfig | None = None,
        presence_cfg: PresenceNormalizationConfig | None = None,
    ) -> None:
        self.weights = weights
        self.positive_cfg = positive_cfg or PositiveMaskConfig()
        self.hard_negative_cfg = hard_negative_cfg or HardNegativeConfig()
        self.presence_cfg = presence_cfg or PresenceNormalizationConfig()

    @staticmethod
    def _safe_zero(*tensors: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((), device=tensors[0].device, dtype=tensors[0].dtype)
        for t in tensors:
            out = out + t.sum() * 0.0
        return out

    def _pair_loss(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        valid_mask: torch.Tensor,
        logit_scale: torch.Tensor,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        n = int(a.shape[0])
        if n < 2:
            return self._safe_zero(a, b), {
                "valid_anchors": 0.0,
                "valid_positives": 0.0,
                "multi_positive_matches": 0.0,
                "hard_negative_count": 0.0,
            }

        logits = (a @ b.T) * logit_scale

        neg_weights: torch.Tensor | None = None
        hard_negative_count = 0
        if hard_negative_mask is not None:
            hard_negative_count = int(hard_negative_mask.sum().item())
            neg_weights = torch.ones_like(logits)
            neg_weights = torch.where(
                hard_negative_mask,
                torch.full_like(neg_weights, float(max(1.0, self.hard_negative_cfg.weight))),
                neg_weights,
            )

        loss_ab, anchors_ab, positives_ab = multi_positive_clip_loss(
            logits,
            positive_mask=positive_mask,
            row_valid_mask=valid_mask,
            negative_weights=neg_weights,
        )
        loss_ba, anchors_ba, positives_ba = multi_positive_clip_loss(
            logits.T,
            positive_mask=positive_mask.T,
            row_valid_mask=valid_mask,
            negative_weights=neg_weights.T if neg_weights is not None else None,
        )

        pair_loss = 0.5 * (loss_ab + loss_ba)
        total_valid = int(valid_mask.sum().item())
        valid_anchors = 0.5 * (anchors_ab + anchors_ba)

        if self.presence_cfg.enabled and self.presence_cfg.mode == "proportional" and total_valid > 0:
            pair_loss = pair_loss * (float(valid_anchors) / float(total_valid))

        metrics = {
            "valid_anchors": float(valid_anchors),
            "valid_positives": float(0.5 * (positives_ab + positives_ba)),
            "multi_positive_matches": float(max(0, int(positive_mask.sum().item()) - n)),
            "hard_negative_count": float(hard_negative_count),
        }
        return pair_loss, metrics

    def __call__(
        self,
        model_out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        graph = model_out["graph"]
        text = model_out["text"]
        image = model_out["image"]
        logit_scale = model_out["logit_scale"]

        h3_ids = [str(x) for x in batch["h3"]]
        ancestors: dict[int, list[str | None]] = {}
        for key, value in batch.items():
            if key.startswith("h3_parent_"):
                resolution = int(key.replace("h3_parent_", ""))
                ancestors[resolution] = [str(item) if str(item) else None for item in value]

        positive_mask = build_positive_mask(
            h3_ids,
            loss_mode=self.positive_cfg.loss_mode,
            allow_cross_resolution_positives=self.positive_cfg.allow_cross_resolution_positives,
            cross_resolution_mode=self.positive_cfg.cross_resolution_mode,
            cross_resolution_ancestor_res=self.positive_cfg.cross_resolution_ancestor_res,
            precomputed_ancestors=ancestors,
        ).to(device=graph.device)

        hard_negative_mask: torch.Tensor | None = None
        if self.hard_negative_cfg.enable and self.hard_negative_cfg.mode != "none":
            hard_negative_mask = build_hard_negative_mask(
                h3_ids,
                mode=self.hard_negative_cfg.mode,
                radius=self.hard_negative_cfg.radius,
                parent_res_delta=self.hard_negative_cfg.parent_res_delta,
                positive_mask=positive_mask.cpu(),
            ).to(device=graph.device)

        graph_present = batch["graph_present"].bool()
        text_present = batch["text_present"].bool()
        image_present = batch["image_present"].bool()

        loss_gt, gt_stats = self._pair_loss(
            graph,
            text,
            graph_present & text_present,
            logit_scale,
            positive_mask=positive_mask,
            hard_negative_mask=hard_negative_mask,
        )
        loss_gi, gi_stats = self._pair_loss(
            graph,
            image,
            graph_present & image_present,
            logit_scale,
            positive_mask=positive_mask,
            hard_negative_mask=hard_negative_mask,
        )
        loss_ti, ti_stats = self._pair_loss(
            text,
            image,
            text_present & image_present,
            logit_scale,
            positive_mask=positive_mask,
            hard_negative_mask=hard_negative_mask,
        )

        total = (
            self.weights.graph_text * loss_gt
            + self.weights.graph_image * loss_gi
            + self.weights.text_image * loss_ti
        )

        metrics = {
            "loss_graph_text": float(loss_gt.detach().item()),
            "loss_graph_image": float(loss_gi.detach().item()),
            "loss_text_image": float(loss_ti.detach().item()),
            "valid_anchors_gt": gt_stats["valid_anchors"],
            "valid_anchors_gi": gi_stats["valid_anchors"],
            "valid_anchors_ti": ti_stats["valid_anchors"],
            "valid_positives_gt": gt_stats["valid_positives"],
            "valid_positives_gi": gi_stats["valid_positives"],
            "valid_positives_ti": ti_stats["valid_positives"],
            "multi_positive_matches_gt": gt_stats["multi_positive_matches"],
            "multi_positive_matches_gi": gi_stats["multi_positive_matches"],
            "multi_positive_matches_ti": ti_stats["multi_positive_matches"],
            "hard_negative_count_gt": gt_stats["hard_negative_count"],
            "hard_negative_count_gi": gi_stats["hard_negative_count"],
            "hard_negative_count_ti": ti_stats["hard_negative_count"],
        }
        return total, metrics
