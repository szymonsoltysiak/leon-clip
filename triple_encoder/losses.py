from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    graph_text: float = 1.0
    graph_image: float = 0.3
    text_image: float = 0.3


class TriModalContrastiveLoss:
    def __init__(self, weights: LossWeights) -> None:
        self.weights = weights

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
    ) -> Tuple[torch.Tensor, int]:
        valid_a = a[valid_mask]
        valid_b = b[valid_mask]
        n = int(valid_a.shape[0])

        if n < 2:
            return self._safe_zero(a, b), n

        logits = (valid_a @ valid_b.T) * logit_scale
        labels = torch.arange(n, device=logits.device)
        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_ab + loss_ba), n

    def __call__(
        self,
        model_out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        graph = model_out["graph"]
        text = model_out["text"]
        image = model_out["image"]
        logit_scale = model_out["logit_scale"]

        graph_present = batch["graph_present"].bool()
        text_present = batch["text_present"].bool()
        image_present = batch["image_present"].bool()

        loss_gt, n_gt = self._pair_loss(graph, text, graph_present & text_present, logit_scale)
        loss_gi, n_gi = self._pair_loss(graph, image, graph_present & image_present, logit_scale)
        loss_ti, n_ti = self._pair_loss(text, image, text_present & image_present, logit_scale)

        total = (
            self.weights.graph_text * loss_gt
            + self.weights.graph_image * loss_gi
            + self.weights.text_image * loss_ti
        )

        metrics = {
            "loss_graph_text": float(loss_gt.detach().item()),
            "loss_graph_image": float(loss_gi.detach().item()),
            "loss_text_image": float(loss_ti.detach().item()),
            "valid_pairs_gt": float(n_gt),
            "valid_pairs_gi": float(n_gi),
            "valid_pairs_ti": float(n_ti),
        }
        return total, metrics
