from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TriModalCLIP(nn.Module):
    def __init__(
        self,
        text_in_dim: int,
        image_in_dim: int,
        graph_in_dim: int,
        proj_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.text_head = ProjectionHead(text_in_dim, proj_dim, hidden_dim)
        self.image_head = ProjectionHead(image_in_dim, proj_dim, hidden_dim)
        self.graph_head = ProjectionHead(graph_in_dim, proj_dim, hidden_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
        self,
        graph_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        graph_proj = F.normalize(self.graph_head(graph_embedding), dim=-1)
        text_proj = F.normalize(self.text_head(text_embedding), dim=-1)
        image_proj = F.normalize(self.image_head(image_embedding), dim=-1)

        return {
            "graph": graph_proj,
            "text": text_proj,
            "image": image_proj,
            "logit_scale": self.logit_scale.exp(),
        }
