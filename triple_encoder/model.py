from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HeadConfig:
    head_type: str = "residual_mlp"
    hidden_dim: int = 512
    dropout: float = 0.1
    residual_layers: int = 1


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ProjectionHead(nn.Module):
    """Configurable projection head used by each modality tower."""

    def __init__(self, in_dim: int, out_dim: int, config: HeadConfig) -> None:
        super().__init__()
        self.head_type = config.head_type
        if self.head_type == "linear":
            self.net = nn.Linear(in_dim, out_dim)
        elif self.head_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(in_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, out_dim),
            )
        elif self.head_type == "residual_mlp":
            layers: list[nn.Module] = [
                nn.Linear(in_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ]
            for _ in range(max(0, config.residual_layers)):
                layers.append(ResidualBlock(config.hidden_dim, config.dropout))
            layers.append(nn.Linear(config.hidden_dim, out_dim))
            self.net = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported head_type: {self.head_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TriModalCLIP(nn.Module):
    """Projection model that maps text, image, and graph embeddings into a shared space."""

    def __init__(
        self,
        text_in_dim: int,
        image_in_dim: int,
        graph_in_dim: int,
        proj_dim: int,
        head_config: HeadConfig,
        use_learned_missing_tokens: bool = True,
        max_logit_scale: float = 100.0,
    ) -> None:
        super().__init__()
        self.text_head = ProjectionHead(text_in_dim, proj_dim, head_config)
        self.image_head = ProjectionHead(image_in_dim, proj_dim, head_config)
        self.graph_head = ProjectionHead(graph_in_dim, proj_dim, head_config)

        self.use_learned_missing_tokens = bool(use_learned_missing_tokens)
        self.text_missing_token = nn.Parameter(torch.zeros(text_in_dim))
        self.image_missing_token = nn.Parameter(torch.zeros(image_in_dim))
        self.graph_missing_token = nn.Parameter(torch.zeros(graph_in_dim))

        self.max_logit_scale = float(max_logit_scale)
        self.min_logit_scale = 1.0 / self.max_logit_scale
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def _clamped_logit_scale(self) -> torch.Tensor:
        return torch.clamp(
            self.logit_scale,
            min=math.log(self.min_logit_scale),
            max=math.log(self.max_logit_scale),
        ).exp()

    def _inject_missing(
        self,
        x: torch.Tensor,
        present: torch.Tensor,
        token: torch.Tensor,
    ) -> torch.Tensor:
        present = present.bool().view(-1, 1)
        if self.use_learned_missing_tokens:
            replacement = token.view(1, -1).expand_as(x)
        else:
            replacement = torch.zeros_like(x)
        return torch.where(present, x, replacement)

    def forward(
        self,
        graph_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
        graph_present: torch.Tensor | None = None,
        text_present: torch.Tensor | None = None,
        image_present: torch.Tensor | None = None,
        return_raw: bool = False,
    ) -> dict[str, torch.Tensor]:
        if graph_present is None:
            graph_present = torch.ones(graph_embedding.shape[0], dtype=torch.bool, device=graph_embedding.device)
        if text_present is None:
            text_present = torch.ones(text_embedding.shape[0], dtype=torch.bool, device=text_embedding.device)
        if image_present is None:
            image_present = torch.ones(image_embedding.shape[0], dtype=torch.bool, device=image_embedding.device)

        graph_in = self._inject_missing(graph_embedding, graph_present, self.graph_missing_token)
        text_in = self._inject_missing(text_embedding, text_present, self.text_missing_token)
        image_in = self._inject_missing(image_embedding, image_present, self.image_missing_token)

        graph_raw = self.graph_head(graph_in)
        text_raw = self.text_head(text_in)
        image_raw = self.image_head(image_in)

        graph_proj = F.normalize(graph_raw, dim=-1)
        text_proj = F.normalize(text_raw, dim=-1)
        image_proj = F.normalize(image_raw, dim=-1)

        out = {
            "graph": graph_proj,
            "text": text_proj,
            "image": image_proj,
            "logit_scale": self._clamped_logit_scale(),
        }
        if return_raw:
            out["graph_raw"] = graph_raw
            out["text_raw"] = text_raw
            out["image_raw"] = image_raw
        return out
