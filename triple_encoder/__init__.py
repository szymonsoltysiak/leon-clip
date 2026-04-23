"""Tri-modal CLIP-style training package."""

from .dataloader import H3TripleDataset, ImageStratifiedBatchSampler, build_dataloader
from .eval import compute_retrieval_metrics, evaluate_all_pairs
from .losses import HardNegativeConfig, LossWeights, PositiveMaskConfig, PresenceNormalizationConfig, TriModalContrastiveLoss
from .model import HeadConfig, TriModalCLIP
from .store import EmbeddingStore

__all__ = [
    "EmbeddingStore",
    "H3TripleDataset",
    "ImageStratifiedBatchSampler",
    "HeadConfig",
    "HardNegativeConfig",
    "LossWeights",
    "PositiveMaskConfig",
    "PresenceNormalizationConfig",
    "TriModalCLIP",
    "TriModalContrastiveLoss",
    "build_dataloader",
    "compute_retrieval_metrics",
    "evaluate_all_pairs",
]
