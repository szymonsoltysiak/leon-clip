from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterator, Optional

import h3
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset

from .store import EmbeddingStore


class H3TripleDataset(Dataset):
    def __init__(
        self,
        store: EmbeddingStore,
        target_resolutions: tuple[int, ...] = (7, 8, 9),
        hierarchy_for: tuple[str, ...] = ("text", "graph"),
        embedding_sample_strategy: str = "random",
        mask_value: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.store = store
        self.target_resolutions = tuple(sorted({int(r) for r in target_resolutions}))
        if not self.target_resolutions:
            raise ValueError("target_resolutions cannot be empty")
        self.hierarchy_for = set(hierarchy_for)
        self.embedding_sample_strategy = str(embedding_sample_strategy)
        if self.embedding_sample_strategy not in {"random", "first", "mean"}:
            raise ValueError("embedding_sample_strategy must be one of: random, first, mean")
        self.mask_value = float(mask_value)
        self.dtype = dtype

        self.text_dim = self.store.get_dim("text") or 1
        self.image_dim = self.store.get_dim("image") or 1
        self.graph_dim = self.store.get_dim("graph") or 1

        self.h3_ids = self.store.get_h3_ids(target_resolutions=self.target_resolutions)

        self._text_h3 = set(self.store.get_h3_ids_for_modality("text", target_resolutions=self.target_resolutions))
        self._image_h3 = set(self.store.get_h3_ids_for_modality("image", target_resolutions=self.target_resolutions))
        self._graph_h3 = set(self.store.get_h3_ids_for_modality("graph", target_resolutions=self.target_resolutions))

        self.has_image_indices = [i for i, h in enumerate(self.h3_ids) if h in self._image_h3]
        self.no_image_indices = [i for i, h in enumerate(self.h3_ids) if h not in self._image_h3]

        self._zero_cache: Dict[tuple[str, int], np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.h3_ids)

    def _zero(self, dim: int, key: str) -> np.ndarray:
        cache_key = (key, dim)
        if cache_key not in self._zero_cache:
            self._zero_cache[cache_key] = np.full((dim,), self.mask_value, dtype=np.float32)
        return self._zero_cache[cache_key]

    def _get_or_zero(self, modality: str, h3_id: str, dim: int) -> np.ndarray:
        vec = self.store.get_embedding(modality, h3_id, strategy=self.embedding_sample_strategy)
        if vec is None:
            return self._zero(dim, f"{modality}_single")
        if vec.shape[0] == dim:
            return vec
        out = self._zero(dim, f"{modality}_single").copy()
        n = min(out.shape[0], vec.shape[0])
        out[:n] = vec[:n]
        return out

    def _hierarchical(self, modality: str, h3_id: str, dim: int) -> np.ndarray:
        cell_res = h3.get_resolution(h3_id)
        parent = h3.cell_to_parent(h3_id, cell_res - 1) if cell_res > 0 else None
        children = list(h3.cell_to_children(h3_id, cell_res + 1)) if cell_res < 15 else []

        vec_parent = self._get_or_zero(modality, parent, dim) if parent is not None else self._zero(dim, f"{modality}_parent")
        vec_center = self._get_or_zero(modality, h3_id, dim)

        child_vecs = [
            self.store.get_embedding(modality, child, strategy=self.embedding_sample_strategy)
            for child in children
        ]
        child_vecs = [v for v in child_vecs if v is not None and v.shape[0] > 0]

        if child_vecs:
            pooled = np.mean(child_vecs, axis=0, dtype=np.float32)
            if pooled.shape[0] != dim:
                fixed = self._zero(dim, f"{modality}_child").copy()
                n = min(dim, pooled.shape[0])
                fixed[:n] = pooled[:n]
                pooled = fixed
        else:
            pooled = self._zero(dim, f"{modality}_child")

        return np.concatenate([vec_parent, vec_center, pooled], axis=0)

    def _hierarchy_present(self, modality: str, h3_id: str) -> bool:
        cell_res = h3.get_resolution(h3_id)
        parent = h3.cell_to_parent(h3_id, cell_res - 1) if cell_res > 0 else None
        children = h3.cell_to_children(h3_id, cell_res + 1) if cell_res < 15 else []
        if self.store.has_embedding(modality, h3_id):
            return True
        if parent is not None and self.store.has_embedding(modality, parent):
            return True
        for child in children:
            if self.store.has_embedding(modality, child):
                return True
        return False

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        h3_id = self.h3_ids[idx]

        if "text" in self.hierarchy_for:
            text_vec = self._hierarchical("text", h3_id, self.text_dim)
        else:
            text_vec = self._get_or_zero("text", h3_id, self.text_dim)

        if "graph" in self.hierarchy_for:
            graph_vec = self._hierarchical("graph", h3_id, self.graph_dim)
        else:
            graph_vec = self._get_or_zero("graph", h3_id, self.graph_dim)

        image_vec = self._get_or_zero("image", h3_id, self.image_dim)

        has_text = self._hierarchy_present("text", h3_id)
        has_image = self._hierarchy_present("image", h3_id)
        has_graph = self._hierarchy_present("graph", h3_id)

        return {
            "h3": h3_id,
            "text_embedding": torch.tensor(text_vec, dtype=self.dtype),
            "image_embedding": torch.tensor(image_vec, dtype=self.dtype),
            "graph_embedding": torch.tensor(graph_vec, dtype=self.dtype),
            "text_present": torch.tensor(1 if has_text else 0, dtype=torch.bool),
            "image_present": torch.tensor(1 if has_image else 0, dtype=torch.bool),
            "graph_present": torch.tensor(1 if has_graph else 0, dtype=torch.bool),
        }


class ImageStratifiedBatchSampler(BatchSampler):
    def __init__(
        self,
        has_image_indices: list[int],
        no_image_indices: list[int],
        batch_size: int,
        min_image_ratio: float = 0.25,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        self.has_image_indices = list(has_image_indices)
        self.no_image_indices = list(no_image_indices)
        self.all_indices = self.has_image_indices + self.no_image_indices
        self.batch_size = int(batch_size)
        self.min_image_ratio = float(min_image_ratio)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.all_indices:
            raise ValueError("sampler received no indices")

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.all_indices) // self.batch_size
        return math.ceil(len(self.all_indices) / self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        num_batches = len(self)

        min_img = int(math.ceil(self.batch_size * self.min_image_ratio))
        min_img = max(0, min(min_img, self.batch_size))

        for _ in range(num_batches):
            batch: list[int] = []

            img_take = min_img
            if not self.has_image_indices:
                img_take = 0

            if img_take > 0:
                if img_take <= len(self.has_image_indices):
                    batch.extend(rng.sample(self.has_image_indices, img_take))
                else:
                    batch.extend(rng.choices(self.has_image_indices, k=img_take))

            remaining = self.batch_size - len(batch)
            pool = self.all_indices
            if remaining > 0:
                if remaining <= len(pool):
                    batch.extend(rng.sample(pool, remaining))
                else:
                    batch.extend(rng.choices(pool, k=remaining))

            rng.shuffle(batch)
            yield batch


def build_dataloader(
    dataset: H3TripleDataset,
    batch_size: int,
    min_image_ratio: float = 0.25,
    num_workers: int = 0,
    drop_last: bool = False,
    seed: int = 42,
) -> DataLoader:
    sampler = ImageStratifiedBatchSampler(
        has_image_indices=dataset.has_image_indices,
        no_image_indices=dataset.no_image_indices,
        batch_size=batch_size,
        min_image_ratio=min_image_ratio,
        drop_last=drop_last,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
