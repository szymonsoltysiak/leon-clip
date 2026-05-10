from __future__ import annotations

import math
import random
from functools import partial
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import h3
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset

from .store import EmbeddingStore


class H3TripleDataset(Dataset):
    """Construct tri-modal triples anchored on H3 cells.

    Text and graph features are hierarchical: the returned tensor concatenates
    parent, center, and pooled child embeddings. Image features remain
    non-hierarchical and use the exact center-cell image embedding only.
    """

    def __init__(
        self,
        store: EmbeddingStore,
        target_resolutions: tuple[int, ...] = (7, 8, 9),
        ancestor_resolutions: tuple[int, ...] = (7,),
        hierarchy_for: tuple[str, ...] = ("text", "graph"),
        active_modalities: tuple[str, ...] | None = None,
        require_modalities: tuple[str, ...] = (),
        embedding_sample_strategy: str = "random",
        mask_value: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._store_path = Path(store.db_path)
        self.target_resolutions = tuple(sorted({int(resolution) for resolution in target_resolutions}))
        if not self.target_resolutions:
            raise ValueError("target_resolutions cannot be empty")
        self.ancestor_resolutions = tuple(sorted({int(resolution) for resolution in ancestor_resolutions}))
        for resolution in self.ancestor_resolutions:
            if resolution < 0 or resolution > 15:
                raise ValueError(f"Invalid ancestor resolution: {resolution}")

        self.hierarchy_for = set(hierarchy_for)
        self.active_modalities = set(active_modalities or ("text", "image", "graph"))
        unknown_modalities = self.active_modalities.difference({"text", "image", "graph"})
        if unknown_modalities:
            raise ValueError(f"Unsupported active modalities: {sorted(unknown_modalities)}")
        self.require_modalities = tuple(sorted(set(require_modalities)))
        unknown_required = set(self.require_modalities).difference({"text", "image", "graph"})
        if unknown_required:
            raise ValueError(f"Unsupported required modalities: {sorted(unknown_required)}")
        self.embedding_sample_strategy = str(embedding_sample_strategy)
        if self.embedding_sample_strategy not in {"random", "first", "mean"}:
            raise ValueError("embedding_sample_strategy must be one of: random, first, mean")

        self.mask_value = float(mask_value)
        self.dtype = dtype

        self.text_dim = store.get_dim("text") or 1
        self.image_dim = store.get_dim("image") or 1
        self.graph_dim = store.get_dim("graph") or 1

        self.h3_ids = store.get_h3_ids(target_resolutions=self.target_resolutions)
        self._modality_cells: Dict[str, set[str]] = {
            "text": store.get_cells_for_modality("text"),
            "image": store.get_cells_for_modality("image"),
            "graph": store.get_cells_for_modality("graph"),
        }
        self._h3_resolution: Dict[str, int] = {h3_id: int(h3.get_resolution(h3_id)) for h3_id in self.h3_ids}
        self._parent_map: Dict[str, Optional[str]] = {}
        self._children_map: Dict[str, tuple[str, ...]] = {}
        self._ancestor_map: Dict[str, Dict[int, str]] = {}

        for h3_id in self.h3_ids:
            cell_resolution = self._h3_resolution[h3_id]
            parent = str(h3.cell_to_parent(h3_id, cell_resolution - 1)) if cell_resolution > 0 else None
            children = tuple(str(child) for child in h3.cell_to_children(h3_id, cell_resolution + 1)) if cell_resolution < 15 else ()
            self._parent_map[h3_id] = parent
            self._children_map[h3_id] = children

            ancestors: Dict[int, str] = {}
            for resolution in self.ancestor_resolutions:
                if resolution <= cell_resolution:
                    ancestors[resolution] = h3_id if resolution == cell_resolution else str(h3.cell_to_parent(h3_id, resolution))
            self._ancestor_map[h3_id] = ancestors

        self._exact_presence_map: Dict[str, Dict[str, bool]] = {
            "text": {h3_id: h3_id in self._modality_cells["text"] for h3_id in self.h3_ids},
            "image": {h3_id: h3_id in self._modality_cells["image"] for h3_id in self.h3_ids},
            "graph": {h3_id: h3_id in self._modality_cells["graph"] for h3_id in self.h3_ids},
        }
        self._hierarchy_presence_map: Dict[str, Dict[str, bool]] = {"text": {}, "graph": {}}
        for modality in ("text", "graph"):
            available_cells = self._modality_cells[modality]
            for h3_id in self.h3_ids:
                parent = self._parent_map[h3_id]
                has_center = h3_id in available_cells
                has_parent = parent is not None and parent in available_cells
                has_child = any(child in available_cells for child in self._children_map[h3_id])
                self._hierarchy_presence_map[modality][h3_id] = bool(has_center or has_parent or has_child)

        if self.require_modalities:
            kept: list[str] = []
            for h3_id in self.h3_ids:
                if all(self._is_present_for_loss(modality, h3_id) for modality in self.require_modalities):
                    kept.append(h3_id)
            kept_set = set(kept)
            self.h3_ids = kept
            self._h3_resolution = {h: self._h3_resolution[h] for h in kept}
            self._parent_map = {h: self._parent_map[h] for h in kept}
            self._children_map = {h: self._children_map[h] for h in kept}
            self._ancestor_map = {h: self._ancestor_map[h] for h in kept}
            self._exact_presence_map = {
                modality: {h: presence[h] for h in kept_set}
                for modality, presence in self._exact_presence_map.items()
            }
            self._hierarchy_presence_map = {
                modality: {h: presence[h] for h in kept_set}
                for modality, presence in self._hierarchy_presence_map.items()
            }

        self._image_h3_ids = {h3_id for h3_id in self.h3_ids if self._exact_presence_map["image"][h3_id]}
        self.has_image_indices = [index for index, h3_id in enumerate(self.h3_ids) if h3_id in self._image_h3_ids]
        self.no_image_indices = [index for index, h3_id in enumerate(self.h3_ids) if h3_id not in self._image_h3_ids]

        self._zero_cache: Dict[tuple[str, int], np.ndarray] = {}
        self._embedding_cache: Dict[tuple[str, str], list[np.ndarray]] = {}
        self._store: EmbeddingStore | None = None

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_store"] = None
        return state

    def __len__(self) -> int:
        return len(self.h3_ids)

    def _get_store(self) -> EmbeddingStore:
        if self._store is None:
            self._store = EmbeddingStore(self._store_path, read_only=True, initialize_schema=False)
        return self._store

    def _zero(self, dim: int, key: str) -> np.ndarray:
        cache_key = (key, dim)
        if cache_key not in self._zero_cache:
            self._zero_cache[cache_key] = np.full((dim,), self.mask_value, dtype=np.float32)
        return self._zero_cache[cache_key]

    def _get_or_zero(self, modality: str, h3_id: str, dim: int) -> np.ndarray:
        vector = self._sample_embedding(modality, h3_id, dim)
        if vector is None:
            return self._zero(dim, f"{modality}_single")
        return vector

    def _get_embeddings_cached(self, modality: str, h3_id: str) -> list[np.ndarray]:
        cache_key = (modality, h3_id)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        vectors = self._get_store().get_embeddings(modality, h3_id)
        self._embedding_cache[cache_key] = vectors
        return vectors

    def _sample_embedding(self, modality: str, h3_id: Optional[str], dim: int) -> Optional[np.ndarray]:
        if h3_id is None:
            return None

        vectors = self._get_embeddings_cached(modality, h3_id)
        if not vectors:
            return None

        if self.embedding_sample_strategy == "first":
            selected = vectors[0]
        elif self.embedding_sample_strategy == "mean":
            selected = np.mean(vectors, axis=0, dtype=np.float32)
        elif self.embedding_sample_strategy == "random":
            selected = random.choice(vectors)
        else:
            raise ValueError(f"Unknown embedding selection strategy: {self.embedding_sample_strategy}")

        if selected.shape[0] != dim:
            raise ValueError(
                f"Stored {modality} embedding for {h3_id} has dimension {selected.shape[0]}, expected {dim}"
            )
        return selected

    def _hierarchical(self, modality: str, h3_id: str, dim: int) -> np.ndarray:
        parent = self._parent_map[h3_id]
        children = self._children_map[h3_id]

        parent_vec = self._get_or_zero(modality, parent, dim) if parent is not None else self._zero(dim, f"{modality}_parent")
        center_vec = self._get_or_zero(modality, h3_id, dim)

        child_vecs = [self._sample_embedding(modality, child, dim) for child in children]
        child_vecs = [vector for vector in child_vecs if vector is not None]

        if child_vecs:
            pooled_child = np.mean(child_vecs, axis=0, dtype=np.float32)
            if pooled_child.shape[0] != dim:
                raise ValueError(
                    f"Stored {modality} child embeddings for {h3_id} have inconsistent dimensions: {pooled_child.shape[0]} vs {dim}"
                )
        else:
            pooled_child = self._zero(dim, f"{modality}_child")

        return np.concatenate([parent_vec, center_vec, pooled_child], axis=0)

    def _hierarchy_present(self, modality: str, h3_id: str) -> bool:
        return self._hierarchy_presence_map[modality][h3_id]

    def _is_present_for_loss(self, modality: str, h3_id: str) -> bool:
        if modality in self.hierarchy_for:
            return self._hierarchy_presence_map[modality][h3_id]
        return self._exact_presence_map[modality][h3_id]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        h3_id = self.h3_ids[idx]
        h3_resolution = self._h3_resolution[h3_id]

        if "text" not in self.active_modalities:
            dim = self.text_dim * 3 if "text" in self.hierarchy_for else self.text_dim
            text_vec = self._zero(dim, "text_inactive")
            text_present = False
        elif "text" in self.hierarchy_for:
            text_vec = self._hierarchical("text", h3_id, self.text_dim)
            text_present = self._hierarchy_present("text", h3_id)
        else:
            text_vec = self._get_or_zero("text", h3_id, self.text_dim)
            text_present = self._exact_presence_map["text"][h3_id]

        if "graph" not in self.active_modalities:
            dim = self.graph_dim * 3 if "graph" in self.hierarchy_for else self.graph_dim
            graph_vec = self._zero(dim, "graph_inactive")
            graph_present = False
        elif "graph" in self.hierarchy_for:
            graph_vec = self._hierarchical("graph", h3_id, self.graph_dim)
            graph_present = self._hierarchy_present("graph", h3_id)
        else:
            graph_vec = self._get_or_zero("graph", h3_id, self.graph_dim)
            graph_present = self._exact_presence_map["graph"][h3_id]

        if "image" not in self.active_modalities:
            dim = self.image_dim * 3 if "image" in self.hierarchy_for else self.image_dim
            image_vec = self._zero(dim, "image_inactive")
            image_present = False
        else:
            image_vec = self._get_or_zero("image", h3_id, self.image_dim)
            image_present = self._exact_presence_map["image"][h3_id]

        item: Dict[str, Any] = {
            "h3": h3_id,
            "h3_resolution": torch.tensor(h3_resolution, dtype=torch.long),
            "text_embedding": torch.tensor(text_vec, dtype=self.dtype),
            "image_embedding": torch.tensor(image_vec, dtype=self.dtype),
            "graph_embedding": torch.tensor(graph_vec, dtype=self.dtype),
            "text_present": torch.tensor(text_present, dtype=torch.bool),
            "image_present": torch.tensor(image_present, dtype=torch.bool),
            "graph_present": torch.tensor(graph_present, dtype=torch.bool),
        }

        for resolution in self.ancestor_resolutions:
            ancestor_key = f"h3_parent_{resolution}"
            item[ancestor_key] = self._ancestor_map[h3_id].get(resolution, "")

        return item


class ImageStratifiedBatchSampler(BatchSampler):
    """Yield each item once per epoch while best-effort balancing image coverage.

    The sampler consumes each index at most once per epoch. It shuffles the image
    and non-image pools independently, then allocates image examples across the
    epoch as evenly as possible so each batch reaches the requested image ratio
    when enough image examples exist.
    """

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
        if not 0.0 <= self.min_image_ratio <= 1.0:
            raise ValueError("min_image_ratio must be between 0 and 1")
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
        image_pool = deque(self.has_image_indices)
        non_image_pool = deque(self.no_image_indices)
        shuffled_images = list(image_pool)
        shuffled_non_images = list(non_image_pool)
        rng.shuffle(shuffled_images)
        rng.shuffle(shuffled_non_images)
        image_pool = deque(shuffled_images)
        non_image_pool = deque(shuffled_non_images)

        batch_sizes = [self.batch_size for _ in range(len(self))]
        if batch_sizes and not self.drop_last:
            batch_sizes[-1] = len(self.all_indices) - self.batch_size * (len(batch_sizes) - 1)

        for batch_index, target_size in enumerate(batch_sizes):
            remaining_batches = len(batch_sizes) - batch_index
            required_images = min(target_size, int(math.ceil(target_size * self.min_image_ratio)))
            fair_share = int(math.ceil(len(image_pool) / remaining_batches)) if remaining_batches else 0
            image_take = min(required_images, fair_share, len(image_pool))

            batch: list[int] = [image_pool.popleft() for _ in range(image_take)]
            remaining_slots = target_size - len(batch)

            non_image_take = min(remaining_slots, len(non_image_pool))
            batch.extend(non_image_pool.popleft() for _ in range(non_image_take))
            remaining_slots -= non_image_take

            if remaining_slots > 0:
                extra_image_take = min(remaining_slots, len(image_pool))
                batch.extend(image_pool.popleft() for _ in range(extra_image_take))
                remaining_slots -= extra_image_take

            if remaining_slots > 0:
                batch.extend(non_image_pool.popleft() for _ in range(min(remaining_slots, len(non_image_pool))))

            rng.shuffle(batch)
            yield batch


def _seed_worker(worker_id: int, base_seed: int) -> None:
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloader(
    dataset: H3TripleDataset,
    batch_size: int,
    min_image_ratio: float = 0.25,
    num_workers: int = 0,
    drop_last: bool = False,
    seed: int = 42,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    sampler = ImageStratifiedBatchSampler(
        has_image_indices=dataset.has_image_indices,
        no_image_indices=dataset.no_image_indices,
        batch_size=batch_size,
        min_image_ratio=min_image_ratio,
        drop_last=drop_last,
        seed=seed,
    )

    worker_init_fn = partial(_seed_worker, base_seed=seed) if num_workers > 0 else None

    data_loader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": worker_init_fn,
    }

    if num_workers > 0:
        data_loader_kwargs["persistent_workers"] = bool(persistent_workers) if persistent_workers is not None else True
        if prefetch_factor is not None:
            data_loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(**data_loader_kwargs)
