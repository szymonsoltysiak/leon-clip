from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import torch

from triple_encoder.dataloader import H3TripleDataset, ImageStratifiedBatchSampler
from triple_encoder.eval import compute_retrieval_metrics
from triple_encoder.losses import (
    HardNegativeConfig,
    LossWeights,
    PositiveMaskConfig,
    PresenceNormalizationConfig,
    TriModalContrastiveLoss,
    build_hard_negative_mask,
    build_positive_mask,
)
from triple_encoder.model import HeadConfig, TriModalCLIP
from triple_encoder.store import EmbeddingStore
from train import (
    ExponentialMovingAverage,
    apply_image_embedding_noise,
    apply_modality_dropout,
    build_optimizer,
    build_warmup_cosine_scheduler,
)


def _make_embeddings_root(root: Path) -> tuple[str, str]:
    text_dir = root / "text_embeddings"
    image_dir = root / "image_embeddings"
    graph_dir = root / "graph_embeddings"
    text_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    center = h3.latlng_to_cell(37.7749, -122.4194, 8)
    sibling = h3.latlng_to_cell(37.8044, -122.2711, 8)

    text_frame = pd.DataFrame(
        {
            "h3": [center, sibling],
            "embedding": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        }
    )
    text_frame.to_parquet(text_dir / "text.parquet", index=False)

    image_payload = {center: np.array([1.0, 2.0, 3.0], dtype=np.float32)}
    graph_payload = {
        center: np.array([0.9, 0.8], dtype=np.float32),
        sibling: np.array([0.7, 0.6], dtype=np.float32),
    }
    with (image_dir / "image.pkl").open("wb") as handle:
        pickle.dump(image_payload, handle)
    with (graph_dir / "graph.pkl").open("wb") as handle:
        pickle.dump(graph_payload, handle)

    return center, sibling


def test_store_ingestion_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "embeddings"
    _make_embeddings_root(root)

    store = EmbeddingStore(tmp_path / "embeddings.sqlite")
    dims_first = store.build_from_embeddings_root(root, rebuild=False)
    count_after_first = store.count_embeddings()
    dims_second = store.build_from_embeddings_root(root, rebuild=False)
    count_after_second = store.count_embeddings()

    assert dims_first == {"text": 4, "image": 3, "graph": 2}
    assert dims_second == dims_first
    assert count_after_first == 5
    assert count_after_second == count_after_first


def test_store_can_be_reopened_read_only(tmp_path: Path) -> None:
    root = tmp_path / "embeddings"
    center, _ = _make_embeddings_root(root)

    store = EmbeddingStore(tmp_path / "embeddings.sqlite")
    store.build_from_embeddings_root(root, rebuild=False)
    store.close()

    read_only_store = EmbeddingStore(tmp_path / "embeddings.sqlite", read_only=True, initialize_schema=False)
    assert read_only_store.count_embeddings() == 5
    assert len(read_only_store.get_embeddings("text", center)) == 1


def test_store_preserves_distinct_vectors_and_deduplicates_exact_duplicates(tmp_path: Path) -> None:
    h3_id = h3.latlng_to_cell(37.7749, -122.4194, 8)
    store = EmbeddingStore(tmp_path / "embeddings.sqlite")

    vec_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    vec_b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

    assert store.insert_embedding("text", h3_id, vec_a)
    assert not store.insert_embedding("text", h3_id, vec_a)
    assert store.insert_embedding("text", h3_id, vec_b)

    vectors = store.get_embeddings("text", h3_id)
    assert len(vectors) == 2
    assert np.allclose(vectors[0], vec_a)
    assert np.allclose(vectors[1], vec_b)

    assert np.allclose(store.get_embedding("text", h3_id, strategy="first"), vec_a)
    assert np.allclose(store.get_embedding("text", h3_id, strategy="mean"), np.array([2.5, 3.5, 4.5], dtype=np.float32))

    rng_samples = {tuple(store.get_embedding("text", h3_id, strategy="random").tolist()) for _ in range(20)}
    assert tuple(vec_a.tolist()) in rng_samples
    assert tuple(vec_b.tolist()) in rng_samples


def test_store_migrates_legacy_schema_to_vec_hash(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE embeddings (
            sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
            modality TEXT NOT NULL,
            h3 TEXT NOT NULL,
            dim INTEGER NOT NULL,
            vec BLOB NOT NULL,
            UNIQUE(modality, h3)
        );
        CREATE TABLE dims (
            modality TEXT PRIMARY KEY,
            dim INTEGER NOT NULL
        );
        """
    )
    h3_id = h3.latlng_to_cell(37.7749, -122.4194, 8)
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    conn.execute(
        "INSERT INTO embeddings(modality, h3, dim, vec) VALUES (?, ?, ?, ?)",
        ("text", h3_id, 3, sqlite3.Binary(vec.tobytes())),
    )
    conn.commit()
    conn.close()

    store = EmbeddingStore(db_path)
    columns = {row[1] for row in store.conn.execute("PRAGMA table_info(embeddings)").fetchall()}
    assert "vec_hash" in columns
    assert store.count_embeddings("text") == 1


def test_dataset_shapes_and_image_presence(tmp_path: Path) -> None:
    root = tmp_path / "embeddings"
    center, sibling = _make_embeddings_root(root)

    store = EmbeddingStore(tmp_path / "embeddings.sqlite")
    store.build_from_embeddings_root(root, rebuild=False)

    dataset = H3TripleDataset(store=store, target_resolutions=(8,))
    assert len(dataset) == 2

    items = {dataset[index]["h3"]: dataset[index] for index in range(len(dataset))}

    center_item = items[center]
    sibling_item = items[sibling]

    assert center_item["text_embedding"].shape == (12,)
    assert center_item["graph_embedding"].shape == (6,)
    assert center_item["image_embedding"].shape == (3,)
    assert torch.is_tensor(center_item["text_present"])
    assert center_item["image_present"].item() is True
    assert sibling_item["image_present"].item() is False
    assert torch.allclose(sibling_item["image_embedding"], torch.zeros_like(sibling_item["image_embedding"]))


def test_sampler_covers_epoch_once_and_keeps_image_ratio() -> None:
    sampler = ImageStratifiedBatchSampler(
        has_image_indices=[0, 1, 2],
        no_image_indices=[3, 4, 5, 6, 7],
        batch_size=4,
        min_image_ratio=0.25,
        seed=7,
    )

    batches = list(iter(sampler))
    flattened = [index for batch in batches for index in batch]

    assert len(batches) == len(sampler) == 2
    assert sorted(flattened) == list(range(8))
    assert len(flattened) == len(set(flattened))
    assert all(sum(index in {0, 1, 2} for index in batch) >= 1 for batch in batches)


def test_model_forward_and_loss_are_finite() -> None:
    model = TriModalCLIP(
        text_in_dim=12,
        image_in_dim=3,
        graph_in_dim=6,
        proj_dim=8,
        head_config=HeadConfig(head_type="residual_mlp", hidden_dim=16, dropout=0.0, residual_layers=1),
        use_learned_missing_tokens=True,
    )
    batch_size = 4
    batch = {
        "h3": [h3.latlng_to_cell(37.7 + i * 0.01, -122.4, 8) for i in range(batch_size)],
        "h3_parent_7": [h3.latlng_to_cell(37.7 + i * 0.01, -122.4, 7) for i in range(batch_size)],
        "text_embedding": torch.randn(batch_size, 12),
        "image_embedding": torch.randn(batch_size, 3),
        "graph_embedding": torch.randn(batch_size, 6),
        "text_present": torch.ones(batch_size, dtype=torch.bool),
        "image_present": torch.ones(batch_size, dtype=torch.bool),
        "graph_present": torch.ones(batch_size, dtype=torch.bool),
    }

    model_out = model(
        graph_embedding=batch["graph_embedding"],
        text_embedding=batch["text_embedding"],
        image_embedding=batch["image_embedding"],
        graph_present=batch["graph_present"],
        text_present=batch["text_present"],
        image_present=batch["image_present"],
        return_raw=True,
    )

    for key in ("graph", "text", "image"):
        norms = torch.linalg.norm(model_out[key], dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)

    loss_fn = TriModalContrastiveLoss(
        LossWeights(),
        positive_cfg=PositiveMaskConfig(loss_mode="multi_positive", allow_cross_resolution_positives=False),
        hard_negative_cfg=HardNegativeConfig(mode="none", enable=False),
        presence_cfg=PresenceNormalizationConfig(enabled=True, mode="mean_valid"),
    )
    loss, metrics = loss_fn(model_out, batch)

    assert torch.isfinite(loss)
    assert all(np.isfinite(value) for value in metrics.values())
    assert "text_raw" in model_out
    assert "graph_raw" in model_out
    assert "image_raw" in model_out


def test_positive_mask_exact_and_cross_resolution() -> None:
    parent = h3.latlng_to_cell(37.7749, -122.4194, 7)
    child_a = h3.cell_to_children(parent, 8)[0]
    child_b = h3.cell_to_children(parent, 8)[1]

    exact_only = build_positive_mask(
        [child_a, child_b],
        loss_mode="multi_positive",
        allow_cross_resolution_positives=False,
        cross_resolution_mode="shared_ancestor",
        cross_resolution_ancestor_res=7,
    )
    assert bool(exact_only[0, 1]) is False

    shared_ancestor = build_positive_mask(
        [child_a, child_b],
        loss_mode="multi_positive",
        allow_cross_resolution_positives=True,
        cross_resolution_mode="shared_ancestor",
        cross_resolution_ancestor_res=7,
    )
    assert bool(shared_ancestor[0, 1]) is True


def test_hard_negative_mask_marks_shared_parent() -> None:
    parent = h3.latlng_to_cell(37.7749, -122.4194, 7)
    child_a = h3.cell_to_children(parent, 8)[0]
    child_b = h3.cell_to_children(parent, 8)[1]
    positive = torch.eye(2, dtype=torch.bool)
    hard = build_hard_negative_mask(
        [child_a, child_b],
        mode="shared_parent",
        radius=1,
        parent_res_delta=1,
        positive_mask=positive,
    )
    assert bool(hard[0, 1]) is True


def test_shared_masks_reused_across_pairs_counts_match() -> None:
    parent = h3.latlng_to_cell(37.7749, -122.4194, 7)
    child_a = h3.cell_to_children(parent, 8)[0]
    child_b = h3.cell_to_children(parent, 8)[1]
    h3_ids = [child_a, child_b]
    batch_size = 2

    model = TriModalCLIP(
        text_in_dim=4,
        image_in_dim=4,
        graph_in_dim=4,
        proj_dim=4,
        head_config=HeadConfig(head_type="linear", hidden_dim=8, dropout=0.0, residual_layers=0),
    )

    batch = {
        "h3": h3_ids,
        "h3_parent_7": [parent, parent],
        "text_embedding": torch.randn(batch_size, 4),
        "image_embedding": torch.randn(batch_size, 4),
        "graph_embedding": torch.randn(batch_size, 4),
        "text_present": torch.ones(batch_size, dtype=torch.bool),
        "image_present": torch.ones(batch_size, dtype=torch.bool),
        "graph_present": torch.ones(batch_size, dtype=torch.bool),
    }
    out = model(
        graph_embedding=batch["graph_embedding"],
        text_embedding=batch["text_embedding"],
        image_embedding=batch["image_embedding"],
        graph_present=batch["graph_present"],
        text_present=batch["text_present"],
        image_present=batch["image_present"],
    )
    loss_fn = TriModalContrastiveLoss(
        weights=LossWeights(),
        positive_cfg=PositiveMaskConfig(
            loss_mode="multi_positive",
            allow_cross_resolution_positives=True,
            cross_resolution_mode="shared_ancestor",
            cross_resolution_ancestor_res=7,
        ),
        hard_negative_cfg=HardNegativeConfig(mode="shared_parent", enable=True),
    )
    _, metrics = loss_fn(out, batch)

    assert metrics["multi_positive_matches_gt"] == metrics["multi_positive_matches_gi"]
    assert metrics["multi_positive_matches_gi"] == metrics["multi_positive_matches_ti"]
    assert metrics["hard_negative_count_gt"] == metrics["hard_negative_count_gi"]
    assert metrics["hard_negative_count_gi"] == metrics["hard_negative_count_ti"]


def test_modality_dropout_updates_effective_presence() -> None:
    batch = {
        "text_present": torch.tensor([True, True, False]),
        "image_present": torch.tensor([True, False, True]),
        "graph_present": torch.tensor([True, True, True]),
    }
    g = torch.Generator(device="cpu")
    g.manual_seed(42)
    stats = apply_modality_dropout(
        batch,
        text_p=1.0,
        image_p=0.0,
        graph_p=0.0,
        training=True,
        generator=g,
    )
    assert stats["dropout/text_dropped"] == 2.0
    assert batch["text_present"].tolist() == [False, False, False]
    assert batch["text_present_raw"].tolist() == [True, True, False]


def test_image_noise_updates_only_present_embeddings() -> None:
    torch.manual_seed(123)
    image = torch.zeros(3, 4)
    batch = {
        "image_embedding": image.clone(),
        "image_present": torch.tensor([True, False, True]),
    }
    stats = apply_image_embedding_noise(
        batch,
        enabled=True,
        noise_std=0.05,
        training=True,
    )

    assert stats["noise/image_applied"] == 2.0
    assert stats["noise/image_std"] == 0.05
    assert torch.any(batch["image_embedding"][0] != 0)
    assert torch.all(batch["image_embedding"][1] == 0)
    assert torch.any(batch["image_embedding"][2] != 0)


def test_missing_tokens_are_used_for_absent_modalities() -> None:
    model = TriModalCLIP(
        text_in_dim=4,
        image_in_dim=3,
        graph_in_dim=5,
        proj_dim=6,
        head_config=HeadConfig(head_type="linear", hidden_dim=8, dropout=0.0, residual_layers=0),
        use_learned_missing_tokens=True,
    )
    with torch.no_grad():
        model.text_missing_token.fill_(2.0)

    batch_size = 2
    out = model(
        graph_embedding=torch.randn(batch_size, 5),
        text_embedding=torch.zeros(batch_size, 4),
        image_embedding=torch.randn(batch_size, 3),
        graph_present=torch.ones(batch_size, dtype=torch.bool),
        text_present=torch.tensor([False, False]),
        image_present=torch.ones(batch_size, dtype=torch.bool),
    )
    assert torch.isfinite(out["text"]).all()


def test_scheduler_warmup_then_cosine_and_logit_group_lr() -> None:
    model = TriModalCLIP(
        text_in_dim=12,
        image_in_dim=3,
        graph_in_dim=6,
        proj_dim=8,
        head_config=HeadConfig(head_type="mlp", hidden_dim=16, dropout=0.0, residual_layers=0),
    )
    optimizer = build_optimizer(model, lr=1e-3, logit_scale_lr=1e-4, weight_decay=1e-2)
    assert optimizer.param_groups[0]["lr"] == 1e-3
    assert optimizer.param_groups[1]["lr"] == 1e-4

    scheduler = build_warmup_cosine_scheduler(optimizer, total_steps=10, warmup_steps=2)
    lrs = []
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        lrs.append(float(optimizer.param_groups[0]["lr"]))

    assert lrs[1] >= lrs[0]
    assert lrs[-1] <= lrs[2]


def test_retrieval_metrics_toy_example() -> None:
    source = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    present = torch.tensor([True, True])
    h3_ids = [h3.latlng_to_cell(37.7, -122.4, 8), h3.latlng_to_cell(37.8, -122.5, 8)]

    metrics = compute_retrieval_metrics(
        source,
        target,
        present,
        present,
        h3_ids,
        loss_mode="single_positive",
        allow_cross_resolution_positives=False,
        cross_resolution_mode="exact",
        cross_resolution_ancestor_res=7,
    )
    assert metrics["recall@1"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["mge@1_km"] == 0.0
    assert metrics["mge@5_km"] == 0.0
    assert metrics["mge@10_km"] == 0.0


def test_ema_update_changes_shadow() -> None:
    model = TriModalCLIP(
        text_in_dim=4,
        image_in_dim=3,
        graph_in_dim=5,
        proj_dim=6,
        head_config=HeadConfig(head_type="linear", hidden_dim=8, dropout=0.0, residual_layers=0),
    )
    ema = ExponentialMovingAverage(model, decay=0.9)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.1)
    old_shadow = {k: v.clone() for k, v in ema.shadow.items()}
    ema.update(model)

    diffs = [(ema.shadow[k] - old_shadow[k]).abs().sum().item() for k in ema.shadow]
    assert any(diff > 0 for diff in diffs)