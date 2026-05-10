from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_log_error,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from triple_encoder import EmbeddingStore, H3TripleDataset
from triple_encoder.model import HeadConfig, TriModalCLIP

from .tasks import TASK_REGISTRY, TaskSpec


ALL_MODALITIES = ("graph", "text", "image")


@dataclass
class RunResult:
    task: str
    task_name: str
    task_type: str
    feature_mode: str
    modalities: list[str]
    n_train: int
    n_test: int
    h3_train_total: int
    h3_train_covered: int
    h3_train_missing: int
    h3_test_total: int
    h3_test_covered: int
    h3_test_missing: int
    modality_coverage_train: dict[str, dict[str, int]]
    modality_coverage_test: dict[str, dict[str, int]]
    metrics: dict[str, float | None]


def _parse_csv_arg(value: str) -> list[str]:
    tokens = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    if not tokens:
        raise ValueError("Expected at least one comma-separated value")
    return tokens


def parse_tasks(task_arg: str) -> list[str]:
    lowered = task_arg.strip().lower()
    if lowered == "all":
        return sorted(TASK_REGISTRY.keys())

    tasks = _parse_csv_arg(task_arg)
    unknown = [task for task in tasks if task not in TASK_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown task(s): {unknown}. Available: {sorted(TASK_REGISTRY.keys())}")
    return tasks


def parse_feature_modes(mode_arg: str) -> list[str]:
    lowered = mode_arg.strip().lower()
    valid = {"no_embeddings", "pre_projection", "post_projection"}
    if lowered == "all":
        return ["no_embeddings", "pre_projection", "post_projection"]

    modes = _parse_csv_arg(lowered)
    unknown = [mode for mode in modes if mode not in valid]
    if unknown:
        raise ValueError(f"Unknown feature mode(s): {unknown}. Available: {sorted(valid)}")
    return modes


def parse_modalities(modalities_arg: str) -> list[str]:
    lowered = modalities_arg.strip().lower()
    if lowered == "all":
        return list(ALL_MODALITIES)

    modalities = _parse_csv_arg(lowered)
    unknown = [modality for modality in modalities if modality not in ALL_MODALITIES]
    if unknown:
        raise ValueError(f"Unknown modalities: {unknown}. Available: {list(ALL_MODALITIES)}")

    seen: set[str] = set()
    deduped: list[str] = []
    for modality in modalities:
        if modality not in seen:
            deduped.append(modality)
            seen.add(modality)
    return deduped


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    values = sorted({int(token.strip()) for token in value.split(",") if token.strip()})
    if not values:
        raise ValueError("Resolution list cannot be empty")
    return tuple(values)


def _build_model_from_checkpoint(
    checkpoint_path: Path,
    dataset: H3TripleDataset,
    device: torch.device,
    use_ema: bool,
) -> TriModalCLIP:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})

    text_dim = int(checkpoint_args.get("text_in_dim", dataset.text_dim)) * 3
    graph_dim = int(checkpoint_args.get("graph_in_dim", dataset.graph_dim)) * 3
    image_dim = int(checkpoint_args.get("image_in_dim", dataset.image_dim))

    head_cfg = HeadConfig(
        head_type=str(checkpoint_args.get("head_type", "residual_mlp")),
        hidden_dim=int(checkpoint_args.get("head_hidden_dim", 512)),
        dropout=float(checkpoint_args.get("head_dropout", 0.1)),
        residual_layers=int(checkpoint_args.get("head_residual_layers", 1)),
    )

    model = TriModalCLIP(
        text_in_dim=text_dim,
        image_in_dim=image_dim,
        graph_in_dim=graph_dim,
        proj_dim=int(checkpoint_args.get("proj_dim", 512)),
        head_config=head_cfg,
        use_learned_missing_tokens=bool(checkpoint_args.get("use_learned_missing_tokens", True)),
    ).to(device)

    if use_ema and checkpoint.get("ema") is not None:
        model.load_state_dict(checkpoint["ema"], strict=True)
    else:
        model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def _prepare_baseline_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    spec: TaskSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    for frame in (train_df, test_df):
        frame["h3_index"] = frame["h3_index"].astype(str)

    all_data = pd.concat([train_df, test_df], keys=["train", "test"])
    categorical_cols = [column for column in spec.categorical_cols if column in all_data.columns]
    all_encoded = pd.get_dummies(all_data, columns=categorical_cols, drop_first=True)

    train_encoded = all_encoded.loc["train"].copy()
    test_encoded = all_encoded.loc["test"].copy()

    h3_train = train_encoded["h3_index"].astype(str)
    h3_test = test_encoded["h3_index"].astype(str)
    y_train = train_df[spec.target].copy()
    y_test = test_df[spec.target].copy()

    drop_set = set(spec.drop_cols)
    drop_set.add(spec.target)
    if "geometry" in train_encoded.columns:
        drop_set.add("geometry")
    if "h3_index" in train_encoded.columns:
        drop_set.add("h3_index")

    feature_cols = [column for column in train_encoded.columns if column not in drop_set]
    x_train = train_encoded[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x_test = test_encoded[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return x_train, x_test, y_train, y_test, h3_train, h3_test


def _select_h3_rows(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    h3_train: pd.Series,
    h3_test: pd.Series,
    allowed_h3: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    train_mask = h3_train.isin(allowed_h3)
    test_mask = h3_test.isin(allowed_h3)

    x_train_sel = x_train.loc[train_mask].to_numpy(dtype=np.float32)
    x_test_sel = x_test.loc[test_mask].to_numpy(dtype=np.float32)
    y_train_sel = y_train.loc[train_mask].to_numpy()
    y_test_sel = y_test.loc[test_mask].to_numpy()
    h3_train_sel = h3_train.loc[train_mask].tolist()
    h3_test_sel = h3_test.loc[test_mask].tolist()
    return x_train_sel, x_test_sel, y_train_sel, y_test_sel, h3_train_sel, h3_test_sel


def _collect_pre_projection_features(
    dataset: H3TripleDataset,
    h3_values: Sequence[str],
    modalities: Sequence[str],
    progress_desc: str,
) -> tuple[np.ndarray, list[int]]:
    h3_to_index = {h3_id: index for index, h3_id in enumerate(dataset.h3_ids)}
    selected_rows: list[np.ndarray] = []
    keep_indices: list[int] = []

    for row_index, h3_id in enumerate(tqdm(h3_values, desc=progress_desc, unit="h3", leave=False)):
        dataset_index = h3_to_index.get(str(h3_id))
        if dataset_index is None:
            continue
        item = dataset[dataset_index]
        vectors = [item[f"{modality}_embedding"].detach().cpu().numpy().reshape(-1) for modality in modalities]
        selected_rows.append(np.concatenate(vectors, axis=0))
        keep_indices.append(row_index)

    if not selected_rows:
        return np.zeros((0, 0), dtype=np.float32), keep_indices
    return np.vstack(selected_rows).astype(np.float32), keep_indices


def _collect_post_projection_features(
    dataset: H3TripleDataset,
    h3_values: Sequence[str],
    modalities: Sequence[str],
    model: TriModalCLIP,
    device: torch.device,
    batch_size: int,
    progress_desc: str,
) -> tuple[np.ndarray, list[int]]:
    h3_to_index = {h3_id: index for index, h3_id in enumerate(dataset.h3_ids)}
    rows: list[dict[str, torch.Tensor]] = []
    keep_indices: list[int] = []
    modality_mask = {modality: torch.tensor(modality in modalities, dtype=torch.bool) for modality in ALL_MODALITIES}

    for row_index, h3_id in enumerate(tqdm(h3_values, desc=progress_desc, unit="h3", leave=False)):
        dataset_index = h3_to_index.get(str(h3_id))
        if dataset_index is None:
            continue
        item = dataset[dataset_index]
        rows.append(item)
        keep_indices.append(row_index)

    if not rows:
        return np.zeros((0, 0), dtype=np.float32), keep_indices

    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for start in tqdm(
            range(0, len(rows), batch_size),
            desc=f"{progress_desc} batches",
            unit="batch",
            leave=False,
        ):
            chunk = rows[start : start + batch_size]
            batch = {
                "graph_embedding": torch.stack([row["graph_embedding"] for row in chunk], dim=0).to(device),
                "text_embedding": torch.stack([row["text_embedding"] for row in chunk], dim=0).to(device),
                "image_embedding": torch.stack([row["image_embedding"] for row in chunk], dim=0).to(device),
                "graph_present": torch.stack([row["graph_present"] for row in chunk], dim=0).to(device),
                "text_present": torch.stack([row["text_present"] for row in chunk], dim=0).to(device),
                "image_present": torch.stack([row["image_present"] for row in chunk], dim=0).to(device),
            }

            out = model(
                graph_embedding=batch["graph_embedding"],
                text_embedding=batch["text_embedding"],
                image_embedding=batch["image_embedding"],
                graph_present=batch["graph_present"] & modality_mask["graph"].to(device),
                text_present=batch["text_present"] & modality_mask["text"].to(device),
                image_present=batch["image_present"] & modality_mask["image"].to(device),
                return_raw=False,
            )
            chunk_features = torch.cat([out[modality] for modality in modalities], dim=-1)
            vectors.append(chunk_features.detach().cpu().numpy())

    return np.vstack(vectors).astype(np.float32), keep_indices


def _subset_by_indices(values: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    if values.size == 0:
        return values
    index_array = np.asarray(indices, dtype=np.int64)
    return values[index_array]


def _parse_hidden_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(token.strip()) for token in value.split(",") if token.strip())
    if not layers:
        raise ValueError("hidden layer specification cannot be empty")
    return layers


def _build_model(task_type: str, probe_model: str, mlp_hidden_layers: tuple[int, ...], mlp_max_iter: int):
    if task_type == "classification" and probe_model == "mlp":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=mlp_hidden_layers,
                        max_iter=int(mlp_max_iter),
                        random_state=42,
                    ),
                ),
            ]
        )

    if task_type == "classification":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000)),
            ]
        )

    if probe_model == "mlp":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "reg",
                    MLPRegressor(
                        hidden_layer_sizes=mlp_hidden_layers,
                        max_iter=int(mlp_max_iter),
                        random_state=42,
                    ),
                ),
            ]
        )

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ]
    )


def _fit_probe_with_progress(
    probe: Pipeline,
    x_train: np.ndarray,
    y_train: np.ndarray,
    probe_model: str,
    mlp_max_iter: int,
    run_label: str,
) -> None:
    if probe_model != "mlp":
        probe.fit(x_train, y_train)
        return

    scaler: StandardScaler = probe.named_steps["scaler"]
    model = probe.named_steps.get("clf", probe.named_steps.get("reg"))
    if model is None:
        raise ValueError("Expected 'clf' or 'reg' step in MLP probe pipeline")

    x_scaled = scaler.fit_transform(x_train)
    model.warm_start = True
    model.max_iter = 1
    model.tol = 0.0

    for _ in tqdm(range(int(mlp_max_iter)), desc=f"Training MLP ({run_label})", unit="epoch", leave=False):
        model.fit(x_scaled, y_train)


def _format_h3_coverage(covered: int, total: int, missing: int) -> str:
    return f"{covered}/{total} (miss {missing})"


def _modality_coverage_for_h3s(
    dataset: H3TripleDataset,
    h3_values: Sequence[str],
    modalities: Sequence[str],
    feature_mode: str,
) -> dict[str, dict[str, int]]:
    """Per-modality coverage among the supplied H3 cells.

    Reports how many of the H3 cells passed in have a *real* embedding for each
    requested modality (anything not present is filled with zeros at probe time).
    Coverage uses hierarchical presence for text/graph in post_projection mode
    (matching the dataset config) and exact presence everywhere else.
    """

    coverage: dict[str, dict[str, int]] = {}
    total = len(h3_values)
    if total == 0:
        for modality in modalities:
            coverage[modality] = {"covered": 0, "missing": 0, "total": 0}
        return coverage

    use_hierarchy = feature_mode == "post_projection"
    for modality in modalities:
        if modality == "image" or not use_hierarchy:
            presence = dataset._exact_presence_map[modality]
        else:
            presence = dataset._hierarchy_presence_map[modality]
        covered = sum(1 for h3_id in h3_values if presence.get(str(h3_id), False))
        coverage[modality] = {
            "covered": int(covered),
            "missing": int(total - covered),
            "total": int(total),
        }
    return coverage


def _format_modality_coverage(coverage: dict[str, dict[str, int]]) -> str:
    if not coverage:
        return "n/a"
    parts = []
    for modality, stats in coverage.items():
        parts.append(f"{modality}={stats['covered']}/{stats['total']}")
    return ", ".join(parts)


def _evaluate(task_type: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    if task_type == "classification":
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    y_true_safe = np.maximum(y_true, 0)
    y_pred_safe = np.maximum(y_pred, 0)
    rmsle: float | None
    try:
        rmsle = float(root_mean_squared_log_error(y_true_safe, y_pred_safe))
    except ValueError:
        rmsle = None

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "rmsle": rmsle,
    }


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def print_results_table(results: Sequence[RunResult]) -> None:
    if not results:
        print("No benchmark results were produced.")
        return

    metric_names = sorted({name for row in results for name in row.metrics.keys()})
    headers = [
        "task",
        "mode",
        "modalities",
        "n_train",
        "n_test",
        "h3_train",
        "h3_test",
        "cov_train",
        "cov_test",
    ] + metric_names
    rows: list[list[str]] = []
    for row in results:
        line = [
            row.task,
            row.feature_mode,
            "+".join(row.modalities),
            str(row.n_train),
            str(row.n_test),
            _format_h3_coverage(row.h3_train_covered, row.h3_train_total, row.h3_train_missing),
            _format_h3_coverage(row.h3_test_covered, row.h3_test_total, row.h3_test_missing),
            _format_modality_coverage(row.modality_coverage_train),
            _format_modality_coverage(row.modality_coverage_test),
        ]
        line.extend(_format_metric(row.metrics.get(metric)) for metric in metric_names)
        rows.append(line)

    widths = [max(len(headers[col]), max(len(r[col]) for r in rows)) for col in range(len(headers))]

    def fmt(line: Sequence[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(line))

    print(fmt(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(fmt(row))


def run_single(
    spec: TaskSpec,
    feature_mode: str,
    modalities: Sequence[str],
    dataset: H3TripleDataset,
    store_h3_set: set[str],
    checkpoint: Path | None,
    device: torch.device,
    batch_size: int,
    use_ema: bool,
    include_baseline_features: bool,
    probe_model: str,
    mlp_hidden_layers: tuple[int, ...],
    mlp_max_iter: int,
) -> RunResult:
    train_df, test_df = spec.load_splits()
    x_train_base, x_test_base, y_train, y_test, h3_train, h3_test = _prepare_baseline_split(train_df, test_df, spec)

    h3_train_total = int(len(h3_train))
    h3_test_total = int(len(h3_test))

    x_train, x_test, y_train_arr, y_test_arr, h3_train_sel, h3_test_sel = _select_h3_rows(
        x_train_base,
        x_test_base,
        y_train,
        y_test,
        h3_train,
        h3_test,
        allowed_h3=store_h3_set,
    )

    if feature_mode == "no_embeddings":
        x_train_mode = x_train
        x_test_mode = x_test
    elif feature_mode == "pre_projection":
        x_train_mode, train_keep = _collect_pre_projection_features(
            dataset,
            h3_train_sel,
            modalities,
            progress_desc=f"Loading pre-proj train ({spec.key})",
        )
        x_test_mode, test_keep = _collect_pre_projection_features(
            dataset,
            h3_test_sel,
            modalities,
            progress_desc=f"Loading pre-proj test ({spec.key})",
        )
        baseline_train = _subset_by_indices(x_train, train_keep)
        baseline_test = _subset_by_indices(x_test, test_keep)
        y_train_arr = _subset_by_indices(y_train_arr, train_keep)
        y_test_arr = _subset_by_indices(y_test_arr, test_keep)
        if include_baseline_features:
            x_train_mode = np.concatenate([baseline_train.astype(np.float32), x_train_mode], axis=1)
            x_test_mode = np.concatenate([baseline_test.astype(np.float32), x_test_mode], axis=1)
    elif feature_mode == "post_projection":
        if checkpoint is None:
            raise ValueError("post_projection mode requires --checkpoint")
        model = _build_model_from_checkpoint(checkpoint, dataset, device=device, use_ema=use_ema)
        x_train_mode, train_keep = _collect_post_projection_features(
            dataset,
            h3_train_sel,
            modalities,
            model=model,
            device=device,
            batch_size=batch_size,
            progress_desc=f"Loading post-proj train ({spec.key})",
        )
        x_test_mode, test_keep = _collect_post_projection_features(
            dataset,
            h3_test_sel,
            modalities,
            model=model,
            device=device,
            batch_size=batch_size,
            progress_desc=f"Loading post-proj test ({spec.key})",
        )
        baseline_train = _subset_by_indices(x_train, train_keep)
        baseline_test = _subset_by_indices(x_test, test_keep)
        y_train_arr = _subset_by_indices(y_train_arr, train_keep)
        y_test_arr = _subset_by_indices(y_test_arr, test_keep)
        if include_baseline_features:
            x_train_mode = np.concatenate([baseline_train.astype(np.float32), x_train_mode], axis=1)
            x_test_mode = np.concatenate([baseline_test.astype(np.float32), x_test_mode], axis=1)
    else:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")

    if x_train_mode.size == 0 or x_test_mode.size == 0:
        raise RuntimeError(
            f"Task '{spec.key}' mode '{feature_mode}' has no usable samples after H3 alignment with embedding store"
        )

    probe = _build_model(
        spec.task_type,
        probe_model=probe_model,
        mlp_hidden_layers=mlp_hidden_layers,
        mlp_max_iter=mlp_max_iter,
    )
    _fit_probe_with_progress(
        probe,
        x_train_mode,
        y_train_arr,
        probe_model=probe_model,
        mlp_max_iter=mlp_max_iter,
        run_label=f"{spec.key}/{feature_mode}",
    )
    y_pred = probe.predict(x_test_mode)
    metrics = _evaluate(spec.task_type, y_test_arr, y_pred)

    h3_train_covered = int(len(y_train_arr))
    h3_test_covered = int(len(y_test_arr))
    h3_train_missing = max(0, h3_train_total - h3_train_covered)
    h3_test_missing = max(0, h3_test_total - h3_test_covered)

    if feature_mode == "no_embeddings":
        modality_coverage_train: dict[str, dict[str, int]] = {}
        modality_coverage_test: dict[str, dict[str, int]] = {}
    else:
        modality_coverage_train = _modality_coverage_for_h3s(
            dataset, h3_train_sel, modalities, feature_mode
        )
        modality_coverage_test = _modality_coverage_for_h3s(
            dataset, h3_test_sel, modalities, feature_mode
        )
        cov_train_str = _format_modality_coverage(modality_coverage_train)
        cov_test_str = _format_modality_coverage(modality_coverage_test)
        print(
            f"[coverage] task={spec.key} mode={feature_mode} "
            f"modalities={'+'.join(modalities)} | "
            f"train hexes with real embedding -> {cov_train_str} (of {len(h3_train_sel)}) | "
            f"test hexes with real embedding -> {cov_test_str} (of {len(h3_test_sel)})"
        )

    return RunResult(
        task=spec.key,
        task_name=spec.name,
        task_type=spec.task_type,
        feature_mode=feature_mode,
        modalities=list(modalities),
        n_train=int(len(y_train_arr)),
        n_test=int(len(y_test_arr)),
        h3_train_total=h3_train_total,
        h3_train_covered=h3_train_covered,
        h3_train_missing=h3_train_missing,
        h3_test_total=h3_test_total,
        h3_test_covered=h3_test_covered,
        h3_test_missing=h3_test_missing,
        modality_coverage_train=modality_coverage_train,
        modality_coverage_test=modality_coverage_test,
        metrics=metrics,
    )


def run_benchmarks(args: argparse.Namespace) -> dict[str, object]:
    tasks = parse_tasks(args.task)
    feature_modes = parse_feature_modes(args.feature_mode)
    modalities = parse_modalities(args.modalities)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_resolutions = _parse_int_tuple(args.target_resolutions)
    ancestor_resolutions = _parse_int_tuple(args.ancestor_resolutions)

    store = EmbeddingStore(args.store_db, read_only=True, initialize_schema=False)
    try:
        with tqdm(total=2, desc="Loading datasets", unit="dataset") as data_bar:
            raw_dataset = H3TripleDataset(
                store=store,
                target_resolutions=target_resolutions,
                ancestor_resolutions=ancestor_resolutions,
                hierarchy_for=(),
                embedding_sample_strategy=args.embedding_sample_strategy,
            )
            data_bar.update(1)
            projected_dataset = H3TripleDataset(
                store=store,
                target_resolutions=target_resolutions,
                ancestor_resolutions=ancestor_resolutions,
                hierarchy_for=("text", "graph"),
                embedding_sample_strategy=args.embedding_sample_strategy,
            )
            data_bar.update(1)
    finally:
        store.close()

    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    mlp_hidden_layers = _parse_hidden_layers(args.mlp_hidden_layers)

    runs = [(task_key, feature_mode) for task_key in tasks for feature_mode in feature_modes]
    results: list[RunResult] = []
    for task_key, feature_mode in tqdm(runs, desc="Running benchmark probes", unit="run"):
        spec = TASK_REGISTRY[task_key]
        if feature_mode == "pre_projection":
            dataset = raw_dataset
        elif feature_mode == "post_projection":
            dataset = projected_dataset
        else:
            dataset = raw_dataset

        store_h3_set = set(dataset.h3_ids)
        result = run_single(
            spec=spec,
            feature_mode=feature_mode,
            modalities=modalities,
            dataset=dataset,
            store_h3_set=store_h3_set,
            checkpoint=checkpoint,
            device=device,
            batch_size=args.batch_size,
            use_ema=args.use_ema,
            include_baseline_features=args.include_baseline_features,
            probe_model=args.probe_model,
            mlp_hidden_layers=mlp_hidden_layers,
            mlp_max_iter=args.mlp_max_iter,
        )
        results.append(result)

    print_results_table(results)

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": {
            "task": tasks,
            "feature_mode": feature_modes,
            "modalities": modalities,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "store_db": args.store_db,
            "target_resolutions": list(target_resolutions),
            "ancestor_resolutions": list(ancestor_resolutions),
            "embedding_sample_strategy": args.embedding_sample_strategy,
            "batch_size": args.batch_size,
            "use_ema": bool(args.use_ema),
            "include_baseline_features": bool(args.include_baseline_features),
            "probe_model": args.probe_model,
            "mlp_hidden_layers": list(mlp_hidden_layers),
            "mlp_max_iter": int(args.mlp_max_iter),
        },
        "results": [asdict(result) for result in results],
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report to {output_path}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run downstream tri-modal benchmark probes")
    parser.add_argument("--task", type=str, default="all", help="Task key or comma-separated task keys, or 'all'")
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="all",
        help="One or more of: no_embeddings,pre_projection,post_projection, or 'all'",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default="all",
        help="Comma-separated modalities from graph,text,image, or 'all'",
    )
    parser.add_argument("--checkpoint", type=str, default="", help="TriModalCLIP checkpoint path for post_projection mode")
    parser.add_argument("--store-db", type=str, default="data/embeddings/embeddings.sqlite")
    parser.add_argument("--output-json", type=str, default="outputs/downstream_benchmarks/report.json")
    parser.add_argument("--target-resolutions", type=str, default="7,8,9")
    parser.add_argument("--ancestor-resolutions", type=str, default="7")
    parser.add_argument("--embedding-sample-strategy", type=str, default="mean", choices=["random", "first", "mean"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--use-ema", action="store_true", help="Load EMA weights from checkpoint when available")
    parser.add_argument(
        "--include-baseline-features",
        action="store_true",
        help="Concatenate baseline task features with embedding features for pre/post projection modes",
    )
    parser.add_argument(
        "--probe-model",
        type=str,
        default="ridge",
        choices=["ridge", "mlp"],
        help="Probe model for downstream prediction",
    )
    parser.add_argument(
        "--mlp-hidden-layers",
        type=str,
        default="512,256",
        help="Comma-separated hidden layer sizes for MLP probe",
    )
    parser.add_argument("--mlp-max-iter", type=int, default=500)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_benchmarks(args)
    return 0
