"""Run every downstream benchmark configuration in parallel subprocesses.

For each (task, feature_mode, modality_combo) tuple this script spawns an
independent invocation of `run_downstream_benchmarks.py`, captures its JSON
report, and writes a single combined report at the end.

Defaults cover the 6 tasks * (1 baseline + 4 pre-proj + 4 post-proj) = 54 runs.
Use `--jobs` to control the level of parallelism.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from triple_encoder.benchmarks.tasks import TASK_REGISTRY


DEFAULT_TASKS: tuple[str, ...] = tuple(sorted(TASK_REGISTRY.keys()))
DEFAULT_MODALITY_COMBOS: tuple[tuple[str, ...], ...] = (
    ("graph",),
    ("text",),
    ("image",),
    ("graph", "text", "image"),
)


@dataclass(frozen=True)
class RunSpec:
    task: str
    feature_mode: str
    modalities: tuple[str, ...]
    output_json: Path


def _config_id(task: str, feature_mode: str, modalities: Sequence[str]) -> str:
    mods = "all" if set(modalities) == {"graph", "text", "image"} else "+".join(modalities)
    return f"{task}__{feature_mode}__{mods}"


def _build_run_specs(
    tasks: Sequence[str],
    feature_modes: Sequence[str],
    modality_combos: Sequence[Sequence[str]],
    output_dir: Path,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for task in tasks:
        for mode in feature_modes:
            if mode == "no_embeddings":
                spec = RunSpec(
                    task=task,
                    feature_mode=mode,
                    modalities=("graph", "text", "image"),
                    output_json=output_dir / f"{_config_id(task, mode, ('graph','text','image'))}.json",
                )
                specs.append(spec)
                continue
            for modalities in modality_combos:
                modalities = tuple(modalities)
                spec = RunSpec(
                    task=task,
                    feature_mode=mode,
                    modalities=modalities,
                    output_json=output_dir / f"{_config_id(task, mode, modalities)}.json",
                )
                specs.append(spec)
    return specs


def _run_one(spec: RunSpec, common_args: list[str], log_dir: Path) -> dict:
    cmd = [
        sys.executable,
        "run_downstream_benchmarks.py",
        "--task",
        spec.task,
        "--feature-mode",
        spec.feature_mode,
        "--modalities",
        ",".join(spec.modalities),
        "--output-json",
        str(spec.output_json),
        *common_args,
    ]
    config_id = _config_id(spec.task, spec.feature_mode, spec.modalities)
    log_path = log_dir / f"{config_id}.log"
    spec.output_json.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    duration = time.time() - start
    return {
        "config_id": config_id,
        "cmd": cmd,
        "returncode": completed.returncode,
        "duration_sec": duration,
        "log_path": str(log_path),
        "output_json": str(spec.output_json),
    }


def _build_common_args(args: argparse.Namespace) -> list[str]:
    common: list[str] = [
        "--store-db",
        args.store_db,
        "--target-resolutions",
        args.target_resolutions,
        "--ancestor-resolutions",
        args.ancestor_resolutions,
        "--embedding-sample-strategy",
        args.embedding_sample_strategy,
        "--batch-size",
        str(args.batch_size),
        "--probe-model",
        args.probe_model,
        "--mlp-hidden-layers",
        args.mlp_hidden_layers,
        "--mlp-max-iter",
        str(args.mlp_max_iter),
    ]
    if args.checkpoint:
        common.extend(["--checkpoint", args.checkpoint])
    if args.use_ema:
        common.append("--use-ema")
    if args.include_baseline_features:
        common.append("--include-baseline-features")
    return common


def _aggregate_reports(specs: Sequence[RunSpec], summary_path: Path) -> dict:
    combined: list[dict] = []
    for spec in specs:
        if not spec.output_json.exists():
            continue
        try:
            with spec.output_json.open("r", encoding="utf-8") as handle:
                report = json.load(handle)
        except json.JSONDecodeError:
            continue
        for result in report.get("results", []):
            combined.append(
                {
                    "config_id": _config_id(spec.task, spec.feature_mode, spec.modalities),
                    "config": report.get("config", {}),
                    **result,
                }
            )
    summary = {
        "n_results": len(combined),
        "results": combined,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _parse_modality_combos(raw: str | None) -> tuple[tuple[str, ...], ...]:
    if not raw:
        return DEFAULT_MODALITY_COMBOS
    combos: list[tuple[str, ...]] = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        combos.append(tuple(token.strip() for token in chunk.split(",") if token.strip()))
    if not combos:
        return DEFAULT_MODALITY_COMBOS
    return tuple(combos)


def _parse_modes(raw: str) -> list[str]:
    valid = {"no_embeddings", "pre_projection", "post_projection"}
    if raw.strip().lower() == "all":
        return ["no_embeddings", "pre_projection", "post_projection"]
    modes = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    unknown = [mode for mode in modes if mode not in valid]
    if unknown:
        raise ValueError(f"Unknown feature modes: {unknown}. Valid: {sorted(valid)}")
    return modes


def _parse_tasks(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(DEFAULT_TASKS)
    tasks = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    unknown = [task for task in tasks if task not in TASK_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}. Valid: {sorted(TASK_REGISTRY.keys())}")
    return tasks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4, help="Number of parallel subprocesses")
    parser.add_argument("--tasks", type=str, default="all", help="Comma-separated task keys or 'all'")
    parser.add_argument(
        "--feature-modes",
        type=str,
        default="all",
        help="Comma-separated feature modes (no_embeddings,pre_projection,post_projection) or 'all'",
    )
    parser.add_argument(
        "--modality-combos",
        type=str,
        default="",
        help=(
            "Semicolon-separated modality sets, each comma-separated. "
            "Default: 'graph;text;image;graph,text,image'. Only applied to embedding modes."
        ),
    )
    parser.add_argument("--checkpoint", type=str, default="", help="Required for post_projection")
    parser.add_argument("--store-db", type=str, default="data/embeddings/embeddings.sqlite")
    parser.add_argument("--output-dir", type=str, default="outputs/downstream_benchmarks/all_runs")
    parser.add_argument("--summary-json", type=str, default="outputs/downstream_benchmarks/all_runs/summary.json")
    parser.add_argument("--target-resolutions", type=str, default="7,8,9")
    parser.add_argument("--ancestor-resolutions", type=str, default="7")
    parser.add_argument("--embedding-sample-strategy", type=str, default="mean", choices=["random", "first", "mean"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument(
        "--include-baseline-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Concatenate baseline task features with embedding features (default: on)",
    )
    parser.add_argument("--probe-model", type=str, default="ridge", choices=["ridge", "mlp"])
    parser.add_argument("--mlp-hidden-layers", type=str, default="512,256")
    parser.add_argument("--mlp-max-iter", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without launching them")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip configs whose output JSON already exists",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    tasks = _parse_tasks(args.tasks)
    feature_modes = _parse_modes(args.feature_modes)
    modality_combos = _parse_modality_combos(args.modality_combos)

    if "post_projection" in feature_modes and not args.checkpoint:
        parser.error("--checkpoint is required when feature-modes includes post_projection")

    output_dir = Path(args.output_dir)
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    specs = _build_run_specs(tasks, feature_modes, modality_combos, output_dir)
    if args.skip_existing:
        specs = [spec for spec in specs if not spec.output_json.exists()]

    common_args = _build_common_args(args)

    print(f"Total configurations: {len(specs)}")
    print(f"Parallel jobs: {args.jobs}")
    print(f"Output dir: {output_dir}")

    if args.dry_run:
        for spec in specs:
            cmd = [
                sys.executable,
                "run_downstream_benchmarks.py",
                "--task",
                spec.task,
                "--feature-mode",
                spec.feature_mode,
                "--modalities",
                ",".join(spec.modalities),
                "--output-json",
                str(spec.output_json),
                *common_args,
            ]
            print(" ".join(cmd))
        return 0

    if not specs:
        print("No work to do.")
        return 0

    failures: list[dict] = []
    completed = 0
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=max(1, int(args.jobs))) as pool:
        future_to_spec = {
            pool.submit(_run_one, spec, common_args, log_dir): spec for spec in specs
        }
        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]
            try:
                outcome = future.result()
            except Exception as exc:
                failures.append({"spec": vars(spec), "error": str(exc)})
                print(f"[ERROR] {_config_id(spec.task, spec.feature_mode, spec.modalities)}: {exc}", flush=True)
                continue
            completed += 1
            status = "ok" if outcome["returncode"] == 0 else "FAIL"
            print(
                f"[{completed}/{len(specs)}] {status} {outcome['config_id']} "
                f"({outcome['duration_sec']:.1f}s) -> {outcome['log_path']}",
                flush=True,
            )
            if outcome["returncode"] != 0:
                failures.append(outcome)

    elapsed = time.time() - start_time
    summary = _aggregate_reports(specs, Path(args.summary_json))
    print(f"\nElapsed: {elapsed:.1f}s. Combined report: {args.summary_json} ({summary['n_results']} results)")
    if failures:
        print(f"\nFailures: {len(failures)}")
        for failure in failures:
            print(f" - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
