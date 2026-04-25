#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import math
import random
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

try:
    import wandb
except Exception:  # pragma: no cover - optional runtime dependency
    wandb = None

from triple_encoder.dataloader import H3TripleDataset, ImageStratifiedBatchSampler
from triple_encoder.eval import evaluate_all_pairs
from triple_encoder.losses import (
    HardNegativeConfig,
    LossWeights,
    PositiveMaskConfig,
    PresenceNormalizationConfig,
    TriModalContrastiveLoss,
)
from triple_encoder.model import HeadConfig, TriModalCLIP
from triple_encoder.store import EmbeddingStore


LOGGER = logging.getLogger(__name__)


def parse_target_resolutions(raw: str) -> tuple[int, ...]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("--target-resolutions cannot be empty")
    output = tuple(sorted({int(item) for item in values}))
    for resolution in output:
        if resolution < 0 or resolution > 15:
            raise ValueError(f"Invalid H3 resolution: {resolution}")
    return output


def parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected a comma-separated integer list")
    return tuple(sorted({int(item) for item in values}))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tri-modal CLIP model")
    parser.add_argument("--embeddings-root", type=str, default="data")
    parser.add_argument("--store-db", type=str, default="data/embeddings/embeddings.sqlite")
    parser.add_argument("--rebuild-store", action="store_true")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-image-ratio", type=float, default=0.25)
    parser.add_argument("--target-resolutions", type=str, default="7,8,9")

    parser.add_argument("--training-mode", type=str, default="improved", choices=["baseline", "improved"])

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--logit-scale-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--head-type", type=str, default="residual_mlp", choices=["linear", "mlp", "residual_mlp"])
    parser.add_argument("--head-hidden-dim", type=int, default=512)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--head-residual-layers", type=int, default=1)

    parser.add_argument("--use-learned-missing-tokens", action="store_true", default=True)
    parser.add_argument("--disable-learned-missing-tokens", action="store_true")

    parser.add_argument("--text-modality-dropout", type=float, default=0.1)
    parser.add_argument("--image-modality-dropout", type=float, default=0.05)
    parser.add_argument("--graph-modality-dropout", type=float, default=0.1)
    parser.add_argument("--use-image-noise", action="store_true")
    parser.add_argument("--image-noise-std", type=float, default=0.05)

    parser.add_argument("--loss-mode", type=str, default="multi_positive", choices=["single_positive", "multi_positive"])
    parser.add_argument("--allow-cross-resolution-positives", action="store_true", default=True)
    parser.add_argument("--disable-cross-resolution-positives", action="store_true")
    parser.add_argument("--cross-resolution-mode", type=str, default="shared_ancestor", choices=["exact", "parent_child", "shared_ancestor"])
    parser.add_argument("--cross-resolution-ancestor-res", type=int, default=7)

    parser.add_argument("--hard-negative-mode", type=str, default="shared_parent", choices=["none", "same_resolution", "shared_parent", "k_ring", "combined"])
    parser.add_argument("--hard-negative-weight", type=float, default=1.5)
    parser.add_argument("--hard-negative-radius", type=int, default=1)
    parser.add_argument("--hard-negative-parent-res-delta", type=int, default=1)

    parser.add_argument("--presence-normalization", type=str, default="mean_valid", choices=["off", "mean_valid", "proportional"])

    parser.add_argument("--lambda-gt", type=float, default=1.0)
    parser.add_argument("--lambda-gi", type=float, default=1.0)
    parser.add_argument("--lambda-ti", type=float, default=1.0)

    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-at-epoch-end", action="store_true")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--max-eval-samples", type=int, default=2048)

    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--disable-ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--eval-use-ema", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--log-branch-norms", action="store_true")
    parser.add_argument("--save-dir", type=str, default="checkpoints")

    parser.add_argument("--wandb-project", type=str, default="geo-text-triple-encoder")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def apply_training_mode_defaults(args: argparse.Namespace) -> None:
    if args.disable_learned_missing_tokens:
        args.use_learned_missing_tokens = False
    if args.disable_cross_resolution_positives:
        args.allow_cross_resolution_positives = False
    if args.disable_ema:
        args.use_ema = False

    if args.training_mode != "baseline":
        return

    args.loss_mode = "single_positive"
    args.allow_cross_resolution_positives = False
    args.hard_negative_mode = "none"
    args.hard_negative_weight = 1.0
    args.text_modality_dropout = 0.0
    args.image_modality_dropout = 0.0
    args.graph_modality_dropout = 0.0
    args.use_learned_missing_tokens = False
    args.head_type = "linear"
    args.head_dropout = 0.0
    args.head_residual_layers = 0
    args.presence_normalization = "off"
    args.use_ema = False
    args.eval_use_ema = False


def split_indices(length: int, val_split: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(length))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = int(round(length * val_split)) if val_split > 0 else 0
    val_count = min(max(0, val_count), max(0, length - 1))
    return indices[val_count:], indices[:val_count]


def _subset_image_indices(dataset: H3TripleDataset, indices: list[int]) -> tuple[list[int], list[int]]:
    global_with_image = set(dataset.has_image_indices)
    has_image_local: list[int] = []
    no_image_local: list[int] = []
    for local_idx, global_idx in enumerate(indices):
        if global_idx in global_with_image:
            has_image_local.append(local_idx)
        else:
            no_image_local.append(local_idx)
    return has_image_local, no_image_local


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {
            name: parameter.detach().clone()
            for name, parameter in model.state_dict().items()
            if torch.is_tensor(parameter) and parameter.dtype.is_floating_point
        }

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            state = model.state_dict()
            for name, parameter in state.items():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        backup: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            state = model.state_dict()
            for name, parameter in state.items():
                if name in self.shadow:
                    backup[name] = parameter.detach().clone()
                    parameter.copy_(self.shadow[name])
        return backup

    def restore(self, model: torch.nn.Module, backup: Mapping[str, torch.Tensor]) -> None:
        with torch.no_grad():
            state = model.state_dict()
            for name, tensor in backup.items():
                if name in state:
                    state[name].copy_(tensor)


def build_optimizer(model: TriModalCLIP, lr: float, logit_scale_lr: float, weight_decay: float) -> AdamW:
    main_params = [parameter for name, parameter in model.named_parameters() if name != "logit_scale"]
    logit_scale_params = [model.logit_scale]
    return AdamW(
        [
            {"params": main_params, "lr": lr, "weight_decay": weight_decay},
            {"params": logit_scale_params, "lr": logit_scale_lr, "weight_decay": 0.0},
        ]
    )


def build_warmup_cosine_scheduler(optimizer: AdamW, total_steps: int, warmup_steps: int) -> LambdaLR:
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def init_wandb(args: argparse.Namespace, config: Dict[str, Any]):
    if args.disable_wandb or wandb is None:
        return None
    try:
        return wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=config)
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        warnings.warn(f"WandB initialisation failed, continuing without logging: {exc}", stacklevel=2)
        return None


def finish_wandb(run: Any) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        LOGGER.exception("WandB cleanup failed")


def apply_modality_dropout(
    batch: Dict[str, torch.Tensor],
    *,
    text_p: float,
    image_p: float,
    graph_p: float,
    training: bool,
    generator: torch.Generator,
) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for modality, prob in (("text", text_p), ("image", image_p), ("graph", graph_p)):
        present_key = f"{modality}_present"
        raw_key = f"{modality}_present_raw"
        batch[raw_key] = batch[present_key].clone()
        if not training or prob <= 0.0:
            stats[f"dropout/{modality}_dropped"] = 0.0
            continue

        available = batch[raw_key].bool()
        sampled = torch.rand(available.shape[0], generator=generator, device="cpu").to(available.device)
        dropped = available & (sampled < float(prob))
        batch[present_key] = available & (~dropped)
        stats[f"dropout/{modality}_dropped"] = float(dropped.sum().item())
    return stats


def apply_image_embedding_noise(
    batch: Dict[str, torch.Tensor],
    *,
    enabled: bool,
    noise_std: float,
    training: bool,
) -> Dict[str, float]:
    if not training or not enabled or noise_std <= 0.0:
        return {"noise/image_applied": 0.0, "noise/image_std": float(noise_std)}

    image_embedding = batch["image_embedding"]
    image_present = batch["image_present"].unsqueeze(-1).to(dtype=image_embedding.dtype)
    noise = torch.randn_like(image_embedding) * float(noise_std)
    noise = noise * image_present
    batch["image_embedding"] = image_embedding + noise
    return {
        "noise/image_applied": float(image_present.sum().item()),
        "noise/image_std": float(noise_std),
    }


def collect_eval_embeddings(
    model: TriModalCLIP,
    dataloader: DataLoader,
    device: torch.device,
    max_eval_samples: int,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], list[str], Dict[int, list[str | None]]]:
    model.eval()
    embeddings: dict[str, list[torch.Tensor]] = {"graph": [], "text": [], "image": []}
    presence: dict[str, list[torch.Tensor]] = {"graph": [], "text": [], "image": []}
    h3_ids: list[str] = []
    ancestors: dict[int, list[str | None]] = {}

    with torch.no_grad():
        seen = 0
        for batch in dataloader:
            batch_on_device = {
                key: value.to(device, non_blocking=torch.cuda.is_available()) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            model_out = model(
                graph_embedding=batch_on_device["graph_embedding"],
                text_embedding=batch_on_device["text_embedding"],
                image_embedding=batch_on_device["image_embedding"],
                graph_present=batch_on_device["graph_present"],
                text_present=batch_on_device["text_present"],
                image_present=batch_on_device["image_present"],
                return_raw=False,
            )
            for modality in ("graph", "text", "image"):
                embeddings[modality].append(model_out[modality].detach().cpu())
                presence[modality].append(batch_on_device[f"{modality}_present"].detach().cpu())

            h3_ids.extend([str(value) for value in batch["h3"]])
            for key, values in batch.items():
                if key.startswith("h3_parent_"):
                    resolution = int(key.replace("h3_parent_", ""))
                    ancestors.setdefault(resolution, [])
                    ancestors[resolution].extend([str(value) if str(value) else None for value in values])

            seen += int(batch_on_device["graph_embedding"].shape[0])
            if seen >= max_eval_samples:
                break

    stacked_embeddings = {key: torch.cat(value, dim=0)[:max_eval_samples] for key, value in embeddings.items()}
    stacked_presence = {key: torch.cat(value, dim=0)[:max_eval_samples] for key, value in presence.items()}
    h3_ids = h3_ids[:max_eval_samples]
    for resolution in list(ancestors.keys()):
        ancestors[resolution] = ancestors[resolution][:max_eval_samples]
    return stacked_embeddings, stacked_presence, h3_ids, ancestors


def compute_branch_norm_metrics(
    batch_on_device: Dict[str, torch.Tensor],
    model_out: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    return {
        "text_input_norm": float(torch.linalg.norm(batch_on_device["text_embedding"], dim=-1).mean().detach().item()),
        "image_input_norm": float(torch.linalg.norm(batch_on_device["image_embedding"], dim=-1).mean().detach().item()),
        "graph_input_norm": float(torch.linalg.norm(batch_on_device["graph_embedding"], dim=-1).mean().detach().item()),
        "text_projected_norm": float(torch.linalg.norm(model_out["text_raw"], dim=-1).mean().detach().item()),
        "image_projected_norm": float(torch.linalg.norm(model_out["image_raw"], dim=-1).mean().detach().item()),
        "graph_projected_norm": float(torch.linalg.norm(model_out["graph_raw"], dim=-1).mean().detach().item()),
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    apply_training_mode_defaults(args)
    seed_everything(args.seed)

    target_resolutions = parse_target_resolutions(args.target_resolutions)
    ancestor_resolutions = parse_int_tuple(str(args.cross_resolution_ancestor_res))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dropout_generator = torch.Generator(device="cpu")
    dropout_generator.manual_seed(args.seed)

    with EmbeddingStore(args.store_db) as store:
        dims = store.build_from_embeddings_root(args.embeddings_root, rebuild=args.rebuild_store)
        dataset = H3TripleDataset(
            store=store,
            target_resolutions=target_resolutions,
            ancestor_resolutions=ancestor_resolutions,
        )
        if len(dataset) == 0:
            raise RuntimeError(f"No H3 samples found under {args.embeddings_root}")

        train_indices, val_indices = split_indices(len(dataset), args.val_split, args.seed)
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices) if val_indices else None

        train_has_image, train_no_image = _subset_image_indices(dataset, train_indices)
        train_sampler = ImageStratifiedBatchSampler(
            has_image_indices=train_has_image,
            no_image_indices=train_no_image,
            batch_size=args.batch_size,
            min_image_ratio=args.min_image_ratio,
            seed=args.seed,
        )
        train_loader_kwargs: Dict[str, Any] = {
            "dataset": train_dataset,
            "batch_sampler": train_sampler,
            "num_workers": args.num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": args.num_workers > 0,
        }
        if args.num_workers > 0:
            train_loader_kwargs["prefetch_factor"] = 2
        train_loader = DataLoader(**train_loader_kwargs)

        val_loader: DataLoader | None = None
        if val_dataset is not None and len(val_dataset) > 0:
            val_workers = max(0, args.num_workers // 2)
            val_loader_kwargs: Dict[str, Any] = {
                "dataset": val_dataset,
                "batch_size": args.batch_size,
                "shuffle": False,
                "num_workers": val_workers,
                "pin_memory": torch.cuda.is_available(),
                "persistent_workers": val_workers > 0,
            }
            if val_workers > 0:
                val_loader_kwargs["prefetch_factor"] = 2
            val_loader = DataLoader(**val_loader_kwargs)

        text_dim = int(dims.get("text", dataset.text_dim)) * 3
        graph_dim = int(dims.get("graph", dataset.graph_dim)) * 3
        image_dim = int(dims.get("image", dataset.image_dim))

        head_cfg = HeadConfig(
            head_type=args.head_type,
            hidden_dim=args.head_hidden_dim,
            dropout=args.head_dropout,
            residual_layers=args.head_residual_layers,
        )
        model = TriModalCLIP(
            text_in_dim=text_dim,
            image_in_dim=image_dim,
            graph_in_dim=graph_dim,
            proj_dim=args.proj_dim,
            head_config=head_cfg,
            use_learned_missing_tokens=args.use_learned_missing_tokens,
        ).to(device)

        positive_cfg = PositiveMaskConfig(
            loss_mode=args.loss_mode,
            allow_cross_resolution_positives=args.allow_cross_resolution_positives,
            cross_resolution_mode=args.cross_resolution_mode,
            cross_resolution_ancestor_res=int(args.cross_resolution_ancestor_res),
        )
        hard_cfg = HardNegativeConfig(
            mode=args.hard_negative_mode,
            weight=args.hard_negative_weight,
            radius=args.hard_negative_radius,
            parent_res_delta=args.hard_negative_parent_res_delta,
            enable=args.hard_negative_mode != "none",
        )
        presence_cfg = PresenceNormalizationConfig(
            enabled=args.presence_normalization != "off",
            mode=args.presence_normalization,
        )
        loss_fn = TriModalContrastiveLoss(
            weights=LossWeights(
                graph_text=args.lambda_gt,
                graph_image=args.lambda_gi,
                text_image=args.lambda_ti,
            ),
            positive_cfg=positive_cfg,
            hard_negative_cfg=hard_cfg,
            presence_cfg=presence_cfg,
        )

        optimizer = build_optimizer(model, args.lr, args.logit_scale_lr, args.weight_decay)
        total_steps = max(1, args.epochs * len(train_loader))
        warmup_steps = args.warmup_steps if args.warmup_steps > 0 else int(total_steps * args.warmup_ratio)
        scheduler = build_warmup_cosine_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

        ema: ExponentialMovingAverage | None = ExponentialMovingAverage(model, args.ema_decay) if args.use_ema else None

        wandb_run = init_wandb(
            args,
            {
                "training_mode": args.training_mode,
                "target_resolutions": list(target_resolutions),
                "head": asdict(head_cfg),
                "positive": asdict(positive_cfg),
                "hard_negative": asdict(hard_cfg),
                "presence": asdict(presence_cfg),
                "lr": args.lr,
                "logit_scale_lr": args.logit_scale_lr,
                "use_image_noise": args.use_image_noise,
                "image_noise_std": args.image_noise_std,
                "warmup_steps": warmup_steps,
                "batch_size": args.batch_size,
            },
        )

        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        def run_eval(step: int) -> dict[str, float]:
            if val_loader is None:
                return {}
            backup: dict[str, torch.Tensor] | None = None
            if args.eval_use_ema and ema is not None:
                backup = ema.copy_to(model)
            try:
                embeddings, presence, h3_ids, ancestors = collect_eval_embeddings(
                    model,
                    val_loader,
                    device=device,
                    max_eval_samples=max(1, args.max_eval_samples),
                )
                metrics = evaluate_all_pairs(
                    embeddings,
                    presence,
                    h3_ids,
                    positive_cfg={
                        "loss_mode": args.loss_mode,
                        "allow_cross_resolution_positives": args.allow_cross_resolution_positives,
                        "cross_resolution_mode": args.cross_resolution_mode,
                        "cross_resolution_ancestor_res": int(args.cross_resolution_ancestor_res),
                    },
                    precomputed_ancestors=ancestors,
                )
                metrics["eval/step"] = float(step)
                return metrics
            finally:
                if backup is not None and ema is not None:
                    ema.restore(model, backup)

        global_step = 0
        try:
            for epoch in range(args.epochs):
                model.train()
                if hasattr(train_loader.batch_sampler, "set_epoch"):
                    train_loader.batch_sampler.set_epoch(epoch)

                progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)
                for batch in progress:
                    optimizer.zero_grad(set_to_none=True)

                    should_log = wandb_run is not None and (global_step % args.log_every == 0)
                    need_raw = bool(args.log_branch_norms and should_log)

                    batch_on_device = {
                        key: value.to(device, non_blocking=torch.cuda.is_available()) if torch.is_tensor(value) else value
                        for key, value in batch.items()
                    }
                    dropout_stats = apply_modality_dropout(
                        batch_on_device,
                        text_p=args.text_modality_dropout,
                        image_p=args.image_modality_dropout,
                        graph_p=args.graph_modality_dropout,
                        training=True,
                        generator=dropout_generator,
                    )
                    image_noise_stats = apply_image_embedding_noise(
                        batch_on_device,
                        enabled=args.use_image_noise,
                        noise_std=args.image_noise_std,
                        training=model.training,
                    )

                    model_out = model(
                        graph_embedding=batch_on_device["graph_embedding"],
                        text_embedding=batch_on_device["text_embedding"],
                        image_embedding=batch_on_device["image_embedding"],
                        graph_present=batch_on_device["graph_present"],
                        text_present=batch_on_device["text_present"],
                        image_present=batch_on_device["image_present"],
                        return_raw=need_raw,
                    )
                    loss, loss_metrics = loss_fn(model_out, batch_on_device)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Encountered non-finite loss")

                    loss.backward()
                    grad_norm = 0.0
                    if args.grad_clip_norm > 0:
                        grad_norm = float(clip_grad_norm_(model.parameters(), args.grad_clip_norm).item())
                    optimizer.step()
                    scheduler.step()
                    if ema is not None:
                        ema.update(model)

                    main_lr = float(optimizer.param_groups[0]["lr"])
                    logit_scale_lr = float(optimizer.param_groups[1]["lr"])
                    temp = float(model_out["logit_scale"].detach().item())

                    progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{main_lr:.2e}", temp=f"{temp:.2f}")

                    train_metrics = {
                        "loss_total": float(loss.detach().item()),
                        "lr/main": main_lr,
                        "lr/logit_scale": logit_scale_lr,
                        "logit_scale": temp,
                        "grad_norm": grad_norm,
                        **dropout_stats,
                        **image_noise_stats,
                        **loss_metrics,
                    }
                    if need_raw:
                        train_metrics.update(compute_branch_norm_metrics(batch_on_device, model_out))

                    if should_log:
                        wandb_run.log(train_metrics, step=global_step)

                    if args.eval_every > 0 and val_loader is not None and global_step > 0 and global_step % args.eval_every == 0:
                        eval_metrics = run_eval(global_step)
                        LOGGER.info("Evaluation at step %d: %s", global_step, eval_metrics)
                        if wandb_run is not None and eval_metrics:
                            wandb_run.log(eval_metrics, step=global_step)

                    global_step += 1

                checkpoint_path = save_dir / f"tri_clip_epoch_{epoch + 1}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "ema": None if ema is None else ema.shadow,
                        "step": global_step,
                        "epoch": epoch,
                        "args": vars(args),
                    },
                    checkpoint_path,
                )

                if args.eval_at_epoch_end and val_loader is not None:
                    epoch_eval = run_eval(global_step)
                    LOGGER.info("Epoch %d eval: %s", epoch + 1, epoch_eval)
                    if wandb_run is not None and epoch_eval:
                        wandb_run.log(epoch_eval, step=global_step)
        finally:
            finish_wandb(wandb_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
