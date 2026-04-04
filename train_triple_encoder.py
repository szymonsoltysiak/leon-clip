#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from triple_encoder.dataloader import H3TripleDataset, build_dataloader
from triple_encoder.losses import LossWeights, TriModalContrastiveLoss
from triple_encoder.model import TriModalCLIP
from triple_encoder.store import EmbeddingStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tri-modal CLIP model")
    parser.add_argument("--embeddings-root", type=str, default="data/embeddings")
    parser.add_argument("--store-db", type=str, default="data/embeddings/embeddings.sqlite")
    parser.add_argument("--rebuild-store", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-image-ratio", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lambda-gt", type=float, default=1.0)
    parser.add_argument("--lambda-gi", type=float, default=0.3)
    parser.add_argument("--lambda-ti", type=float, default=0.3)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default="geo-text-triple-encoder")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--target-resolutions",
        type=str,
        default="7,8,9",
        help="Comma-separated list of H3 resolutions to include in dataset (e.g. 7,8,9)",
    )
    return parser.parse_args()


def parse_target_resolutions(raw: str) -> tuple[int, ...]:
    items = [s.strip() for s in str(raw).split(",") if s.strip()]
    if not items:
        raise ValueError("--target-resolutions cannot be empty")
    out = tuple(sorted({int(s) for s in items}))
    for res in out:
        if res < 0 or res > 15:
            raise ValueError(f"Invalid H3 resolution in --target-resolutions: {res}")
    return out


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: TriModalCLIP, lr: float, weight_decay: float) -> AdamW:
    no_decay = [model.logit_scale]
    decay = [p for n, p in model.named_parameters() if n != "logit_scale"]
    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    target_resolutions = parse_target_resolutions(args.target_resolutions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    store = EmbeddingStore(args.store_db)
    try:
        dims = store.build_from_embeddings_root(args.embeddings_root, rebuild=args.rebuild_store)

        dataset = H3TripleDataset(store=store, target_resolutions=target_resolutions)
        dataloader = build_dataloader(
            dataset,
            batch_size=args.batch_size,
            min_image_ratio=args.min_image_ratio,
            num_workers=args.num_workers,
            seed=args.seed,
        )

        text_dim = int(dims.get("text", dataset.text_dim)) * 3
        graph_dim = int(dims.get("graph", dataset.graph_dim)) * 3
        image_dim = int(dims.get("image", dataset.image_dim))

        model = TriModalCLIP(
            text_in_dim=text_dim,
            image_in_dim=image_dim,
            graph_in_dim=graph_dim,
            proj_dim=args.proj_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)

        loss_fn = TriModalContrastiveLoss(
            weights=LossWeights(
                graph_text=args.lambda_gt,
                graph_image=args.lambda_gi,
                text_image=args.lambda_ti,
            )
        )

        optimizer = build_optimizer(model, args.lr, args.weight_decay)
        total_steps = max(1, args.epochs * len(dataloader))
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "min_image_ratio": args.min_image_ratio,
                "lambda_gt": args.lambda_gt,
                "lambda_gi": args.lambda_gi,
                "lambda_ti": args.lambda_ti,
                "text_base_dim": int(dims.get("text", dataset.text_dim)),
                "graph_base_dim": int(dims.get("graph", dataset.graph_dim)),
                "image_base_dim": int(dims.get("image", dataset.image_dim)),
                "text_in_dim": text_dim,
                "graph_in_dim": graph_dim,
                "image_in_dim": image_dim,
                "proj_dim": args.proj_dim,
                "hidden_dim": args.hidden_dim,
                "target_resolutions": list(target_resolutions),
            },
        )

        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        global_step = 0
        for epoch in range(args.epochs):
            model.train()
            if hasattr(dataloader.batch_sampler, "set_epoch"):
                dataloader.batch_sampler.set_epoch(epoch)

            epoch_bar = tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{args.epochs}",
                leave=True,
                dynamic_ncols=True,
            )

            for batch in epoch_bar:
                optimizer.zero_grad(set_to_none=True)

                graph_embedding = batch["graph_embedding"].to(device)
                text_embedding = batch["text_embedding"].to(device)
                image_embedding = batch["image_embedding"].to(device)

                model_out = model(
                    graph_embedding=graph_embedding,
                    text_embedding=text_embedding,
                    image_embedding=image_embedding,
                )

                loss, metrics = loss_fn(model_out, {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()})
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_bar.set_postfix(
                    loss=f"{loss.detach().item():.4f}",
                    gt=f"{metrics['loss_graph_text']:.4f}",
                    gi=f"{metrics['loss_graph_image']:.4f}",
                    ti=f"{metrics['loss_text_image']:.4f}",
                    temp=f"{model.logit_scale.detach().exp().item():.3f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

                if global_step % args.log_every == 0:
                    log_data: Dict[str, float] = {
                        "loss_total": float(loss.detach().item()),
                        "lr": float(scheduler.get_last_lr()[0]),
                        "logit_scale": float(model.logit_scale.detach().exp().item()),
                    }
                    log_data.update(metrics)
                    wandb.log(log_data, step=global_step)

                global_step += 1

            ckpt_path = save_dir / f"tri_clip_epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "args": vars(args),
                },
                ckpt_path,
            )

        run.finish()
    finally:
        store.close()


if __name__ == "__main__":
    main()
