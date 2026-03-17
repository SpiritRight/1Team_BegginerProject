#!/usr/bin/env python3
"""Argument parser for train_resnet.py."""

from __future__ import annotations

import argparse
from pathlib import Path


def resolve_path(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else (root / value)


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Faster R-CNN (ResNet-50-FPN) baseline using merged annotations."
    )
    parser.add_argument("--root", type=Path, default=default_root, help="Project root.")
    parser.add_argument(
        "--ann-dir",
        type=Path,
        default=Path("data/new_merged_annonation/new_merged_annonation"),
        help="Merged annotation directory.",
    )
    parser.add_argument(
        "--train-img-dir",
        type=Path,
        default=Path("data/train_images"),
        help="Training image directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/baseline_resnet"),
        help="Output directory for checkpoints and logs.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adamw"],
        help="Optimizer type.",
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD.")
    parser.add_argument("--step-size", type=int, default=8, help="StepLR step_size.")
    parser.add_argument("--gamma", type=float, default=0.1, help="StepLR gamma.")
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help='Training device, e.g. "cuda", "cuda:0", "cpu". Auto-detect if empty.',
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use torchvision pretrained Faster R-CNN backbone/head init.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_false",
        dest="pretrained",
        help="Disable pretrained weights.",
    )
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--log-interval", type=int, default=20, help="Step log interval.")
    parser.add_argument(
        "--eval-score-thr",
        type=float,
        default=0.001,
        help="Score threshold for validation mAP computation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build dataset/model and exit before training.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint path to resume from.",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="codeit_1team",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="",
        help="W&B entity/team name. Leave empty to use default.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default="baseline_v1",
        help="W&B run name. Leave empty for auto-generated name.",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default="",
        help="W&B group name.",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default="baseline, resnet50",
        help="Comma-separated W&B tags (e.g. baseline,resnet50).",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode.",
    )
    args = parser.parse_args()

    args.root = args.root.resolve()
    args.ann_dir = resolve_path(args.root, args.ann_dir).resolve()
    args.train_img_dir = resolve_path(args.root, args.train_img_dir).resolve()
    args.output_dir = resolve_path(args.root, args.output_dir).resolve()
    if args.resume is not None:
        args.resume = resolve_path(args.root, args.resume).resolve()

    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be between 0 and 1.")
    return args
