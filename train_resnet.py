#!/usr/bin/env python3
"""Train a Faster R-CNN (ResNet-50-FPN) baseline with merged annotations."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from train_resnet_args import parse_args

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor



try:
    import wandb

    _WANDB_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency
    wandb = None
    _WANDB_IMPORT_ERROR = exc


@dataclass
class ObjectAnnotation:
    category_id: int
    bbox_xywh: Tuple[float, float, float, float]


@dataclass
class ImageAnnotation:
    file_name: str
    width: int
    height: int
    objects: List[ObjectAnnotation]


def load_merged_annotations(
    ann_dir: Path,
) -> Tuple[List[ImageAnnotation], Dict[int, str]]:
    json_files = sorted(ann_dir.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {ann_dir}")

    image_map: Dict[str, ImageAnnotation] = {}
    class_id_to_name: Dict[int, str] = {}

    for json_path in json_files:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])
        if not images:
            continue

        image_info = images[0]
        file_name = str(image_info["file_name"])
        width = int(image_info["width"])
        height = int(image_info["height"])

        # In these merged files, annotation category_id is remapped (0..55),
        # and categories list is aligned to annotations order.
        local_id_to_name: Dict[int, str] = {}
        if len(annotations) == len(categories):
            for ann, cat in zip(annotations, categories):
                mapped_id = int(ann["category_id"])
                mapped_name = str(cat["name"]).strip()
                local_id_to_name[mapped_id] = mapped_name
                class_id_to_name.setdefault(mapped_id, mapped_name)
        else:
            by_cat_id = {int(cat["id"]): str(cat["name"]).strip() for cat in categories}
            for ann in annotations:
                mapped_id = int(ann["category_id"])
                if mapped_id in by_cat_id:
                    class_id_to_name.setdefault(mapped_id, by_cat_id[mapped_id])
                    local_id_to_name[mapped_id] = by_cat_id[mapped_id]

        objects: List[ObjectAnnotation] = []
        for ann in annotations:
            cat_id = int(ann["category_id"])
            bbox = ann.get("bbox", [0, 0, 0, 0])
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue
            objects.append(
                ObjectAnnotation(
                    category_id=cat_id,
                    bbox_xywh=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                )
            )
            if cat_id not in class_id_to_name:
                class_id_to_name[cat_id] = local_id_to_name.get(cat_id, f"class_{cat_id}")

        if file_name in image_map:
            image_map[file_name].objects.extend(objects)
        else:
            image_map[file_name] = ImageAnnotation(
                file_name=file_name,
                width=width,
                height=height,
                objects=objects,
            )

    entries = sorted(image_map.values(), key=lambda x: x.file_name)
    return entries, class_id_to_name


def split_entries(
    entries: Iterable[ImageAnnotation], val_ratio: float, seed: int
) -> Tuple[List[ImageAnnotation], List[ImageAnnotation]]:
    entries = list(entries)
    if len(entries) < 2:
        raise ValueError("Need at least two images for train/val split.")

    rnd = random.Random(seed)
    rnd.shuffle(entries)

    val_size = max(1, int(round(len(entries) * val_ratio)))
    val_size = min(val_size, len(entries) - 1)
    val_entries = sorted(entries[:val_size], key=lambda x: x.file_name)
    train_entries = sorted(entries[val_size:], key=lambda x: x.file_name)
    return train_entries, val_entries


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    if np is not None:
        np.random.seed(worker_seed)


def clip_xywh_to_xyxy(
    bbox_xywh: Tuple[float, float, float, float], width: int, height: int
) -> Tuple[float, float, float, float] | None:
    x, y, w, h = bbox_xywh
    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + w))
    y2 = max(0.0, min(float(height), y + h))
    if (x2 - x1) <= 1.0 or (y2 - y1) <= 1.0:
        return None
    return x1, y1, x2, y2


class PillDetectionDataset(Dataset):
    def __init__(self, entries: List[ImageAnnotation], image_dir: Path):
        self.entries = entries
        self.image_dir = image_dir
        self.to_tensor = ToTensor()

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        image_path = self.image_dir / entry.file_name
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.to_tensor(image)

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for obj in entry.objects:
            box = clip_xywh_to_xyxy(obj.bbox_xywh, entry.width, entry.height)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            boxes.append([x1, y1, x2, y2])
            # Faster R-CNN uses 0 as background, so shift by +1.
            labels.append(int(obj.category_id) + 1)
            areas.append((x2 - x1) * (y2 - y1))
            iscrowd.append(0)

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            areas_t = torch.tensor(areas, dtype=torch.float32)
            iscrowd_t = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            areas_t = torch.zeros((0,), dtype=torch.float32)
            iscrowd_t = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": areas_t,
            "iscrowd": iscrowd_t,
        }
        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes: int, pretrained: bool):
    used_pretrained = pretrained
    try:
        try:
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
            # Ensure offline-safe behavior when pretrained=False.
            model = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=None)
        except TypeError:
            # Backward compatibility for older torchvision versions.
            model = fasterrcnn_resnet50_fpn(
                pretrained=pretrained,
                pretrained_backbone=pretrained,
            )
    except Exception as exc:
        if not pretrained:
            raise
        print(
            f"Warning: failed to load pretrained weights ({exc}). "
            "Falling back to randomly initialized model."
        )
        used_pretrained = False
        try:
            model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        except TypeError:
            model = fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model, used_pretrained


def to_device(batch_images, batch_targets, device):
    images = [img.to(device) for img in batch_images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in batch_targets]
    return images, targets


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    epoch: int,
    log_interval: int,
    scaler,
) -> float:
    model.train()
    running_loss = 0.0
    steps = 0

    for step, (images, targets) in enumerate(data_loader, start=1):
        images, targets = to_device(images, targets, device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

        running_loss += float(losses.item())
        steps += 1

        if step % log_interval == 0:
            print(
                f"[Epoch {epoch:03d}] step {step:04d}/{len(data_loader):04d} "
                f"loss={losses.item():.4f}"
            )

    return running_loss / max(steps, 1)


@torch.no_grad()
def validate_one_epoch(model, data_loader, device) -> float:
    # Detection models return losses only in train mode.
    model.train()
    running_loss = 0.0
    steps = 0

    for images, targets in data_loader:
        images, targets = to_device(images, targets, device)
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        running_loss += float(losses.item())
        steps += 1

    return running_loss / max(steps, 1)


def box_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    x11, y11, x12, y12 = boxes1[:, 0:1], boxes1[:, 1:2], boxes1[:, 2:3], boxes1[:, 3:4]
    x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    inter_x1 = torch.maximum(x11, x21)
    inter_y1 = torch.maximum(y11, y21)
    inter_x2 = torch.minimum(x12, x22)
    inter_y2 = torch.minimum(y12, y22)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (x12 - x11).clamp(min=0) * (y12 - y11).clamp(min=0)
    area2 = (x22 - x21).clamp(min=0) * (y22 - y21).clamp(min=0)
    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-12)


def compute_ap_from_tp_fp(tp: torch.Tensor, fp: torch.Tensor, num_gt: int) -> float:
    if num_gt <= 0:
        return float("nan")
    if tp.numel() == 0:
        return 0.0

    tp_cum = torch.cumsum(tp, dim=0)
    fp_cum = torch.cumsum(fp, dim=0)
    recalls = tp_cum / float(num_gt)
    precisions = tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-12)

    mrec = torch.cat(
        [torch.tensor([0.0]), recalls, torch.tensor([1.0])], dim=0
    )
    mpre = torch.cat(
        [torch.tensor([0.0]), precisions, torch.tensor([0.0])], dim=0
    )

    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

    idx = torch.where(mrec[1:] != mrec[:-1])[0]
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).item()
    return float(ap)


def compute_ap_for_class_iou(
    predictions: List[dict],
    targets: List[dict],
    class_label: int,
    iou_thr: float,
    score_thr: float,
) -> float:
    gt_by_img: Dict[int, torch.Tensor] = {}
    n_gt = 0
    for img_idx, target in enumerate(targets):
        mask = target["labels"] == class_label
        gt_boxes = target["boxes"][mask]
        gt_by_img[img_idx] = gt_boxes
        n_gt += int(gt_boxes.shape[0])

    if n_gt == 0:
        return float("nan")

    pred_records = []
    for img_idx, pred in enumerate(predictions):
        mask = pred["labels"] == class_label
        boxes = pred["boxes"][mask]
        scores = pred["scores"][mask]
        if score_thr > 0:
            keep = scores >= score_thr
            boxes = boxes[keep]
            scores = scores[keep]
        for b, s in zip(boxes, scores):
            pred_records.append((img_idx, float(s.item()), b))

    if not pred_records:
        return 0.0

    pred_records.sort(key=lambda x: x[1], reverse=True)
    matched = {
        img_idx: torch.zeros((gt_boxes.shape[0],), dtype=torch.bool)
        for img_idx, gt_boxes in gt_by_img.items()
    }

    tp = torch.zeros((len(pred_records),), dtype=torch.float32)
    fp = torch.zeros((len(pred_records),), dtype=torch.float32)

    for i, (img_idx, _, pred_box) in enumerate(pred_records):
        gt_boxes = gt_by_img[img_idx]
        if gt_boxes.numel() == 0:
            fp[i] = 1.0
            continue

        ious = box_iou_matrix(pred_box.unsqueeze(0), gt_boxes).squeeze(0)
        best_iou, best_gt_idx = torch.max(ious, dim=0)
        if float(best_iou.item()) >= iou_thr and not matched[img_idx][best_gt_idx]:
            tp[i] = 1.0
            matched[img_idx][best_gt_idx] = True
        else:
            fp[i] = 1.0

    return compute_ap_from_tp_fp(tp=tp, fp=fp, num_gt=n_gt)


@torch.no_grad()
def evaluate_map75_95(
    model,
    data_loader,
    device,
    num_classes_without_bg: int,
    score_thr: float = 0.001,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    all_predictions: List[dict] = []
    all_targets: List[dict] = []

    for images, targets in data_loader:
        images_dev = [img.to(device) for img in images]
        outputs = model(images_dev)

        for pred, target in zip(outputs, targets):
            all_predictions.append(
                {
                    "boxes": pred["boxes"].detach().cpu(),
                    "labels": pred["labels"].detach().cpu(),
                    "scores": pred["scores"].detach().cpu(),
                }
            )
            all_targets.append(
                {
                    "boxes": target["boxes"].detach().cpu(),
                    "labels": target["labels"].detach().cpu(),
                }
            )

    iou_thresholds = [0.75, 0.80, 0.85, 0.90, 0.95]
    iou_to_map: Dict[str, float] = {}
    maps = []

    for iou_thr in iou_thresholds:
        ap_values = []
        for class_label in range(1, num_classes_without_bg + 1):
            ap = compute_ap_for_class_iou(
                predictions=all_predictions,
                targets=all_targets,
                class_label=class_label,
                iou_thr=iou_thr,
                score_thr=score_thr,
            )
            if ap == ap:  # not nan
                ap_values.append(ap)

        map_iou = float(sum(ap_values) / len(ap_values)) if ap_values else 0.0
        iou_key = f"{iou_thr:.2f}"
        iou_to_map[iou_key] = map_iou
        maps.append(map_iou)

    map75_95 = float(sum(maps) / len(maps)) if maps else 0.0
    return map75_95, iou_to_map


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_wandb_config(args: argparse.Namespace) -> dict:
    payload = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def main() -> None:
    args = parse_args()

    if not args.ann_dir.exists():
        raise FileNotFoundError(f"Annotation dir not found: {args.ann_dir}")
    if not args.train_img_dir.exists():
        raise FileNotFoundError(f"Train image dir not found: {args.train_img_dir}")

    set_global_seed(args.seed)

    entries, class_id_to_name = load_merged_annotations(args.ann_dir)
    train_entries, val_entries = split_entries(entries, args.val_ratio, args.seed)
    num_classes = len(class_id_to_name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    split_payload = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "train_files": [x.file_name for x in train_entries],
        "val_files": [x.file_name for x in val_entries],
    }
    save_json(args.output_dir / "split.json", split_payload)
    save_json(
        args.output_dir / "class_id_to_name.json",
        {str(k): v for k, v in sorted(class_id_to_name.items())},
    )

    train_dataset = PillDetectionDataset(train_entries, args.train_img_dir)
    val_dataset = PillDetectionDataset(val_entries, args.train_img_dir)
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(args.seed + 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=train_generator,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=val_generator,
        collate_fn=collate_fn,
    )

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, used_pretrained = build_model(num_classes=num_classes, pretrained=args.pretrained)
    model.to(device)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    start_epoch = 1
    best_val_loss = float("inf")
    best_map75_95 = -1.0
    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        best_map75_95 = float(checkpoint.get("best_map75_95", -1.0))
        print(f"Resumed from: {args.resume} (start_epoch={start_epoch})")

    print("=== Training Configuration ===")
    print(f"ann_dir       : {args.ann_dir}")
    print(f"train_img_dir : {args.train_img_dir}")
    print(f"output_dir    : {args.output_dir}")
    print(f"num_classes   : {num_classes} (+1 background)")
    print(f"train_images  : {len(train_entries)}")
    print(f"val_images    : {len(val_entries)}")
    print(f"seed          : {args.seed}")
    print(f"device        : {device}")
    print(f"optimizer     : {args.optimizer}")
    print(f"epochs        : {args.epochs}")
    print(f"pretrained    : {used_pretrained}")

    if args.dry_run:
        print("Dry run complete. Exiting before training.")
        return

    wandb_run = None
    if args.use_wandb:
        if _WANDB_IMPORT_ERROR is not None:
            raise RuntimeError(
                "wandb is not installed. Install first:\n"
                "  pip install wandb"
            ) from _WANDB_IMPORT_ERROR

        tags = [x.strip() for x in args.wandb_tags.split(",") if x.strip()]
        wandb_kwargs = {
            "project": args.wandb_project,
            "config": build_wandb_config(args),
            "mode": args.wandb_mode,
        }
        if args.wandb_entity:
            wandb_kwargs["entity"] = args.wandb_entity
        if args.wandb_run_name:
            wandb_kwargs["name"] = args.wandb_run_name
        if args.wandb_group:
            wandb_kwargs["group"] = args.wandb_group
        if tags:
            wandb_kwargs["tags"] = tags

        wandb_run = wandb.init(**wandb_kwargs)
        wandb.config.update(
            {
                "num_classes_without_bg": num_classes,
                "train_images": len(train_entries),
                "val_images": len(val_entries),
                "used_pretrained": used_pretrained,
            },
            allow_val_change=True,
        )
        print(f"W&B enabled   : run={wandb_run.name} ({wandb_run.id})")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.time()
            train_loss = train_one_epoch(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                log_interval=args.log_interval,
                scaler=scaler,
            )
            val_loss = validate_one_epoch(model=model, data_loader=val_loader, device=device)
            val_map75_95, val_map_by_iou = evaluate_map75_95(
                model=model,
                data_loader=val_loader,
                device=device,
                num_classes_without_bg=num_classes,
                score_thr=args.eval_score_thr,
            )
            scheduler.step()
            elapsed = time.time() - epoch_start

            print(
                f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_mAP75-95={val_map75_95:.4f} "
                f"time={elapsed:.1f}s"
            )
            print(f"  val mAP by IoU: {val_map_by_iou}")

            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": min(best_val_loss, val_loss),
                "val_map75_95": val_map75_95,
                "val_map_by_iou": val_map_by_iou,
                "best_map75_95": max(best_map75_95, val_map75_95),
                "num_classes_without_bg": num_classes,
                "class_id_to_name": class_id_to_name,
                "label_offset": 1,
                "args": vars(args),
            }
            if wandb_run is not None:
                state["wandb_run_id"] = wandb_run.id
            torch.save(state, ckpt_dir / "last.pt")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(state, ckpt_dir / "best.pt")
                print(f"  -> best checkpoint updated: {ckpt_dir / 'best.pt'}")

            if val_map75_95 > best_map75_95:
                best_map75_95 = val_map75_95
                torch.save(state, ckpt_dir / "best_map.pt")
                print(f"  -> best mAP checkpoint updated: {ckpt_dir / 'best_map.pt'}")

            if wandb_run is not None:
                log_payload = {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/mAP75-95": val_map75_95,
                    "best/val_loss": best_val_loss,
                    "best/mAP75-95": best_map75_95,
                    "time/epoch_sec": elapsed,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
                for iou_key, iou_map in val_map_by_iou.items():
                    log_payload[f"val/mAP@{iou_key}"] = iou_map
                wandb.log(log_payload, step=epoch)
    finally:
        if wandb_run is not None:
            wandb_run.summary["best_val_loss"] = best_val_loss
            wandb_run.summary["best_val_mAP75-95"] = best_map75_95
            wandb.finish()

    print("Training finished.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val mAP75-95: {best_map75_95:.4f}")
    print(f"Last checkpoint: {ckpt_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
