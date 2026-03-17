#!/usr/bin/env python3
"""Run Faster R-CNN inference and export predictions as submission CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

try:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.transforms import ToTensor

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - import guard for argparse/help usage
    torch = None
    Image = None
    DataLoader = None
    Dataset = object
    fasterrcnn_resnet50_fpn = None
    FastRCNNPredictor = None
    ToTensor = None
    _IMPORT_ERROR = exc


@dataclass(frozen=True)
class TestImageInfo:
    path: Path
    image_id: int


def resolve_path(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else (root / value)


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Inference for Faster R-CNN checkpoint and CSV submission export."
    )
    parser.add_argument("--root", type=Path, default=default_root, help="Project root.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/resnet_baseline_new_merged/checkpoints/best.pt"),
        help="Trained checkpoint path (.pt).",
    )
    parser.add_argument(
        "--test-img-dir",
        type=Path,
        default=Path("data/test_images"),
        help="Directory containing test images.",
    )
    parser.add_argument(
        "--ann-dir",
        type=Path,
        default=Path("data/new_merged_annonation/new_merged_annonation"),
        help="Merged annotation directory used to recover category-id mapping.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("runs/resnet_baseline_new_merged/submission.csv"),
        help="Path to output CSV.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help='Inference device, e.g. "cuda", "cuda:0", "cpu". Auto-detect if empty.',
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.05,
        help="Minimum score threshold for keeping predictions.",
    )
    parser.add_argument(
        "--max-dets-per-img",
        type=int,
        default=4,
        help="Maximum number of detections to keep per image (<=0 means keep all).",
    )
    parser.add_argument(
        "--agnostic-nms-iou-thr",
        type=float,
        default=0.3,
        help=(
            "Class-agnostic NMS IoU threshold. "
            "Set to <1.0 to suppress overlapping boxes across different classes."
        ),
    )
    parser.add_argument(
        "--exts",
        type=str,
        default=".png,.jpg,.jpeg,.bmp",
        help="Comma-separated image extensions to read.",
    )
    parser.add_argument(
        "--category-id-format",
        type=str,
        default="dl_idx",
        choices=["dl_idx", "original", "train_json", "remapped", "one_based"],
        help=(
            'Output category_id format: "dl_idx"/"original" (e.g., 1900), '
            '"train_json"/"remapped" (0~N-1), "one_based" (1~N).'
        ),
    )
    args = parser.parse_args()

    args.root = args.root.resolve()
    args.checkpoint = resolve_path(args.root, args.checkpoint).resolve()
    args.test_img_dir = resolve_path(args.root, args.test_img_dir).resolve()
    args.ann_dir = resolve_path(args.root, args.ann_dir).resolve()
    args.output_csv = resolve_path(args.root, args.output_csv).resolve()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0.")
    if args.score_thr < 0:
        raise ValueError("--score-thr must be >= 0.")
    if not (0.0 <= args.agnostic_nms_iou_thr <= 1.0):
        raise ValueError("--agnostic-nms-iou-thr must be in [0, 1].")
    return args


def build_model(num_classes: int):
    try:
        # Ensure offline-safe behavior for inference.
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    except TypeError:
        model = fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


def parse_image_id_from_name(path: Path) -> int:
    # Submission rule: image_id must match numeric part of file name.
    match = re.search(r"\d+", path.stem)
    if match is None:
        raise ValueError(f"Cannot parse numeric image_id from file name: {path.name}")
    return int(match.group(0))


def load_test_images(test_img_dir: Path, exts: Sequence[str]) -> List[TestImageInfo]:
    ext_set = {x.lower().strip() for x in exts if x.strip()}
    if not ext_set:
        raise ValueError("No valid image extensions were provided.")

    files = [p for p in test_img_dir.iterdir() if p.is_file() and p.suffix.lower() in ext_set]
    if not files:
        raise FileNotFoundError(
            f"No test images found in {test_img_dir} with extensions: {sorted(ext_set)}"
        )

    infos = [TestImageInfo(path=f, image_id=parse_image_id_from_name(f)) for f in files]
    infos.sort(key=lambda x: (x.image_id, x.path.name))

    seen_ids = set()
    for info in infos:
        if info.image_id in seen_ids:
            raise ValueError(
                f"Duplicate image_id detected after parsing file names: {info.image_id}"
            )
        seen_ids.add(info.image_id)
    return infos


class InferenceDataset(Dataset):
    def __init__(self, image_infos: List[TestImageInfo]):
        self.image_infos = image_infos
        self.to_tensor = ToTensor()

    def __len__(self) -> int:
        return len(self.image_infos)

    def __getitem__(self, idx: int):
        info = self.image_infos[idx]
        image = Image.open(info.path).convert("RGB")
        width, height = image.size
        image_tensor = self.to_tensor(image)
        meta = {
            "image_id": info.image_id,
            "width": width,
            "height": height,
            "file_name": info.path.name,
        }
        return image_tensor, meta


def collate_fn(batch):
    return tuple(zip(*batch))


def clamp_box_xyxy(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> tuple[float, float, float, float]:
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    return x1, y1, x2, y2


def load_remapped_to_original_id_map(ann_dir: Path) -> dict[int, int]:
    json_files = sorted(ann_dir.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in annotation dir: {ann_dir}")

    mapping: dict[int, int] = {}
    for json_path in json_files:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        annotations = data.get("annotations", [])
        categories = data.get("categories", [])
        if not annotations or not categories:
            continue

        # merged annotation format: each annotation has aligned category entry
        if len(annotations) == len(categories):
            for ann, cat in zip(annotations, categories):
                remapped_id = int(ann["category_id"])
                original_id = int(cat["id"])
                prev = mapping.get(remapped_id)
                if prev is None:
                    mapping[remapped_id] = original_id
                elif prev != original_id:
                    raise ValueError(
                        f"Conflicting mapping for remapped category_id={remapped_id}: "
                        f"{prev} vs {original_id} (source: {json_path})"
                    )

    if not mapping:
        raise ValueError(
            "Failed to build remapped->original category-id mapping from annotation files."
        )
    return mapping


def convert_category_id(
    remapped_category_id: int,
    category_id_format: str,
    remapped_to_original: dict[int, int] | None,
) -> int:
    if category_id_format in {"train_json", "remapped"}:
        return remapped_category_id
    if category_id_format == "one_based":
        return remapped_category_id + 1
    if category_id_format in {"dl_idx", "original"}:
        if remapped_to_original is None:
            raise ValueError(
                "category-id-format requires remapped->original mapping, but it is missing."
            )
        if remapped_category_id not in remapped_to_original:
            raise ValueError(
                f"No original category_id found for remapped id: {remapped_category_id}"
            )
        return remapped_to_original[remapped_category_id]

    raise ValueError(f"Unsupported category-id format: {category_id_format}")


def class_agnostic_nms(
    boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float
) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = torch.argsort(scores, descending=True)

    keep: List[int] = []
    while order.numel() > 0:
        i = int(order[0].item())
        keep.append(i)
        if order.numel() == 1:
            break

        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])

        inter_w = (xx2 - xx1).clamp(min=0)
        inter_h = (yy2 - yy1).clamp(min=0)
        inter = inter_w * inter_h
        union = areas[i] + areas[rest] - inter
        iou = inter / union.clamp(min=1e-12)

        order = rest[iou <= iou_thr]

    return torch.tensor(keep, dtype=torch.int64)


@torch.no_grad()
def run_inference(
    model,
    data_loader,
    device,
    label_offset: int,
    category_id_format: str,
    remapped_to_original: dict[int, int] | None,
    score_thr: float,
    max_dets_per_img: int,
    agnostic_nms_iou_thr: float,
) -> List[dict]:
    rows: List[dict] = []
    model.eval()

    for images, metas in data_loader:
        images_dev = [img.to(device) for img in images]
        outputs = model(images_dev)

        for meta, pred in zip(metas, outputs):
            boxes = pred["boxes"].detach().cpu()
            labels = pred["labels"].detach().cpu()
            scores = pred["scores"].detach().cpu()

            if score_thr > 0:
                keep = scores >= score_thr
                boxes = boxes[keep]
                labels = labels[keep]
                scores = scores[keep]

            if agnostic_nms_iou_thr < 1.0 and boxes.shape[0] > 0:
                keep = class_agnostic_nms(
                    boxes=boxes,
                    scores=scores,
                    iou_thr=agnostic_nms_iou_thr,
                )
                boxes = boxes[keep]
                labels = labels[keep]
                scores = scores[keep]

            if max_dets_per_img > 0 and boxes.shape[0] > max_dets_per_img:
                boxes = boxes[:max_dets_per_img]
                labels = labels[:max_dets_per_img]
                scores = scores[:max_dets_per_img]

            width = int(meta["width"])
            height = int(meta["height"])
            image_id = int(meta["image_id"])

            for box_t, label_t, score_t in zip(boxes, labels, scores):
                x1, y1, x2, y2 = map(float, box_t.tolist())
                x1, y1, x2, y2 = clamp_box_xyxy(x1, y1, x2, y2, width=width, height=height)
                w = x2 - x1
                h = y2 - y1
                if w <= 0.0 or h <= 0.0:
                    continue

                category_id = int(label_t.item()) - label_offset
                if category_id < 0:
                    continue

                output_category_id = convert_category_id(
                    remapped_category_id=category_id,
                    category_id_format=category_id_format,
                    remapped_to_original=remapped_to_original,
                )

                rows.append(
                    {
                        "image_id": image_id,
                        "category_id": output_category_id,
                        "bbox_x": round(x1, 4),
                        "bbox_y": round(y1, 4),
                        "bbox_w": round(w, 4),
                        "bbox_h": round(h, 4),
                        "score": round(float(score_t.item()), 6),
                    }
                )
    return rows


def save_submission_csv(output_csv: Path, rows: List[dict]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "annotation_id",
        "image_id",
        "category_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "score",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for annotation_id, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "annotation_id": annotation_id,
                    "image_id": row["image_id"],
                    "category_id": row["category_id"],
                    "bbox_x": row["bbox_x"],
                    "bbox_y": row["bbox_y"],
                    "bbox_w": row["bbox_w"],
                    "bbox_h": row["bbox_h"],
                    "score": row["score"],
                }
            )


def main() -> None:
    args = parse_args()

    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "PyTorch/torchvision/Pillow is not installed. Install first:\n"
            "  pip install torch torchvision pillow"
        ) from _IMPORT_ERROR

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.test_img_dir.exists():
        raise FileNotFoundError(f"Test image dir not found: {args.test_img_dir}")
    if args.category_id_format in {"dl_idx", "original"} and not args.ann_dir.exists():
        raise FileNotFoundError(f"Annotation dir not found: {args.ann_dir}")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
    else:
        model_state_dict = checkpoint

    if "num_classes_without_bg" in checkpoint:
        num_classes = int(checkpoint["num_classes_without_bg"])
    elif "class_id_to_name" in checkpoint:
        num_classes = len(checkpoint["class_id_to_name"])
    else:
        raise KeyError(
            "Cannot infer `num_classes_without_bg` from checkpoint. "
            "Expected key: `num_classes_without_bg`."
        )

    label_offset = int(checkpoint.get("label_offset", 1))
    remapped_to_original = None
    if args.category_id_format in {"dl_idx", "original"}:
        remapped_to_original = load_remapped_to_original_id_map(args.ann_dir)

    model = build_model(num_classes=num_classes)
    model.load_state_dict(model_state_dict)
    model.to(device)

    exts = [x.strip() for x in args.exts.split(",")]
    image_infos = load_test_images(args.test_img_dir, exts)
    dataset = InferenceDataset(image_infos)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    rows = run_inference(
        model=model,
        data_loader=data_loader,
        device=device,
        label_offset=label_offset,
        category_id_format=args.category_id_format,
        remapped_to_original=remapped_to_original,
        score_thr=args.score_thr,
        max_dets_per_img=args.max_dets_per_img,
        agnostic_nms_iou_thr=args.agnostic_nms_iou_thr,
    )
    save_submission_csv(args.output_csv, rows)

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Test images : {len(image_infos)}")
    print(f"Pred rows   : {len(rows)}")
    print(f"Category ID : {args.category_id_format}")
    print(f"Output CSV  : {args.output_csv}")


if __name__ == "__main__":
    main()
