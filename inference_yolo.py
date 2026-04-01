#!/usr/bin/env python3
"""Run YOLO inference and export predictions as submission CSV.

This script is variable-config based (no argparse).
Update the CONFIG section, then run:
  python inference_yolo.py
"""

from __future__ import annotations

import csv
import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


# CONFIG
ROOT = Path(__file__).resolve().parent

YOLO_WEIGHTS = ROOT / "runs/no_blur/no_blur.pt"
TEST_IMG_DIR = ROOT / "data/test_images"
ANN_DIR = ROOT / "data/new_merged_annonation/new_merged_annonation"
OUTPUT_CSV = ROOT / "runs/no_blur/submission.csv"
DEVICE = ""
IMGSZ = 640
BATCH_SIZE = 1
PREDICT_CHUNK_SIZE = 16
USE_FP16_ON_CUDA = True

AUTO_RECOVER_FROM_OOM = True
OOM_RETRY_IMGSZ = (640, 512, 448, 384)

SCORE_THR = 0.05
NMS_IOU_THR = 0.5
AGNOSTIC_NMS = True
MAX_DETS_PER_IMG = 4

EXTS = (".png", ".jpg", ".jpeg", ".bmp")

CATEGORY_ID_FORMAT = "dl_idx" 


@dataclass(frozen=True)
class TestImageInfo:
    path: Path
    image_id: int


def parse_image_id_from_name(path: Path) -> int:
    match = re.search(r"\d+", path.stem)
    if match is None:
        raise ValueError(f"Cannot parse numeric image_id from file name: {path.name}")
    return int(match.group(0))


def load_test_images(test_img_dir: Path, exts: Sequence[str]) -> List[TestImageInfo]:
    ext_set = {x.lower().strip() for x in exts if x.strip()}
    if not ext_set:
        raise ValueError("No valid image extensions were provided.")
    if not test_img_dir.exists():
        raise FileNotFoundError(f"Test image dir not found: {test_img_dir}")

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


def is_cuda_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg and "cuda" in msg


def build_imgsz_try_list() -> List[int]:
    candidates: List[int] = [int(IMGSZ)]
    for sz in OOM_RETRY_IMGSZ:
        sz_i = int(sz)
        if sz_i not in candidates:
            candidates.append(sz_i)
    candidates = sorted([x for x in candidates if x > 0], reverse=True)
    return candidates


def chunked(items: Sequence, chunk_size: int):
    if chunk_size <= 0:
        raise ValueError("PREDICT_CHUNK_SIZE must be > 0.")
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def predict_with_oom_recovery(model, source_paths: List[str]):
    try:
        import torch
    except Exception:
        torch = None

    requested_device = DEVICE if DEVICE else None
    use_fp16 = USE_FP16_ON_CUDA and (DEVICE == "" or str(DEVICE).startswith("cuda"))
    imgsz_candidates = build_imgsz_try_list()

    last_exc: Exception | None = None
    for idx, imgsz in enumerate(imgsz_candidates):
        try:
            return model.predict(
                source=source_paths,
                conf=SCORE_THR,
                iou=NMS_IOU_THR,
                agnostic_nms=AGNOSTIC_NMS,
                max_det=MAX_DETS_PER_IMG if MAX_DETS_PER_IMG > 0 else 30000,
                imgsz=imgsz,
                batch=BATCH_SIZE if idx == 0 else 1,
                device=requested_device,
                half=use_fp16,
                verbose=False,
                stream=False,
            )
        except RuntimeError as exc:
            last_exc = exc
            if not AUTO_RECOVER_FROM_OOM or not is_cuda_oom_error(exc):
                raise
            print(
                f"[OOM] CUDA 메모리 부족. imgsz={imgsz}에서 실패 -> "
                "더 작은 해상도로 재시도합니다."
            )
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not AUTO_RECOVER_FROM_OOM or last_exc is None:
        raise RuntimeError("Prediction failed for unknown reason.")

    print("[OOM] GPU 재시도 실패. CPU로 전환해서 추론합니다 (느릴 수 있음).")
    # final fallback on CPU
    return model.predict(
        source=source_paths,
        conf=SCORE_THR,
        iou=NMS_IOU_THR,
        agnostic_nms=AGNOSTIC_NMS,
        max_det=MAX_DETS_PER_IMG if MAX_DETS_PER_IMG > 0 else 30000,
        imgsz=min(build_imgsz_try_list()),
        batch=1,
        device="cpu",
        half=False,
        verbose=False,
        stream=False,
    )


def run_inference_yolo(
    model,
    image_infos: List[TestImageInfo],
    category_id_format: str,
    remapped_to_original: dict[int, int] | None,
) -> List[dict]:
    rows: List[dict] = []

    try:
        import torch
    except Exception:
        torch = None

    for info_chunk in chunked(image_infos, PREDICT_CHUNK_SIZE):
        source_paths = [str(info.path) for info in info_chunk]
        results = predict_with_oom_recovery(model=model, source_paths=source_paths)

        for info, result in zip(info_chunk, results):
            h, w = result.orig_shape[:2]
            image_id = info.image_id

            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            classes = boxes.cls.cpu().tolist()

            for box, score, cls_idx in zip(xyxy, confs, classes):
                x1, y1, x2, y2 = map(float, box)
                x1, y1, x2, y2 = clamp_box_xyxy(x1, y1, x2, y2, width=w, height=h)
                bw = x2 - x1
                bh = y2 - y1
                if bw <= 0.0 or bh <= 0.0:
                    continue

                remapped_id = int(cls_idx)
                output_category_id = convert_category_id(
                    remapped_category_id=remapped_id,
                    category_id_format=category_id_format,
                    remapped_to_original=remapped_to_original,
                )

                rows.append(
                    {
                        "image_id": image_id,
                        "category_id": output_category_id,
                        "bbox_x": round(x1, 4),
                        "bbox_y": round(y1, 4),
                        "bbox_w": round(bw, 4),
                        "bbox_h": round(bh, 4),
                        "score": round(float(score), 6),
                    }
                )

        del results
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "ultralytics is not installed. Install first:\n"
            "  pip install ultralytics"
        ) from exc

    if not YOLO_WEIGHTS.exists():
        raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS}")

    category_id_format = CATEGORY_ID_FORMAT.strip()
    valid_formats = {"dl_idx", "original", "train_json", "remapped", "one_based"}
    if category_id_format not in valid_formats:
        raise ValueError(
            f"Invalid CATEGORY_ID_FORMAT: {category_id_format}. "
            f"Choose one of {sorted(valid_formats)}"
        )

    remapped_to_original = None
    if category_id_format in {"dl_idx", "original"}:
        if not ANN_DIR.exists():
            raise FileNotFoundError(f"Annotation dir not found: {ANN_DIR}")
        remapped_to_original = load_remapped_to_original_id_map(ANN_DIR)

    image_infos = load_test_images(TEST_IMG_DIR, EXTS)
    model = YOLO(str(YOLO_WEIGHTS))

    rows = run_inference_yolo(
        model=model,
        image_infos=image_infos,
        category_id_format=category_id_format,
        remapped_to_original=remapped_to_original,
    )
    save_submission_csv(OUTPUT_CSV, rows)

    print(f"YOLO weights : {YOLO_WEIGHTS}")
    print(f"Test images  : {len(image_infos)}")
    print(f"Pred rows    : {len(rows)}")
    print(f"Category ID  : {category_id_format}")
    print(f"Output CSV   : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
