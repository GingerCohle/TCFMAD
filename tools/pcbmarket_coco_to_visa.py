#!/usr/bin/env python3
"""Convert PCBMarket COCO detection data into a VisA-style anomaly dataset."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
if hasattr(Image, "Resampling"):
    BILINEAR = Image.Resampling.BILINEAR
    NEAREST = Image.Resampling.NEAREST
else:
    BILINEAR = Image.BILINEAR
    NEAREST = Image.NEAREST


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src_root",
        type=Path,
        default=Path(
            "/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/pcbmarket"
        ),
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path(
            "/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/pcbmarket_visa"
        ),
    )
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--min_side", type=int, default=384)
    parser.add_argument("--max_side", type=int, default=768)
    parser.add_argument("--resize", type=int, default=512)
    parser.add_argument("--ok_per_ko", type=int, default=1)
    parser.add_argument("--iou_thr", type=float, default=0.0)
    parser.add_argument("--max_try", type=int, default=50)
    parser.add_argument("--max_anns_per_image", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write_verify", type=parse_bool, default=True)
    parser.add_argument("--verify_max", type=int, default=50)
    return parser.parse_args()


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def sanitize_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return safe.strip("._") or "class"


def discover_paths(src_root: Path) -> dict:
    json_candidates = sorted(src_root.rglob("*.json"))
    image_dirs = []
    for path in sorted(src_root.rglob("*")):
        if not path.is_dir():
            continue
        try:
            if any(child.suffix.lower() in IMAGE_EXTS for child in path.iterdir()):
                image_dirs.append(path)
        except OSError:
            continue
    image_dirs = sorted(set(image_dirs))

    def choose_json(keyword: str) -> Path:
        ranked = []
        for path in json_candidates:
            stem = path.stem.lower()
            score = 0
            if "instances" in stem:
                score += 10
            if keyword in stem:
                score += 5
            if "annotations" in str(path.parent).lower():
                score += 2
            ranked.append((score, str(path), path))
        ranked.sort(reverse=True)
        for score, _, path in ranked:
            if keyword in path.stem.lower():
                return path
        if ranked:
            return ranked[0][2]
        raise RuntimeError(f"No json annotation file found for split={keyword}")

    train_json = choose_json("train")
    val_json = choose_json("val")
    train_data = json.loads(train_json.read_text())
    val_data = json.loads(val_json.read_text())

    def infer_img_root(data: dict, split_name: str) -> Path:
        sample_names = [img["file_name"] for img in data.get("images", [])[:20] if img.get("file_name")]
        ranked = []
        for path in image_dirs:
            score = 0
            path_lower = str(path).lower()
            if split_name in path_lower:
                score += 10
            if path.name.lower() == "images":
                score += 3
            hits = sum(1 for file_name in sample_names if (path / file_name).exists())
            ranked.append((hits + score, hits, str(path), path))
        ranked.sort(reverse=True)
        for total_score, hits, _, path in ranked:
            if hits > 0:
                return path
        raise RuntimeError(f"Could not infer image root for split={split_name}")

    train_img_root = infer_img_root(train_data, "train")
    val_img_root = infer_img_root(val_data, "val")
    return {
        "json_candidates": json_candidates,
        "image_dirs": image_dirs,
        "train_json": train_json,
        "val_json": val_json,
        "train_img_root": train_img_root,
        "val_img_root": val_img_root,
    }


def load_coco(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_indices(data: dict) -> dict:
    image_by_id = {img["id"]: img for img in data.get("images", [])}
    anns_by_image = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)
    category_by_id = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    return {
        "image_by_id": image_by_id,
        "anns_by_image": anns_by_image,
        "category_by_id": category_by_id,
    }


def bbox_to_xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    inter = intersection_area(a, b)
    if inter <= 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def compute_square_crop(
    bbox: list[float],
    image_w: int,
    image_h: int,
    alpha: float,
    min_side: int,
    max_side: int,
) -> tuple[tuple[int, int, int, int], int, bool]:
    _, _, w, h = bbox
    side = int(round(alpha * max(w, h)))
    side = max(min_side, min(max_side, side))
    side = min(side, image_w, image_h)
    side = max(1, side)
    x, y, bw, bh = bbox
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = x1 + side
    y2 = y1 + side
    clamped = False
    if x1 < 0:
        x2 -= x1
        x1 = 0
        clamped = True
    if y1 < 0:
        y2 -= y1
        y1 = 0
        clamped = True
    if x2 > image_w:
        shift = x2 - image_w
        x1 -= shift
        x2 = image_w
        clamped = True
    if y2 > image_h:
        shift = y2 - image_h
        y1 -= shift
        y2 = image_h
        clamped = True
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_w, x2)
    y2 = min(image_h, y2)
    return (x1, y1, x2, y2), min(x2 - x1, y2 - y1), clamped


def relative_box(
    bbox: list[float],
    crop_box: tuple[int, int, int, int],
) -> tuple[float, float, float, float] | None:
    crop_xyxy = (crop_box[0], crop_box[1], crop_box[2], crop_box[3])
    bbox_xyxy = bbox_to_xyxy(bbox)
    if intersection_area(crop_xyxy, bbox_xyxy) <= 0:
        return None
    x1 = max(crop_box[0], bbox_xyxy[0]) - crop_box[0]
    y1 = max(crop_box[1], bbox_xyxy[1]) - crop_box[1]
    x2 = min(crop_box[2], bbox_xyxy[2]) - crop_box[0]
    y2 = min(crop_box[3], bbox_xyxy[3]) - crop_box[1]
    return (x1, y1, x2, y2)


def sample_ok_crop(
    image_w: int,
    image_h: int,
    side: int,
    anns: list[dict],
    rng: random.Random,
    max_try: int,
    iou_thr: float,
) -> tuple[tuple[int, int, int, int] | None, int]:
    max_x = max(0, image_w - side)
    max_y = max(0, image_h - side)
    bbox_xyxy = [bbox_to_xyxy(ann["bbox"]) for ann in anns]
    for attempt in range(1, max_try + 1):
        x1 = rng.randint(0, max_x) if max_x > 0 else 0
        y1 = rng.randint(0, max_y) if max_y > 0 else 0
        crop_box = (x1, y1, x1 + side, y1 + side)
        if all(box_iou(crop_box, bbox) <= iou_thr for bbox in bbox_xyxy):
            return crop_box, attempt
    return None, max_try


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int]) -> None:
    x, y = xy
    draw.rectangle((x - 1, y - 1, x + 7 * len(text) + 4, y + 11), fill=fill)
    draw.text((x + 1, y), text, fill=(255, 255, 255))


def make_original_overlay(image: Image.Image, anns: list[dict], category_by_id: dict[int, str]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for ann in anns:
        x, y, w, h = ann["bbox"]
        rect = (int(x), int(y), int(x + w), int(y + h))
        draw.rectangle(rect, outline=(255, 0, 0), width=4)
        draw_label(draw, (rect[0] + 2, max(0, rect[1] - 14)), category_by_id.get(ann["category_id"], str(ann["category_id"])), (255, 0, 0))
    return canvas


def make_ko_overlay(
    image: Image.Image,
    crop_box: tuple[int, int, int, int],
    ann: dict,
    category_name: str,
) -> Image.Image:
    crop = image.crop(crop_box).convert("RGBA")
    overlay = Image.new("RGBA", crop.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rel = relative_box(ann["bbox"], crop_box)
    if rel is not None:
        rel_int = (int(rel[0]), int(rel[1]), int(rel[2]), int(rel[3]))
        draw.rectangle(rel_int, fill=(255, 255, 255, 80), outline=(0, 255, 0, 255), width=4)
        draw_label(draw, (rel_int[0] + 2, max(0, rel_int[1] - 14)), category_name, (0, 160, 0))
    return Image.alpha_composite(crop, overlay).convert("RGB")


def make_ok_overlay(
    image: Image.Image,
    crop_box: tuple[int, int, int, int],
    anns: list[dict],
    category_by_id: dict[int, str],
) -> Image.Image:
    crop = image.crop(crop_box).convert("RGB")
    draw = ImageDraw.Draw(crop)
    for ann in anns:
        rel = relative_box(ann["bbox"], crop_box)
        if rel is None:
            continue
        rel_int = (int(rel[0]), int(rel[1]), int(rel[2]), int(rel[3]))
        draw.rectangle(rel_int, outline=(255, 0, 0), width=4)
        draw_label(draw, (rel_int[0] + 2, max(0, rel_int[1] - 14)), category_by_id.get(ann["category_id"], str(ann["category_id"])), (180, 0, 0))
    return crop


def fit_to_height(image: Image.Image, target_height: int) -> Image.Image:
    if image.height == target_height:
        return image
    ratio = target_height / image.height
    width = max(1, int(round(image.width * ratio)))
    return image.resize((width, target_height))


def make_panel(original: Image.Image, ko: Image.Image | None, ok: Image.Image | None) -> Image.Image:
    target_height = 320
    original_fit = fit_to_height(original, target_height)
    ko_fit = fit_to_height(ko, target_height) if ko is not None else Image.new("RGB", (target_height, target_height), (32, 32, 32))
    ok_fit = fit_to_height(ok, target_height) if ok is not None else Image.new("RGB", (target_height, target_height), (32, 32, 32))
    images = [original_fit, ko_fit, ok_fit]
    gap = 12
    width = sum(img.width for img in images) + gap * (len(images) - 1)
    panel = Image.new("RGB", (width, target_height), (18, 18, 18))
    draw = ImageDraw.Draw(panel)
    labels = ["original", "ko", "ok"]
    x = 0
    for img, label in zip(images, labels):
        panel.paste(img, (x, 0))
        draw_label(draw, (x + 4, 4), label, (60, 60, 60))
        x += img.width + gap
    return panel


def ensure_class_dirs(out_root: Path, class_names: list[str]) -> dict[str, dict[str, Path]]:
    class_dirs = {}
    for name in class_names:
        safe = sanitize_name(name)
        root = out_root / safe
        paths = {
            "train_ok": root / "train" / "ok",
            "test_ok": root / "test" / "ok",
            "test_ko": root / "test" / "ko",
            "gt_ko": root / "ground_truth" / "ko",
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        class_dirs[name] = paths
    return class_dirs


def summarize_numeric(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


def verify_one_category_per_image(indices: dict, split_name: str) -> tuple[int, list[dict]]:
    violations = []
    for image_id, anns in indices["anns_by_image"].items():
        category_ids = sorted({ann["category_id"] for ann in anns})
        if len(category_ids) > 1:
            violations.append(
                {
                    "split": split_name,
                    "image_id": image_id,
                    "file_name": indices["image_by_id"].get(image_id, {}).get("file_name", ""),
                    "category_ids": category_ids,
                    "category_names": [indices["category_by_id"].get(cat_id, str(cat_id)) for cat_id in category_ids],
                }
            )
    return len(violations), violations


def process_split(
    split_name: str,
    data: dict,
    img_root: Path,
    class_dirs: dict[str, dict[str, Path]],
    args: argparse.Namespace,
    rng: random.Random,
    verify_dir: Path,
    verify_budget: list[int],
    stats: dict,
) -> None:
    indices = build_indices(data)
    bbox_ratios = stats["bbox_area_over_crop_area"]
    ok_tries = stats["ok_try_counts"]
    image_ids = sorted(indices["image_by_id"])

    for image_id in image_ids:
        image_meta = indices["image_by_id"][image_id]
        image_path = img_root / image_meta["file_name"]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            stats["missing_images"].append({"split": split_name, "image_id": image_id, "path": str(image_path), "error": str(exc)})
            print(f"[WARN] missing image {split_name} image_id={image_id}: {image_path} ({exc})")
            continue

        anns = indices["anns_by_image"].get(image_id, [])
        anns = anns[: args.max_anns_per_image]
        if not anns:
            continue

        first_ko_overlay = None
        first_ok_overlay = None
        original_overlay = None

        for ann_idx, ann in enumerate(anns):
            category_name = indices["category_by_id"].get(ann["category_id"], str(ann["category_id"]))
            safe_name = sanitize_name(category_name)
            stats["crop_anchor_count"] += 1
            crop_box, side, clamped = compute_square_crop(
                ann["bbox"],
                image_meta["width"],
                image_meta["height"],
                args.alpha,
                args.min_side,
                args.max_side,
            )
            if clamped:
                stats["clamped_count"] += 1
            bbox_area = float(ann["bbox"][2]) * float(ann["bbox"][3])
            crop_area = (crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1])
            if crop_area > 0:
                bbox_ratios.append(bbox_area / crop_area)

            if split_name == "val":
                ko_crop = image.crop(crop_box).resize((args.resize, args.resize), BILINEAR)
                rel = relative_box(ann["bbox"], crop_box)
                mask = Image.new("L", (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]), 0)
                if rel is not None:
                    mask_draw = ImageDraw.Draw(mask)
                    rect = (int(rel[0]), int(rel[1]), int(rel[2]), int(rel[3]))
                    mask_draw.rectangle(rect, fill=255)
                mask = mask.resize((args.resize, args.resize), NEAREST)
                stem = f"ko_val_img{image_id}_ann{ann['id']}_{safe_name}_s{side}"
                ko_path = class_dirs[category_name]["test_ko"] / f"{stem}.jpg"
                mask_path = class_dirs[category_name]["gt_ko"] / f"{stem}.png"
                ko_crop.save(ko_path, quality=95)
                mask.save(mask_path)
                stats["per_class"][category_name]["test_ko"] += 1
                stats["per_class"][category_name]["gt_ko"] += 1
                if first_ko_overlay is None:
                    first_ko_overlay = make_ko_overlay(image, crop_box, ann, category_name)

            for ok_idx in range(args.ok_per_ko):
                ok_crop_box, tries = sample_ok_crop(
                    image_meta["width"],
                    image_meta["height"],
                    side,
                    anns,
                    rng,
                    args.max_try,
                    args.iou_thr,
                )
                if ok_crop_box is None:
                    stats["ok_failures"] += 1
                    stats["ok_failure_examples"].append(
                        {
                            "split": split_name,
                            "image_id": image_id,
                            "ann_id": ann["id"],
                            "category": category_name,
                            "side": side,
                        }
                    )
                    print(
                        f"[WARN] OK crop failure split={split_name} image_id={image_id} "
                        f"ann_id={ann['id']} category={safe_name} side={side}"
                    )
                    continue
                ok_tries.append(tries)
                ok_crop = image.crop(ok_crop_box).resize((args.resize, args.resize), BILINEAR)
                prefix = "ok_train" if split_name == "train" else "ok_val"
                stem = f"{prefix}_img{image_id}_ann{ann['id']}_k{ok_idx}_{safe_name}_s{side}"
                target_key = "train_ok" if split_name == "train" else "test_ok"
                out_path = class_dirs[category_name][target_key] / f"{stem}.jpg"
                ok_crop.save(out_path, quality=95)
                stats["per_class"][category_name][target_key] += 1
                if first_ok_overlay is None:
                    first_ok_overlay = make_ok_overlay(image, ok_crop_box, anns, indices["category_by_id"])

        if (
            args.write_verify
            and verify_budget[0] < args.verify_max
            and first_ko_overlay is not None
            and first_ok_overlay is not None
        ):
            if original_overlay is None:
                original_overlay = make_original_overlay(image, anns, indices["category_by_id"])
            panel = make_panel(original_overlay, first_ko_overlay, first_ok_overlay)
            panel_path = verify_dir / f"{split_name}_img{image_id}_panel.jpg"
            panel.save(panel_path, quality=95)
            verify_budget[0] += 1
            stats["verify_panels"].append(str(panel_path.resolve()))


def main() -> int:
    args = parse_args()
    src_root = args.src_root.resolve()
    out_root = args.out_root.resolve()
    verify_dir = Path("/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_visa_verify")
    verify_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    paths = discover_paths(src_root)
    print(f"src_root: {src_root}")
    print("expected_layout_check:")
    for path in [
        src_root / "annotations" / "instances_train.json",
        src_root / "annotations" / "instances_val.json",
        src_root / "train2017",
        src_root / "val2017",
    ]:
        print(f"  - {path}: exists={path.exists()}")
    print("discovered_json_candidates:")
    for path in paths["json_candidates"]:
        print(f"  - {path.resolve()}")
    print("discovered_image_dirs:")
    for path in paths["image_dirs"]:
        print(f"  - {path.resolve()}")
    print(f"selected_train_json: {paths['train_json'].resolve()}")
    print(f"selected_val_json: {paths['val_json'].resolve()}")
    print(f"selected_train_img_root: {paths['train_img_root'].resolve()}")
    print(f"selected_val_img_root: {paths['val_img_root'].resolve()}")
    print(f"out_root: {out_root}")
    print(f"verify_dir: {verify_dir.resolve()}")

    train_data = load_coco(paths["train_json"])
    val_data = load_coco(paths["val_json"])
    train_indices = build_indices(train_data)
    val_indices = build_indices(val_data)
    category_names = sorted(
        {cat["name"] for cat in train_data.get("categories", [])}
        | {cat["name"] for cat in val_data.get("categories", [])}
    )
    class_dirs = ensure_class_dirs(out_root, category_names)

    train_violation_count, train_violations = verify_one_category_per_image(train_indices, "train")
    val_violation_count, val_violations = verify_one_category_per_image(val_indices, "val")
    print(f"train_one_category_violations: {train_violation_count}")
    print(f"val_one_category_violations: {val_violation_count}")

    stats = {
        "src_root": str(src_root),
        "out_root": str(out_root),
        "verify_dir": str(verify_dir.resolve()),
        "args": json_ready(vars(args)),
        "selected_paths": json_ready(paths),
        "category_names": category_names,
        "one_category_violations": {
            "train_count": train_violation_count,
            "val_count": val_violation_count,
            "examples": train_violations[:20] + val_violations[:20],
        },
        "per_class": {
            name: {"train_ok": 0, "test_ok": 0, "test_ko": 0, "gt_ko": 0}
            for name in category_names
        },
        "bbox_area_over_crop_area": [],
        "ok_try_counts": [],
        "ok_failures": 0,
        "ok_failure_examples": [],
        "clamped_count": 0,
        "crop_anchor_count": 0,
        "missing_images": [],
        "verify_panels": [],
    }

    verify_budget = [0]
    rng = random.Random(args.seed)
    process_split(
        split_name="train",
        data=train_data,
        img_root=paths["train_img_root"],
        class_dirs=class_dirs,
        args=args,
        rng=rng,
        verify_dir=verify_dir,
        verify_budget=verify_budget,
        stats=stats,
    )
    process_split(
        split_name="val",
        data=val_data,
        img_root=paths["val_img_root"],
        class_dirs=class_dirs,
        args=args,
        rng=rng,
        verify_dir=verify_dir,
        verify_budget=verify_budget,
        stats=stats,
    )

    bbox_ratio_stats = summarize_numeric(stats["bbox_area_over_crop_area"])
    ok_try_stats = summarize_numeric(stats["ok_try_counts"])
    stats["bbox_ratio_stats"] = bbox_ratio_stats
    stats["ok_try_stats"] = ok_try_stats
    stats["clamped_fraction"] = (
        stats["clamped_count"] / stats["crop_anchor_count"]
        if stats["crop_anchor_count"]
        else 0.0
    )

    print("per_class_counts:")
    for class_name in category_names:
        values = stats["per_class"][class_name]
        print(
            f"  - {class_name}: train/ok={values['train_ok']} "
            f"test/ok={values['test_ok']} test/ko={values['test_ko']} gt/ko={values['gt_ko']}"
        )
    print(
        "bbox_area_over_crop_area_mean_min_max: "
        f"{bbox_ratio_stats['mean']:.4f} / {bbox_ratio_stats['min']:.4f} / {bbox_ratio_stats['max']:.4f}"
    )
    print(
        "ok_try_mean_min_max: "
        f"{ok_try_stats['mean']:.2f} / {ok_try_stats['min']:.0f} / {ok_try_stats['max']:.0f}"
    )
    print(f"ok_failures: {stats['ok_failures']}")
    print(f"clamped_fraction: {stats['clamped_fraction']:.4f}")
    print(f"verify_panels_written: {len(stats['verify_panels'])}")

    summary_path = out_root / "_conversion_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(stats), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"summary_json: {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
