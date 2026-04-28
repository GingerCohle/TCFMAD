#!/usr/bin/env python3
"""Dry-run cropper for converting PCBMarket COCO data into VisA-like OK/KO crops."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path(
            "/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/pcbmarket"
        ),
    )
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    parser.add_argument("--num_images_train", type=int, default=20)
    parser.add_argument("--num_images_val", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--min_side", type=int, default=384)
    parser.add_argument("--max_side", type=int, default=768)
    parser.add_argument("--ok_per_ko", type=int, default=1)
    parser.add_argument("--iou_thr", type=float, default=0.0)
    parser.add_argument("--max_try", type=int, default=50)
    parser.add_argument("--max_anns_per_image", type=int, default=6)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/pcbmarket_dryrun"))
    return parser.parse_args()


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def discover_paths(dataset_root: Path) -> dict:
    json_candidates = sorted(dataset_root.rglob("*.json"))
    image_dirs = []
    for path in sorted(dataset_root.rglob("*")):
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
            ranked.append((score, str(path), path))
        ranked.sort(reverse=True)
        for score, _, path in ranked:
            if keyword in path.stem.lower():
                return path
        if ranked:
            return ranked[0][2]
        raise RuntimeError(f"No json candidates found for split={keyword}")

    train_json = choose_json("train")
    val_json = choose_json("val")

    train_data = json.loads(train_json.read_text())
    val_data = json.loads(val_json.read_text())

    def infer_img_root(data: dict) -> Path:
        sample_names = [img["file_name"] for img in data.get("images", [])[:20]]
        ranked = []
        for path in image_dirs:
            hits = sum(1 for file_name in sample_names if (path / file_name).exists())
            ranked.append((hits, str(path), path))
        ranked.sort(reverse=True)
        if not ranked or ranked[0][0] == 0:
            raise RuntimeError("Could not infer image root from sample file names.")
        return ranked[0][2]

    train_img_root = infer_img_root(train_data)
    val_img_root = infer_img_root(val_data)
    return {
        "json_candidates": json_candidates,
        "image_dirs": image_dirs,
        "train_json": train_json,
        "val_json": val_json,
        "train_img_root": train_img_root,
        "val_img_root": val_img_root,
    }


def load_coco(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as handle:
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


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


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
    side = int(clip(side, min_side, max_side))
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
        max_iou = 0.0
        for bbox in bbox_xyxy:
            max_iou = max(max_iou, box_iou(crop_box, bbox))
            if max_iou > iou_thr:
                break
        if max_iou <= iou_thr:
            return crop_box, attempt
    return None, max_try


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int]) -> None:
    x, y = xy
    draw.rectangle((x - 1, y - 1, x + 7 * len(text) + 3, y + 11), fill=fill)
    draw.text((x + 1, y), text, fill=(255, 255, 255))


def make_original_overlay(image: Image.Image, anns: list[dict], category_by_id: dict[int, str]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for ann in anns:
        x, y, w, h = ann["bbox"]
        rect = (int(x), int(y), int(x + w), int(y + h))
        draw.rectangle(rect, outline=(255, 0, 0), width=4)
        draw_label(draw, (rect[0] + 2, max(0, rect[1] - 14)), category_by_id[ann["category_id"]], (255, 0, 0))
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
        draw_label(draw, (rel_int[0] + 2, max(0, rel_int[1] - 14)), category_by_id[ann["category_id"]], (180, 0, 0))
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
    x = 0
    labels = ["original", "ko", "ok"]
    for img, label in zip(images, labels):
        panel.paste(img, (x, 0))
        draw_label(draw, (x + 4, 4), label, (60, 60, 60))
        x += img.width + gap
    return panel


def summarize_numeric(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


def process_split(
    split_name: str,
    json_path: Path,
    img_root: Path,
    args: argparse.Namespace,
    out_dir: Path,
    seed_offset: int,
) -> dict:
    data = load_coco(json_path)
    indices = build_indices(data)
    image_ids = sorted(indices["image_by_id"])
    target_num = args.num_images_train if split_name == "train" else args.num_images_val
    rng = random.Random(args.seed + seed_offset)
    sampled_ids = rng.sample(image_ids, min(target_num, len(image_ids)))

    split_dirs = {
        "ko": out_dir / split_name / "ko",
        "ok": out_dir / split_name / "ok",
        "panels": out_dir / split_name / "panels",
    }
    for path in split_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    ko_ratios = []
    ok_tries = []
    clamped_count = 0
    ko_count = 0
    ok_count = 0
    ok_failures = 0
    panel_count = 0
    sampled_files = []
    ko_examples = []
    ok_examples = []

    for image_id in sampled_ids:
        image_meta = indices["image_by_id"][image_id]
        image_path = img_root / image_meta["file_name"]
        sampled_files.append(str(image_path.resolve()))
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            print(f"[WARN] {split_name} image open failed: {image_path} ({exc})")
            continue

        anns = indices["anns_by_image"].get(image_id, [])
        anns = anns[: args.max_anns_per_image]
        original_overlay = make_original_overlay(image, anns, indices["category_by_id"])
        first_ko = None
        first_ok = None

        for ann_idx, ann in enumerate(anns):
            category_name = indices["category_by_id"][ann["category_id"]]
            crop_box, side, clamped = compute_square_crop(
                ann["bbox"],
                image_meta["width"],
                image_meta["height"],
                args.alpha,
                args.min_side,
                args.max_side,
            )
            if clamped:
                clamped_count += 1
            bbox_area = ann["bbox"][2] * ann["bbox"][3]
            crop_area = (crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1])
            if crop_area > 0:
                ko_ratios.append(bbox_area / crop_area)

            ko_image = make_ko_overlay(image, crop_box, ann, category_name)
            ko_name = (
                f"{split_name}_img{image_id}_ann{ann['id']}_{category_name}_side{side}_ko.jpg"
            )
            ko_path = split_dirs["ko"] / ko_name
            ko_image.save(ko_path, quality=95)
            ko_count += 1
            if len(ko_examples) < 5:
                ko_examples.append(str(ko_path.resolve()))
            if first_ko is None:
                first_ko = ko_image

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
                    ok_failures += 1
                    print(
                        f"[WARN] {split_name} image_id={image_id} ann_id={ann['id']} "
                        f"failed to find OK crop after {args.max_try} tries"
                    )
                    continue
                ok_tries.append(tries)
                ok_image = make_ok_overlay(image, ok_crop_box, anns, indices["category_by_id"])
                ok_name = f"{split_name}_img{image_id}_ok{ann_idx}_{ok_idx}_side{side}.jpg"
                ok_path = split_dirs["ok"] / ok_name
                ok_image.save(ok_path, quality=95)
                ok_count += 1
                if len(ok_examples) < 5:
                    ok_examples.append(str(ok_path.resolve()))
                if first_ok is None:
                    first_ok = ok_image

        panel = make_panel(original_overlay, first_ko, first_ok)
        panel_path = split_dirs["panels"] / f"{split_name}_img{image_id}_panel.jpg"
        panel.save(panel_path, quality=95)
        panel_count += 1

    ratio_stats = summarize_numeric(ko_ratios)
    ok_try_stats = summarize_numeric(ok_tries)
    return {
        "split": split_name,
        "ann_json": str(json_path.resolve()),
        "img_root": str(img_root.resolve()),
        "sampled_image_count": len(sampled_ids),
        "sampled_images": sampled_files,
        "ko_count": ko_count,
        "ok_count": ok_count,
        "panel_count": panel_count,
        "ko_ratio_stats": ratio_stats,
        "clamped_fraction": (clamped_count / ko_count) if ko_count else 0.0,
        "avg_ok_tries": ok_try_stats["mean"],
        "ok_try_stats": ok_try_stats,
        "ok_failures": ok_failures,
        "ko_examples": ko_examples,
        "ok_examples": ok_examples,
    }


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = discover_paths(dataset_root)
    print(f"dataset_root: {dataset_root}")
    print("json_candidates:")
    for path in paths["json_candidates"]:
        print(f"  - {path.resolve()}")
    print("image_dirs:")
    for path in paths["image_dirs"]:
        print(f"  - {path.resolve()}")
    print(f"selected_train_json: {paths['train_json'].resolve()}")
    print(f"selected_val_json: {paths['val_json'].resolve()}")
    print(f"selected_train_img_root: {paths['train_img_root'].resolve()}")
    print(f"selected_val_img_root: {paths['val_img_root'].resolve()}")

    split_order = ["train", "val"] if args.split == "both" else [args.split]
    summary = {
        "dataset_root": str(dataset_root),
        "out_dir": str(out_dir),
        "args": json_ready(vars(args)),
        "discovered": {
            "json_candidates": [str(p.resolve()) for p in paths["json_candidates"]],
            "image_dirs": [str(p.resolve()) for p in paths["image_dirs"]],
            "train_json": str(paths["train_json"].resolve()),
            "val_json": str(paths["val_json"].resolve()),
            "train_img_root": str(paths["train_img_root"].resolve()),
            "val_img_root": str(paths["val_img_root"].resolve()),
        },
        "splits": {},
    }

    for idx, split_name in enumerate(split_order):
        result = process_split(
            split_name=split_name,
            json_path=paths[f"{split_name}_json"],
            img_root=paths[f"{split_name}_img_root"],
            args=args,
            out_dir=out_dir,
            seed_offset=idx * 1000,
        )
        summary["splits"][split_name] = result
        print(f"[{split_name}]")
        print(f"  sampled_image_count: {result['sampled_image_count']}")
        print(f"  ko_count: {result['ko_count']}")
        print(f"  ok_count: {result['ok_count']}")
        print(f"  panel_count: {result['panel_count']}")
        print(
            "  ko_bbox_area_over_crop_area_mean_min_max: "
            f"{result['ko_ratio_stats']['mean']:.4f} / "
            f"{result['ko_ratio_stats']['min']:.4f} / "
            f"{result['ko_ratio_stats']['max']:.4f}"
        )
        print(f"  clamped_fraction: {result['clamped_fraction']:.4f}")
        print(
            "  ok_try_mean_min_max: "
            f"{result['ok_try_stats']['mean']:.2f} / "
            f"{result['ok_try_stats']['min']:.0f} / "
            f"{result['ok_try_stats']['max']:.0f}"
        )
        print(f"  ok_failures: {result['ok_failures']}")
        print("  ko_examples:")
        for item in result["ko_examples"]:
            print(f"    - {item}")
        print("  ok_examples:")
        for item in result["ok_examples"]:
            print(f"    - {item}")

    total_outputs = sum(
        len(list((out_dir / split / kind).glob("*.jpg")))
        for split in split_order
        for kind in ["ko", "ok", "panels"]
    )
    summary["total_output_images"] = total_outputs
    summary_path = out_dir / "dryrun_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"total_output_images: {total_outputs}")
    print(f"summary_json: {summary_path.resolve()}")
    print(f"outputs_saved_to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
