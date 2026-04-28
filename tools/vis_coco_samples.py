#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

DEFAULT_DATASET_ROOT = Path("/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/DsPCBSD")


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_category_ids(text: Optional[str]) -> Optional[set[int]]:
    if not text:
        return None
    out: set[int] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        out.add(int(item))
    return out or None


@dataclass
class CocoIndex:
    images: Dict[int, Dict[str, Any]]
    anns_by_image: Dict[int, List[Dict[str, Any]]]
    categories: Dict[int, str]
    source: str


def discover_annotation_json(dataset_root: Path) -> Tuple[List[Path], Path]:
    candidates = sorted(
        p for p in dataset_root.rglob("*.json") if p.is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"No JSON annotations found under {dataset_root}")

    def score(path: Path) -> Tuple[int, str]:
        name = path.name.lower()
        rel = str(path.relative_to(dataset_root))
        if name == "instances_val.json":
            return (0, rel)
        if name == "instances_train.json":
            return (1, rel)
        if name.startswith("instances_"):
            return (2, rel)
        if "annotations" in rel:
            return (3, rel)
        return (4, rel)

    selected = sorted(candidates, key=score)[0]
    return candidates, selected


def load_coco_index(ann_json: Path) -> CocoIndex:
    try:
        from pycocotools.coco import COCO  # type: ignore

        coco = COCO(str(ann_json))
        images = {int(img_id): coco.loadImgs([img_id])[0] for img_id in coco.getImgIds()}
        anns_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for ann_id in coco.getAnnIds():
            ann = coco.anns[ann_id]
            anns_by_image[int(ann["image_id"])].append(ann)
        categories = {int(cat_id): coco.cats[cat_id]["name"] for cat_id in coco.getCatIds()}
        return CocoIndex(images=images, anns_by_image=anns_by_image, categories=categories, source="pycocotools")
    except Exception:
        data = json.loads(ann_json.read_text())
        images = {int(img["id"]): img for img in data.get("images", [])}
        anns_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for ann in data.get("annotations", []):
            anns_by_image[int(ann["image_id"])].append(ann)
        categories = {int(cat["id"]): cat["name"] for cat in data.get("categories", [])}
        return CocoIndex(images=images, anns_by_image=anns_by_image, categories=categories, source="json")


def _ann_split_hint(ann_json: Path) -> Optional[str]:
    stem = ann_json.stem.lower()
    if "train" in stem:
        return "train"
    if "val" in stem:
        return "val"
    if "test" in stem:
        return "test"
    return None


def infer_image_root(dataset_root: Path, ann_json: Path, images: Dict[int, Dict[str, Any]]) -> Path:
    if not images:
        raise ValueError("No images found in annotation JSON.")

    sample_file_names = [str(meta["file_name"]) for meta in list(images.values())[:100]]
    split_hint = _ann_split_hint(ann_json)

    dirs = [dataset_root] + sorted(p for p in dataset_root.rglob("*") if p.is_dir())
    best_dir: Optional[Path] = None
    best_score = (-1, -1, "")

    for directory in dirs:
        hit_count = 0
        for file_name in sample_file_names:
            if (directory / file_name).is_file():
                hit_count += 1
        split_bonus = 1 if split_hint and split_hint in directory.name.lower() else 0
        score = (hit_count, split_bonus, str(directory))
        if score > best_score:
            best_score = score
            best_dir = directory

    if best_dir is None or best_score[0] == 0:
        raise FileNotFoundError(
            f"Could not infer image root under {dataset_root} for annotation file {ann_json}"
        )

    validate_image_root(best_dir, images, min_success=5)
    return best_dir


def validate_image_root(img_root: Path, images: Dict[int, Dict[str, Any]], min_success: int = 5) -> None:
    success = 0
    failures: List[str] = []
    for meta in images.values():
        img_path = img_root / str(meta["file_name"])
        if not img_path.is_file():
            failures.append(str(img_path))
            continue
        try:
            with Image.open(img_path) as im:
                im.verify()
            success += 1
            if success >= min_success:
                return
        except Exception:
            failures.append(str(img_path))
    raise FileNotFoundError(
        f"Image root validation failed for {img_root}. Successful opens={success}, failures(sample)={failures[:5]}"
    )


def filter_annotations(
    anns: Sequence[Dict[str, Any]],
    min_area: float,
    category_ids: Optional[set[int]],
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for ann in anns:
        cat_id = int(ann.get("category_id", -1))
        if category_ids is not None and cat_id not in category_ids:
            continue
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        area = float(ann.get("area", bbox[2] * bbox[3]))
        if area < min_area:
            continue
        filtered.append(ann)
    return filtered


def choose_images(
    index: CocoIndex,
    num: int,
    seed: int,
    min_area: float,
    category_ids: Optional[set[int]],
    only_images_with_boxes: bool,
) -> Tuple[List[Tuple[int, Dict[str, Any], List[Dict[str, Any]]]], Counter]:
    eligible: List[Tuple[int, Dict[str, Any], List[Dict[str, Any]]]] = []
    category_counter: Counter = Counter()

    for image_id, meta in index.images.items():
        anns = filter_annotations(index.anns_by_image.get(image_id, []), min_area, category_ids)
        if only_images_with_boxes and not anns:
            continue
        eligible.append((image_id, meta, anns))
        for ann in anns:
            category_counter[int(ann["category_id"])] += 1

    rng = random.Random(seed)
    rng.shuffle(eligible)
    return eligible[:num], category_counter


def clip_bbox(bbox: Sequence[float], width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
    x, y, w, h = [float(v) for v in bbox]
    x0 = max(0.0, min(x, width))
    y0 = max(0.0, min(y, height))
    x1 = max(0.0, min(x + max(w, 0.0), width))
    y1 = max(0.0, min(y + max(h, 0.0), height))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def category_color(category_id: int) -> Tuple[int, int, int]:
    rng = random.Random(category_id)
    return (
        rng.randint(64, 255),
        rng.randint(64, 255),
        rng.randint(64, 255),
    )


def draw_boxes(
    image: Image.Image,
    anns: Sequence[Dict[str, Any]],
    categories: Dict[int, str],
    draw_labels: bool,
) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size

    for ann in anns:
        clipped = clip_bbox(ann["bbox"], width, height)
        if clipped is None:
            continue
        x0, y0, x1, y1 = clipped
        cat_id = int(ann.get("category_id", -1))
        color = category_color(cat_id)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        if draw_labels:
            name = categories.get(cat_id, str(cat_id))
            area = float(ann.get("area", (x1 - x0) * (y1 - y0)))
            label = f"{name} ({int(area)})"
            text_bbox = draw.textbbox((x0, y0), label, font=font)
            tx0, ty0, tx1, ty1 = text_bbox
            pad = 2
            bg = (tx0 - pad, ty0 - pad, tx1 + pad, ty1 + pad)
            draw.rectangle(bg, fill=color)
            draw.text((x0, y0), label, fill=(0, 0, 0), font=font)
    return image


def safe_output_name(image_id: int, file_name: str) -> str:
    stem = Path(file_name).stem.replace(" ", "_")
    return f"{image_id}_{stem}.jpg"


def write_gallery(out_dir: Path, saved_items: Sequence[Tuple[str, str]]) -> None:
    rows = []
    for file_name, caption in saved_items:
        rows.append(
            f"<div class='card'><img src='{html.escape(file_name)}' alt='{html.escape(caption)}'>"
            f"<div class='caption'>{html.escape(caption)}</div></div>"
        )
    rows_html = "\n".join(rows)
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>COCO Sample Visualization</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid #ddd; padding: 10px; border-radius: 8px; background: #fafafa; }}
    img {{ width: 100%; height: auto; display: block; }}
    .caption {{ margin-top: 8px; font-size: 14px; word-break: break-word; }}
  </style>
</head>
<body>
  <h1>COCO Sample Visualization</h1>
  <div class="grid">
    {rows_html}
  </div>
</body>
</html>"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")


def print_summary(
    dataset_root: Path,
    ann_json: Path,
    img_root: Path,
    total_images_scanned: int,
    images_saved: int,
    avg_boxes_per_image: float,
    top_categories: List[Tuple[str, int]],
) -> None:
    print("\nSummary")
    print(f"dataset_root: {dataset_root}")
    print(f"ann_json used: {ann_json}")
    print(f"img_root used: {img_root}")
    print(f"total images scanned: {total_images_scanned}")
    print(f"images saved: {images_saved}")
    print(f"avg boxes per image: {avg_boxes_per_image:.2f}")
    print(f"top categories: {top_categories}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize COCO-format samples with bounding boxes.")
    parser.add_argument("--ann_json", type=Path, default=None, help="Optional COCO annotation JSON path.")
    parser.add_argument("--img_root", type=Path, default=None, help="Optional image root path.")
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT, help="Dataset root.")
    parser.add_argument("--out_dir", type=Path, default=Path("./coco_vis_dspcbsd"), help="Output directory.")
    parser.add_argument("--num", type=int, default=20, help="Number of images to save.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--draw_labels", type=str2bool, default=True, help="Whether to draw labels.")
    parser.add_argument("--min_area", type=float, default=0.0, help="Minimum bbox area.")
    parser.add_argument("--category_ids", type=str, default=None, help="Optional CSV of category IDs.")
    parser.add_argument("--only_images_with_boxes", type=str2bool, default=True, help="Sample only images with boxes.")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    discovered_candidates: List[Path] = []
    if args.ann_json is None:
        discovered_candidates, ann_json = discover_annotation_json(dataset_root)
    else:
        ann_json = args.ann_json.resolve()
        discovered_candidates = [ann_json]

    print("COCO JSON candidates:")
    for candidate in discovered_candidates:
        print(f"  - {candidate}")
    print(f"Selected ann_json: {ann_json}")

    index = load_coco_index(ann_json)
    print(f"Annotation loader: {index.source}")
    print(f"Loaded images={len(index.images)} annotations={sum(len(v) for v in index.anns_by_image.values())} categories={len(index.categories)}")

    if args.img_root is None:
        img_root = infer_image_root(dataset_root, ann_json, index.images)
    else:
        img_root = args.img_root.resolve()
        validate_image_root(img_root, index.images, min_success=5)
    print(f"Selected img_root: {img_root}")

    category_ids = parse_category_ids(args.category_ids)
    chosen, category_counter = choose_images(
        index=index,
        num=args.num,
        seed=args.seed,
        min_area=args.min_area,
        category_ids=category_ids,
        only_images_with_boxes=args.only_images_with_boxes,
    )

    saved_items: List[Tuple[str, str]] = []
    saved_count = 0
    skipped_missing = 0
    skipped_empty = 0
    boxes_saved = 0

    for image_id, meta, anns in chosen:
        if not anns and args.only_images_with_boxes:
            skipped_empty += 1
            continue
        file_name = str(meta["file_name"])
        image_path = img_root / file_name
        if not image_path.is_file():
            print(f"[WARN] Missing image: {image_path}", file=sys.stderr)
            skipped_missing += 1
            continue
        try:
            with Image.open(image_path) as image:
                rendered = draw_boxes(image, anns, index.categories, args.draw_labels)
                out_name = safe_output_name(image_id, file_name)
                rendered.save(out_dir / out_name, quality=95)
        except Exception as exc:
            print(f"[WARN] Failed to process {image_path}: {exc}", file=sys.stderr)
            skipped_missing += 1
            continue

        saved_count += 1
        boxes_saved += len(anns)
        caption = f"id={image_id} file={file_name} boxes={len(anns)}"
        saved_items.append((out_name, caption))

    write_gallery(out_dir, saved_items)

    top_categories = [
        (index.categories.get(cat_id, str(cat_id)), count)
        for cat_id, count in category_counter.most_common(10)
    ]
    avg_boxes = (boxes_saved / saved_count) if saved_count else 0.0

    print_summary(
        dataset_root=dataset_root,
        ann_json=ann_json,
        img_root=img_root,
        total_images_scanned=len(index.images),
        images_saved=saved_count,
        avg_boxes_per_image=avg_boxes,
        top_categories=top_categories,
    )
    print(f"Skipped missing images: {skipped_missing}")
    print(f"Skipped empty images: {skipped_empty}")
    print(f"Gallery written to: {out_dir / 'index.html'}")

    print("\nRun examples")
    print(
        "python tools/vis_coco_samples.py \\\n"
        f"  --dataset_root {dataset_root} \\\n"
        "  --out_dir ./coco_vis_dspcbsd \\\n"
        "  --num 30 --seed 0 --only_images_with_boxes true"
    )
    print(
        "python tools/vis_coco_samples.py \\\n"
        f"  --ann_json {ann_json} \\\n"
        f"  --img_root {img_root} \\\n"
        "  --out_dir ./coco_vis_dspcbsd \\\n"
        "  --num 30 --seed 0"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
