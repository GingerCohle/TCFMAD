#!/usr/bin/env python3
"""Audit PCBMarket dataset layout, annotation format, and sample visualizations."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
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
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("outputs/pcbmarket_vis"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_vis_per_split", type=int, default=5)
    return parser.parse_args()


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def compute_disk_usage(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                continue
    return total


def build_tree(root: Path, max_depth: int = 3) -> str:
    lines = [str(root)]

    def walk(path: Path, depth: int) -> None:
        if depth >= max_depth:
            return
        try:
            children = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        except OSError:
            return
        for child in children:
            rel_depth = len(child.relative_to(root).parts)
            indent = "    " * (rel_depth - 1)
            suffix = "/" if child.is_dir() else ""
            lines.append(f"{indent}{child.name}{suffix}")
            if child.is_dir():
                walk(child, depth + 1)

    walk(root, 0)
    return "\n".join(lines)


def scan_candidates(root: Path) -> dict[str, list[Path]]:
    coco_jsons = sorted(
        {
            *root.glob("annotations/*.json"),
            *root.glob("instances_*.json"),
            *root.glob("**/instances_*.json"),
        }
    )
    voc_xmls = sorted(root.glob("**/Annotations/*.xml"))
    voc_splits = sorted(root.glob("**/ImageSets/Main/*.txt"))
    masks = sorted(
        {
            *root.glob("**/*_mask.png"),
            *root.glob("**/masks/*"),
            *root.glob("**/ground_truth/*"),
        }
    )
    custom = sorted(
        {
            *root.glob("**/*.pkl"),
            *root.glob("**/*.npz"),
            *root.glob("**/*.csv"),
        }
    )
    image_dirs = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        try:
            has_image = any(child.suffix.lower() in IMAGE_EXTS for child in path.iterdir())
        except OSError:
            has_image = False
        if has_image:
            image_dirs.append(path)
    image_dirs = sorted(set(image_dirs))
    return {
        "coco_jsons": coco_jsons,
        "voc_xmls": voc_xmls,
        "voc_splits": voc_splits,
        "masks": masks,
        "custom": custom,
        "image_dirs": image_dirs,
    }


def infer_image_root(root: Path, data: dict, candidate_dirs: list[Path]) -> Path:
    sample_files = [img.get("file_name", "") for img in data.get("images", [])[:20] if img.get("file_name")]
    if not sample_files:
        raise RuntimeError("COCO json has no image file_name entries.")
    scored = []
    for candidate in candidate_dirs:
        hits = 0
        for file_name in sample_files:
            if (candidate / file_name).exists():
                hits += 1
        scored.append((hits, str(candidate), candidate))
    scored.sort(reverse=True)
    if not scored or scored[0][0] == 0:
        raise RuntimeError("Could not infer image root from COCO file_name entries.")
    return scored[0][2]


def detect_format(root: Path, candidates: dict[str, list[Path]]) -> dict:
    if candidates["coco_jsons"]:
        coco_summaries = []
        for json_path in candidates["coco_jsons"]:
            with json_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if {"images", "annotations", "categories"}.issubset(data.keys()):
                coco_summaries.append((json_path, data))
        if coco_summaries:
            split_map = {}
            for json_path, data in coco_summaries:
                split_name = "unknown"
                stem = json_path.stem.lower()
                if "train" in stem:
                    split_name = "train"
                elif "val" in stem:
                    split_name = "val"
                elif "test" in stem:
                    split_name = "test"
                img_root = infer_image_root(root, data, candidates["image_dirs"])
                split_map[split_name] = {
                    "json_path": json_path,
                    "img_root": img_root,
                    "data": data,
                }
            conflicts = []
            if candidates["voc_xmls"]:
                conflicts.append("VOC XML files also detected")
            if candidates["masks"]:
                conflicts.append("mask files also detected")
            return {
                "format": "COCO detection",
                "priority_reason": "instances-style JSON with images/annotations/categories keys",
                "splits": split_map,
                "conflicts": conflicts,
            }
    if candidates["voc_xmls"]:
        return {
            "format": "VOC detection",
            "priority_reason": "VOC XML files detected and no COCO instances json selected",
            "splits": {},
            "conflicts": [],
        }
    if candidates["masks"]:
        return {
            "format": "Segmentation masks/custom pairing",
            "priority_reason": "mask files detected and no COCO/VOC selected",
            "splits": {},
            "conflicts": [],
        }
    return {
        "format": "Unknown/custom",
        "priority_reason": "No supported annotation format detected",
        "splits": {},
        "conflicts": [],
    }


def clip_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int] | None:
    x, y, w, h = box
    x1 = max(0, min(width - 1, int(round(x))))
    y1 = max(0, min(height - 1, int(round(y))))
    x2 = max(0, min(width - 1, int(round(x + w))))
    y2 = max(0, min(height - 1, int(round(y + h))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def label_colors() -> list[tuple[int, int, int]]:
    return [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
    ]


def draw_coco_visualization(
    image_path: Path,
    image_meta: dict,
    annotations: list[dict],
    category_by_id: dict[int, str],
    out_path: Path,
) -> bool:
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return False
    draw = ImageDraw.Draw(image)
    colors = label_colors()
    for ann in annotations:
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        clipped = clip_box(bbox, image.width, image.height)
        if clipped is None:
            continue
        color = colors[(ann.get("category_id", 0) - 1) % len(colors)]
        draw.rectangle(clipped, outline=color, width=4)
        label = category_by_id.get(ann.get("category_id"), str(ann.get("category_id")))
        tx = clipped[0] + 4
        ty = max(0, clipped[1] - 16)
        draw.rectangle((tx - 2, ty - 2, tx + 8 * len(label), ty + 12), fill=color)
        draw.text((tx, ty), label, fill=(255, 255, 255))
    image.save(out_path)
    return True


def analyze_coco_split(
    split_name: str,
    split_info: dict,
    out_dir: Path,
    seed: int,
    max_vis: int,
) -> dict:
    data = split_info["data"]
    img_root = split_info["img_root"]
    json_path = split_info["json_path"]
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    category_by_id = {cat["id"]: cat["name"] for cat in categories}
    class_names = [cat["name"] for cat in categories]
    image_sizes = [(img["width"], img["height"]) for img in images]
    width_values = [size[0] for size in image_sizes]
    height_values = [size[1] for size in image_sizes]
    size_counter = Counter(image_sizes)
    anns_by_image = defaultdict(list)
    class_counter = Counter()
    bbox_by_category = defaultdict(list)
    for ann in annotations:
        image_id = ann.get("image_id")
        anns_by_image[image_id].append(ann)
        class_counter[ann.get("category_id")] += 1
        bbox = ann.get("bbox")
        if bbox and len(bbox) == 4:
            _, _, w, h = bbox
            bbox_by_category[ann.get("category_id")].append((float(w), float(h), float(w) * float(h)))

    existing_opened = 0
    checked_samples = 0
    vis_paths = []
    rng = random.Random(seed)
    sampled_images = images[:]
    rng.shuffle(sampled_images)
    for image_meta in sampled_images[: min(max_vis, len(sampled_images))]:
        image_path = img_root / image_meta.get("file_name", "")
        try:
            with Image.open(image_path) as img:
                img.verify()
            checked_samples += 1
            existing_opened += 1
        except Exception:
            checked_samples += 1
            continue
        out_path = out_dir / f"{split_name}_{image_meta['id']}_{Path(image_meta['file_name']).name}"
        if draw_coco_visualization(
            image_path=image_path,
            image_meta=image_meta,
            annotations=anns_by_image.get(image_meta["id"], []),
            category_by_id=category_by_id,
            out_path=out_path,
        ):
            vis_paths.append(str(out_path.resolve()))

    top_classes = [
        {"name": category_by_id[cat_id], "count": count}
        for cat_id, count in class_counter.most_common()
    ]
    image_size_stats = {
        "unique_sizes": len(size_counter),
        "top_sizes": [
            {"width": width, "height": height, "count": count}
            for (width, height), count in size_counter.most_common(10)
        ],
        "width": {
            "min": min(width_values) if width_values else 0,
            "mean": statistics.mean(width_values) if width_values else 0.0,
            "max": max(width_values) if width_values else 0,
        },
        "height": {
            "min": min(height_values) if height_values else 0,
            "mean": statistics.mean(height_values) if height_values else 0.0,
            "max": max(height_values) if height_values else 0,
        },
    }
    bbox_stats = []
    for category in categories:
        cat_id = category["id"]
        values = bbox_by_category.get(cat_id, [])
        if not values:
            continue
        widths = [item[0] for item in values]
        heights = [item[1] for item in values]
        areas = [item[2] for item in values]
        bbox_stats.append(
            {
                "name": category["name"],
                "count": len(values),
                "width": {
                    "min": min(widths),
                    "mean": statistics.mean(widths),
                    "max": max(widths),
                },
                "height": {
                    "min": min(heights),
                    "mean": statistics.mean(heights),
                    "max": max(heights),
                },
                "area": {
                    "min": min(areas),
                    "median": statistics.median(areas),
                    "mean": statistics.mean(areas),
                    "max": max(areas),
                },
            }
        )
    annotated_images = sum(1 for image in images if anns_by_image.get(image["id"]))
    return {
        "split": split_name,
        "ann_json": str(json_path.resolve()),
        "img_root": str(img_root.resolve()),
        "num_images": len(images),
        "num_annotations": len(annotations),
        "num_annotated_images": annotated_images,
        "num_classes": len(categories),
        "class_names": class_names,
        "avg_boxes_per_image": (len(annotations) / len(images)) if images else 0.0,
        "image_size_stats": image_size_stats,
        "bbox_stats": bbox_stats,
        "top_classes": top_classes,
        "opened_samples": existing_opened,
        "checked_samples": checked_samples,
        "visualizations": vis_paths,
    }


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"dataset_root: {dataset_root}")
    disk_usage_bytes = compute_disk_usage(dataset_root)
    print(f"disk_usage: {human_size(disk_usage_bytes)} ({disk_usage_bytes} bytes)")
    print("tree_L3:")
    print(build_tree(dataset_root, max_depth=3))

    candidates = scan_candidates(dataset_root)
    print("candidate_files:")
    for key in ["coco_jsons", "voc_xmls", "voc_splits", "masks", "custom", "image_dirs"]:
        print(f"  {key}:")
        values = candidates[key]
        if values:
            for value in values:
                print(f"    - {value.resolve()}")
        else:
            print("    - none")

    detected = detect_format(dataset_root, candidates)
    print(f"detected_format: {detected['format']}")
    print(f"priority_reason: {detected['priority_reason']}")
    if detected["conflicts"]:
        print("conflicts:")
        for item in detected["conflicts"]:
            print(f"  - {item}")
    else:
        print("conflicts: none")

    summary = {
        "dataset_root": str(dataset_root),
        "disk_usage_bytes": disk_usage_bytes,
        "disk_usage_human": human_size(disk_usage_bytes),
        "tree_L3": build_tree(dataset_root, max_depth=3),
        "detected_format": detected["format"],
        "priority_reason": detected["priority_reason"],
        "conflicts": detected["conflicts"],
        "splits": {},
    }

    if detected["format"] == "COCO detection":
        for split_name, split_info in sorted(detected["splits"].items()):
            result = analyze_coco_split(
                split_name=split_name,
                split_info=split_info,
                out_dir=out_dir,
                seed=args.seed + sum(ord(c) for c in split_name),
                max_vis=args.max_vis_per_split,
            )
            summary["splits"][split_name] = result
            print(f"[{split_name}]")
            print(f"  ann_json: {result['ann_json']}")
            print(f"  img_root: {result['img_root']}")
            print(f"  num_images: {result['num_images']}")
            print(f"  num_annotations: {result['num_annotations']}")
            print(f"  num_annotated_images: {result['num_annotated_images']}")
            print(f"  num_classes: {result['num_classes']}")
            print(f"  class_names: {', '.join(result['class_names'])}")
            print(f"  avg_boxes_per_image: {result['avg_boxes_per_image']:.4f}")
            print(
                "  image_size_width_min_mean_max: "
                f"{result['image_size_stats']['width']['min']}/"
                f"{result['image_size_stats']['width']['mean']:.2f}/"
                f"{result['image_size_stats']['width']['max']}"
            )
            print(
                "  image_size_height_min_mean_max: "
                f"{result['image_size_stats']['height']['min']}/"
                f"{result['image_size_stats']['height']['mean']:.2f}/"
                f"{result['image_size_stats']['height']['max']}"
            )
            print(f"  unique_image_sizes: {result['image_size_stats']['unique_sizes']}")
            print("  top_image_sizes:")
            for item in result["image_size_stats"]["top_sizes"]:
                print(f"    - {item['width']}x{item['height']}: {item['count']}")
            print(
                f"  opened_samples: {result['opened_samples']}/{result['checked_samples']}"
            )
            print("  top_classes:")
            for item in result["top_classes"][:10]:
                print(f"    - {item['name']}: {item['count']}")
            print("  bbox_stats:")
            for item in result["bbox_stats"]:
                print(
                    "    - "
                    f"{item['name']}: count={item['count']}, "
                    f"w[min/mean/max]={item['width']['min']:.1f}/{item['width']['mean']:.2f}/{item['width']['max']:.1f}, "
                    f"h[min/mean/max]={item['height']['min']:.1f}/{item['height']['mean']:.2f}/{item['height']['max']:.1f}, "
                    f"area[min/median/mean/max]={item['area']['min']:.1f}/{item['area']['median']:.2f}/{item['area']['mean']:.2f}/{item['area']['max']:.1f}"
                )
            print("  visualizations:")
            if result["visualizations"]:
                for vis in result["visualizations"]:
                    print(f"    - {vis}")
            else:
                print("    - none")

    summary_path = out_dir / "audit_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"summary_json: {summary_path}")
    print(
        "reproduce: "
        f"python tools/pcbmarket_audit.py --dataset_root {dataset_root} --out_dir {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
