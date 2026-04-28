#!/usr/bin/env python3
"""Find COCO images that have no annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report COCO images that appear in images[] but not annotations[]."
    )
    parser.add_argument("--train_json", required=True, type=Path)
    parser.add_argument("--val_json", required=True, type=Path)
    parser.add_argument("--train_img_root", required=True, type=Path)
    parser.add_argument("--val_img_root", required=True, type=Path)
    parser.add_argument(
        "--out_dir",
        default=Path("./coco_unlabeled_report"),
        type=Path,
        help="Directory for split text files and summary.json",
    )
    parser.add_argument(
        "--max_list",
        default=50,
        type=int,
        help="Maximum number of example unlabeled file paths to print and save",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def analyze_split(split_name: str, ann_path: Path, img_root: Path, max_list: int) -> dict:
    data = load_json(ann_path)
    images = data.get("images", [])
    annotations = data.get("annotations", [])

    image_by_id = {}
    image_ids = set()
    for image in images:
        image_id = image.get("id")
        if image_id is None:
            continue
        image_by_id[image_id] = image
        image_ids.add(image_id)

    annotated_ids = set()
    for ann in annotations:
        image_id = ann.get("image_id")
        if image_id in image_by_id:
            annotated_ids.add(image_id)

    unlabeled_ids = [
        image["id"]
        for image in images
        if image.get("id") in image_ids and image.get("id") not in annotated_ids
    ]

    unlabeled_paths = []
    existing_count = 0
    missing_count = 0
    for image_id in unlabeled_ids:
        file_name = image_by_id[image_id].get("file_name", "")
        file_path = (img_root / file_name).resolve()
        unlabeled_paths.append(str(file_path))
        if file_path.exists():
            existing_count += 1
        else:
            missing_count += 1

    existence_rate = 1.0
    if unlabeled_ids:
        existence_rate = existing_count / len(unlabeled_ids)

    example_paths = unlabeled_paths[:max_list]
    example_names = [Path(path).name for path in example_paths]

    return {
        "split": split_name,
        "ann_json": str(ann_path.resolve()),
        "img_root": str(img_root.resolve()),
        "num_images": len(image_ids),
        "num_annotated_images": len(annotated_ids),
        "num_unlabeled_images": len(unlabeled_ids),
        "num_existing_unlabeled_files": existing_count,
        "num_missing_unlabeled_files": missing_count,
        "existence_pass_rate": existence_rate,
        "example_file_names": example_names,
        "example_file_paths": example_paths,
    }


def write_list(path: Path, rows: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row}\n")


def print_split_summary(result: dict, max_list: int) -> None:
    print(f"[{result['split']}]")
    print(f"  ann_json: {result['ann_json']}")
    print(f"  img_root: {result['img_root']}")
    print(f"  num_images: {result['num_images']}")
    print(f"  num_annotated_images: {result['num_annotated_images']}")
    print(f"  num_unlabeled_images: {result['num_unlabeled_images']}")
    print(
        "  file_existence_pass_rate: "
        f"{result['existence_pass_rate']:.4f} "
        f"({result['num_existing_unlabeled_files']}/{result['num_unlabeled_images']})"
    )
    print(f"  example_file_names (up to {max_list}):")
    if result["example_file_names"]:
        for name in result["example_file_names"]:
            print(f"    - {name}")
    else:
        print("    - none")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_result = analyze_split(
        split_name="train",
        ann_path=args.train_json,
        img_root=args.train_img_root,
        max_list=args.max_list,
    )
    val_result = analyze_split(
        split_name="val",
        ann_path=args.val_json,
        img_root=args.val_img_root,
        max_list=args.max_list,
    )

    write_list(args.out_dir / "train_unlabeled.txt", train_result["example_file_paths"])
    write_list(args.out_dir / "val_unlabeled.txt", val_result["example_file_paths"])

    summary = {"train": train_result, "val": val_result}
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"out_dir: {args.out_dir.resolve()}")
    print_split_summary(train_result, args.max_list)
    print_split_summary(val_result, args.max_list)
    print(f"summary_json: {(args.out_dir / 'summary.json').resolve()}")
    print(f"train_list: {(args.out_dir / 'train_unlabeled.txt').resolve()}")
    print(f"val_list: {(args.out_dir / 'val_unlabeled.txt').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
