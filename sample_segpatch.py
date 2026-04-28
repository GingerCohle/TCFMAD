from pathlib import Path
import argparse
import cv2
import numpy as np


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    保留最大连通域，去掉小噪点。
    输入 mask 为灰度图，前景 > 0
    输出为 0/255 的 uint8 mask
    """
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        return np.zeros_like(mask, dtype=np.uint8)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + np.argmax(areas)

    cleaned = np.zeros_like(binary, dtype=np.uint8)
    cleaned[labels == largest_idx] = 255
    return cleaned


def get_bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return x1, y1, x2, y2


def find_image_path(test_ko_dir: Path, stem: str):
    candidates = [
        test_ko_dir / f"{stem}.JPG",
        test_ko_dir / f"{stem}.jpg",
        test_ko_dir / f"{stem}.JPEG",
        test_ko_dir / f"{stem}.jpeg",
        test_ko_dir / f"{stem}.png",
        ]
    for p in candidates:
        if p.exists():
            return p
    return None


def extract_seg_patch_black_bg(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    生成真正的 seg patch：
    - 先取 mask 的最小外接矩形
    - 再在该矩形区域内，仅保留缺陷像素的原始 RGB
    - 非缺陷像素严格置为纯黑 (0,0,0)
    输出 patch 尺寸 = bbox 尺寸
    """
    bbox = get_bbox_from_mask(mask)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox

    img_crop = image[y1:y2 + 1, x1:x2 + 1].copy()
    mask_crop = mask[y1:y2 + 1, x1:x2 + 1]

    out = np.zeros_like(img_crop, dtype=np.uint8)
    fg = mask_crop > 0
    out[fg] = img_crop[fg]

    return out


def process_category(
        category_dir: Path,
        save_root: Path,
        shot: int,
        keep_largest: bool = True,
):
    gt_ko_dir = category_dir / "ground_truth" / "ko"
    test_ko_dir = category_dir / "test" / "ko"

    # 输出到：<save_root>/train/<category>/seg_patch/
    out_dir = save_root / "train" / category_dir.name / "seg_patch"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not gt_ko_dir.exists() or not test_ko_dir.exists():
        return {
            "category": category_dir.name,
            "total": 0,
            "success": 0,
            "skipped": 0,
            "reason": "missing ground_truth/ko or test/ko",
        }

    mask_paths = sorted(gt_ko_dir.glob("*.png"))
    selected_masks = mask_paths[:shot]

    total = 0
    success = 0
    skipped = 0

    print(f"\n[Processing] {category_dir.name}")
    print(f"  in mask dir : {gt_ko_dir}")
    print(f"  in image dir: {test_ko_dir}")
    print(f"  out dir     : {out_dir}")
    print(f"  shot        : {shot}")
    print(f"  selected    : {len(selected_masks)}")

    for mask_path in selected_masks:
        total += 1
        stem = mask_path.stem
        img_path = find_image_path(test_ko_dir, stem)
        out_path = out_dir / f"{stem}.png"

        if img_path is None:
            print(f"  [skip] image not found for: {stem}")
            skipped += 1
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        if mask is None:
            print(f"  [skip] failed to read mask: {mask_path.name}")
            skipped += 1
            continue
        if image is None:
            print(f"  [skip] failed to read image: {img_path.name}")
            skipped += 1
            continue

        if mask.shape[:2] != image.shape[:2]:
            print(
                f"  [skip] size mismatch: {stem}, "
                f"mask={mask.shape[:2]}, image={image.shape[:2]}"
            )
            skipped += 1
            continue

        if keep_largest:
            cleaned_mask = keep_largest_component(mask)
        else:
            cleaned_mask = np.where(mask > 0, 255, 0).astype(np.uint8)

        if cleaned_mask.max() == 0:
            print(f"  [skip] empty mask after cleaning: {mask_path.name}")
            skipped += 1
            continue

        patch = extract_seg_patch_black_bg(image, cleaned_mask)
        if patch is None:
            print(f"  [skip] failed to extract seg patch: {mask_path.name}")
            skipped += 1
            continue

        ok = cv2.imwrite(str(out_path), patch)
        if not ok:
            print(f"  [skip] failed to save: {out_path.name}")
            skipped += 1
            continue

        success += 1

    print(f"  done: success={success}, skipped={skipped}, total={total}")
    return {
        "category": category_dir.name,
        "total": total,
        "success": success,
        "skipped": skipped,
        "reason": "",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract clean seg patches from VISA test/ko using ground_truth/ko masks and save to <save_root>/train/<category>/seg_patch/"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/media/disk/kejunjie_only/dino_anomaly/TCFMAD/visa",
        help="source VISA dataset root",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        required=True,
        help="target root, final structure will be <save_root>/train/<category>/seg_patch/",
    )
    parser.add_argument(
        "--shot",
        type=int,
        default=1,
        help="number of samples to extract per category, e.g. 1, 2, 4",
    )
    parser.add_argument(
        "--no-keep-largest",
        action="store_true",
        help="disable largest connected component filtering",
    )
    args = parser.parse_args()

    if args.shot <= 0:
        raise ValueError("--shot must be a positive integer")

    data_root = Path(args.data_root)
    save_root = Path(args.save_root)

    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")

    save_root.mkdir(parents=True, exist_ok=True)

    category_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])

    total_all = 0
    success_all = 0
    skipped_all = 0

    print(f"Source data root: {data_root}")
    print(f"Target save root: {save_root}")
    print(f"Found {len(category_dirs)} category dirs")
    print(f"Shot: {args.shot}")

    for category_dir in category_dirs:
        result = process_category(
            category_dir=category_dir,
            save_root=save_root,
            shot=args.shot,
            keep_largest=not args.no_keep_largest,
        )
        total_all += result["total"]
        success_all += result["success"]
        skipped_all += result["skipped"]

    print("\n===== SUMMARY =====")
    print(f"total   : {total_all}")
    print(f"success : {success_all}")
    print(f"skipped : {skipped_all}")


if __name__ == "__main__":
    main()