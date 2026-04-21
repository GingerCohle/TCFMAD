#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from segment_anything import SamPredictor, sam_model_registry


DEFAULT_VISA_ROOT = Path("/media/disk/kejunjie_only/dino_anomaly/foundPCB/pcbmarket_visa")
DEFAULT_OUTPUT_ROOT = Path("/media/disk/kejunjie_only/dino_anomaly/seg_sam/outputs/sam_ft")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate refined KO masks from box masks using fine-tuned SAM.")
    parser.add_argument("--visa_root", "--dataset_root", dest="visa_root", type=Path, default=DEFAULT_VISA_ROOT)
    parser.add_argument("--ckpt", "--ft_ckpt", dest="ckpt", type=Path, required=True)
    parser.add_argument("--sam_ckpt", type=Path, required=True)
    parser.add_argument("--sam_model_type", choices=("vit_h", "vit_l", "vit_b"), default=None)
    parser.add_argument("--max_images_per_class", type=int, default=-1)
    parser.add_argument("--seg_suffix", default="_seg")
    parser.add_argument("--force", type=str2bool, default=False)
    parser.add_argument("--neg_points", choices=(4, 8), type=int, default=4)
    parser.add_argument("--neg_margin_px", type=int, default=8)
    parser.add_argument("--min_area", type=int, default=50)
    parser.add_argument("--keep_largest_cc", type=str2bool, default=True)
    parser.add_argument("--max_ratio", type=float, default=0.85)
    parser.add_argument("--tighten_iters", type=int, default=2)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--vis_dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--vis_max", type=int, default=300)
    parser.add_argument("--classes", default="", help="Optional comma-separated subset of classes.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def str2bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value}")


def infer_sam_model_type(sam_ckpt: Path) -> str:
    name = sam_ckpt.name.lower()
    if "vit_h" in name:
        return "vit_h"
    if "vit_l" in name:
        return "vit_l"
    if "vit_b" in name:
        return "vit_b"
    raise SystemExit(
        f"cannot infer --sam_model_type from checkpoint name: {sam_ckpt}. "
        "Please set --sam_model_type explicitly."
    )


def load_checkpoint_payload(path: Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_sam_model(model_type: str, sam_ckpt: Path, fine_tuned_ckpt: Path, device: torch.device) -> SamPredictor:
    sam = sam_model_registry[model_type](checkpoint=str(sam_ckpt))
    payload = load_checkpoint_payload(fine_tuned_ckpt)
    state_dict = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    sam.load_state_dict(state_dict, strict=False)
    sam.to(device)
    sam.eval()
    return SamPredictor(sam)


def iter_classes(visa_root: Path, selected: Sequence[str]) -> List[Path]:
    classes = [path for path in sorted(visa_root.iterdir()) if path.is_dir()]
    if not selected:
        return classes
    selected_set = set(selected)
    return [path for path in classes if path.name in selected_set]


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as m:
        return np.array(m.convert("L")) > 0


def tight_box_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("empty box mask")
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def sample_positive_point(box_mask: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(box_mask)
    if len(xs) == 0:
        raise ValueError("cannot sample positive point from empty box mask")
    cx = 0.5 * (box_xyxy[0] + box_xyxy[2])
    cy = 0.5 * (box_xyxy[1] + box_xyxy[3])
    if box_mask[int(round(cy)), int(round(cx))]:
        return np.array([[cx, cy]], dtype=np.float32)
    centroid = np.array([[xs.mean(), ys.mean()]], dtype=np.float32)
    return centroid


def sample_negative_points(
    box_mask: np.ndarray,
    box_xyxy: np.ndarray,
    neg_points: int,
    margin_px: int,
) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy.tolist()]
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, box_mask.shape[1] - 1)
    y2 = min(y2, box_mask.shape[0] - 1)
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    margin = max(1, min(margin_px, width // 2, height // 2))
    candidates = [
        (x1 + margin, y1 + margin),
        (x2 - margin, y1 + margin),
        (x1 + margin, y2 - margin),
        (x2 - margin, y2 - margin),
        (x1 + margin, int(round(0.5 * (y1 + y2)))),
        (x2 - margin, int(round(0.5 * (y1 + y2)))),
        (int(round(0.5 * (x1 + x2))), y1 + margin),
        (int(round(0.5 * (x1 + x2))), y2 - margin),
    ]
    chosen = candidates[:neg_points]
    if len(chosen) < neg_points:
        chosen += [chosen[-1]] * (neg_points - len(chosen))
    return np.array(chosen, dtype=np.float32)


def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = 1 + int(np.argmax(areas))
    return labels == best_label


def postprocess_mask(mask: np.ndarray, min_area: int, keep_largest_cc: bool) -> np.ndarray:
    mask = mask.astype(bool)
    if keep_largest_cc and mask.any():
        mask = keep_largest_connected_component(mask)
    if int(mask.sum()) < min_area:
        return np.zeros_like(mask, dtype=bool)
    return mask


def enforce_area_limit(mask: np.ndarray, prob_map: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0:
        return np.zeros_like(mask, dtype=bool)
    if int(mask.sum()) <= limit:
        return mask
    limited = topk_inside_box(prob_map, mask, max_pixels=limit)
    if int(limited.sum()) > limit:
        limited = topk_inside_box(prob_map, limited, max_pixels=limit)
    return limited


def topk_inside_box(prob_map: np.ndarray, box_mask: np.ndarray, max_pixels: int) -> np.ndarray:
    flat_idx = np.flatnonzero(box_mask)
    if len(flat_idx) == 0:
        return np.zeros_like(box_mask, dtype=bool)
    limit = min(len(flat_idx), max(max_pixels, 0))
    if limit <= 0:
        return np.zeros_like(box_mask, dtype=bool)
    values = prob_map.reshape(-1)[flat_idx]
    order = np.argpartition(values, -limit)[-limit:]
    chosen_flat = flat_idx[order]
    out = np.zeros_like(box_mask.reshape(-1), dtype=bool)
    out[chosen_flat] = True
    return out.reshape(box_mask.shape)


def build_points(pos_point: np.ndarray, neg_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    point_coords = np.concatenate([pos_point, neg_points], axis=0)
    point_labels = np.array([1] * len(pos_point) + [0] * len(neg_points), dtype=np.int64)
    return point_coords, point_labels


def choose_candidate(logits: np.ndarray, scores: np.ndarray, box_mask: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, float]:
    best_idx = int(np.argmax(scores))
    prob_map = 1.0 / (1.0 + np.exp(-logits[best_idx]))
    mask = (prob_map >= 0.5) & box_mask
    mask = postprocess_mask(mask, args.min_area, args.keep_largest_cc)
    area_ratio = float(mask.sum()) / max(float(box_mask.sum()), 1.0)
    return mask, prob_map, area_ratio


def finalize_mask(
    mask: np.ndarray,
    prob_map: np.ndarray,
    box_mask: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, float]:
    box_area = int(box_mask.sum())
    limit = int(math.floor(args.max_ratio * float(box_area)))

    mask = mask.astype(bool) & box_mask
    mask = postprocess_mask(mask, args.min_area, args.keep_largest_cc)
    mask = enforce_area_limit(mask, prob_map, limit)
    mask &= box_mask
    mask = postprocess_mask(mask, args.min_area, args.keep_largest_cc)
    mask = enforce_area_limit(mask, prob_map, limit)
    mask &= box_mask

    if limit <= 0:
        mask = np.zeros_like(box_mask, dtype=bool)

    if int(mask.sum()) > max(limit, 0):
        mask = topk_inside_box(prob_map, box_mask, max_pixels=max(limit, 0))
        mask = postprocess_mask(mask, args.min_area, args.keep_largest_cc)
        mask = enforce_area_limit(mask, prob_map, limit)
        mask &= box_mask

    area_ratio = float(mask.sum()) / max(float(box_area), 1.0)
    if area_ratio > args.max_ratio + 1e-8:
        raise RuntimeError(
            f"final mask violates area constraint: ratio={area_ratio:.6f} max_ratio={args.max_ratio:.6f}"
        )
    return mask, area_ratio


def render_panel(
    image: np.ndarray,
    box_mask: np.ndarray,
    seg_mask: np.ndarray,
    box_xyxy: np.ndarray,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    out_path: Path,
) -> None:
    base = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    overlay.paste((255, 255, 0, 80), mask=Image.fromarray((box_mask.astype(np.uint8) * 80), mode="L"))
    overlay.paste((0, 255, 0, 100), mask=Image.fromarray((seg_mask.astype(np.uint8) * 120), mode="L"))
    x1, y1, x2, y2 = [float(v) for v in box_xyxy.tolist()]
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=3)
    for (x, y), label in zip(point_coords.tolist(), point_labels.tolist()):
        color = (0, 255, 0, 255) if label == 1 else (255, 0, 0, 255)
        r = 5
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    merged = Image.alpha_composite(base, overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save(out_path)


def run_prediction(
    predictor: SamPredictor,
    image: np.ndarray,
    box_mask: np.ndarray,
    box_xyxy: np.ndarray,
    neg_points: int,
    margin_px: int,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    predictor.set_image(image)
    pos_point = sample_positive_point(box_mask, box_xyxy)
    neg = sample_negative_points(box_mask, box_xyxy, neg_points=neg_points, margin_px=margin_px)
    point_coords, point_labels = build_points(pos_point, neg)
    mask_logits, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box_xyxy.astype(np.float32),
        multimask_output=True,
        return_logits=True,
    )
    mask, prob_map, area_ratio = choose_candidate(mask_logits, scores, box_mask, args)
    return mask, prob_map, point_coords, point_labels, area_ratio


def main() -> None:
    args = parse_args()
    args.visa_root = args.visa_root.resolve()
    args.ckpt = args.ckpt.resolve()
    args.sam_ckpt = args.sam_ckpt.resolve()
    args.output_root = args.output_root.resolve()
    if args.sam_model_type is None:
        args.sam_model_type = infer_sam_model_type(args.sam_ckpt)
    args.vis_dir = args.vis_dir.resolve() if args.vis_dir is not None else args.output_root / "vis_gen"
    args.vis_dir.mkdir(parents=True, exist_ok=True)

    if not args.visa_root.is_dir():
        raise SystemExit(f"--visa_root not found: {args.visa_root}")
    if not args.ckpt.is_file():
        raise SystemExit(f"--ckpt not found: {args.ckpt}")
    if not args.sam_ckpt.is_file():
        raise SystemExit(f"--sam_ckpt not found: {args.sam_ckpt}")

    selected_classes = [item.strip() for item in args.classes.split(",") if item.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = load_sam_model(args.sam_model_type, args.sam_ckpt, args.ckpt, device)

    total_written = 0
    total_skipped = 0
    total_area_ratios: List[float] = []
    vis_saved = 0

    print("Generate KO seg masks")
    print(f"  visa_root  : {args.visa_root}")
    print(f"  ckpt       : {args.ckpt}")
    print(f"  sam_ckpt   : {args.sam_ckpt}")
    print(f"  device     : {device}")
    print(f"  max_ratio  : {args.max_ratio}")

    for class_dir in iter_classes(args.visa_root, selected_classes):
        ko_dir = class_dir / "test" / "ko"
        box_dir = class_dir / "ground_truth" / "ko"
        if not ko_dir.is_dir() or not box_dir.is_dir():
            continue
        image_paths = sorted(
            path
            for path in ko_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if args.max_images_per_class > 0:
            image_paths = image_paths[: args.max_images_per_class]

        class_written = 0
        class_skipped = 0
        for image_path in image_paths:
            stem = image_path.stem
            box_path = box_dir / f"{stem}.png"
            out_path = box_dir / f"{stem}{args.seg_suffix}.png"
            if out_path.exists() and not args.force:
                class_skipped += 1
                total_skipped += 1
                print(f"[SKIP] {class_dir.name}/{stem}: exists")
                continue
            if not box_path.is_file():
                class_skipped += 1
                total_skipped += 1
                print(f"[SKIP] {class_dir.name}/{stem}: missing box mask")
                continue

            image = load_image(image_path)
            box_mask = load_mask(box_path)
            if not box_mask.any():
                class_skipped += 1
                total_skipped += 1
                print(f"[SKIP] {class_dir.name}/{stem}: empty box mask")
                continue

            box_xyxy = tight_box_from_mask(box_mask)
            mask, prob_map, point_coords, point_labels, area_ratio = run_prediction(
                predictor=predictor,
                image=image,
                box_mask=box_mask,
                box_xyxy=box_xyxy,
                neg_points=args.neg_points,
                margin_px=args.neg_margin_px,
                args=args,
            )

            for retry in range(args.tighten_iters):
                if area_ratio <= args.max_ratio:
                    break
                mask, prob_map, point_coords, point_labels, area_ratio = run_prediction(
                    predictor=predictor,
                    image=image,
                    box_mask=box_mask,
                    box_xyxy=box_xyxy,
                    neg_points=8,
                    margin_px=args.neg_margin_px + (retry + 1) * 4,
                    args=args,
                )

            if area_ratio > args.max_ratio:
                limit = int(math.floor(args.max_ratio * float(box_mask.sum())))
                mask = topk_inside_box(prob_map, box_mask, max_pixels=limit)
                mask, area_ratio = finalize_mask(mask, prob_map, box_mask, args)
            else:
                mask, area_ratio = finalize_mask(mask, prob_map, box_mask, args)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(out_path)
            class_written += 1
            total_written += 1
            total_area_ratios.append(area_ratio)
            print(
                f"[OK] {class_dir.name}/{stem}: "
                f"area_ratio={area_ratio:.4f} out={out_path.name}"
            )

            if vis_saved < args.vis_max:
                render_panel(
                    image=image,
                    box_mask=box_mask,
                    seg_mask=mask,
                    box_xyxy=box_xyxy,
                    point_coords=point_coords,
                    point_labels=point_labels,
                    out_path=args.vis_dir / f"{class_dir.name}_{stem}.jpg",
                )
                vis_saved += 1

        print(
            f"[class {class_dir.name}] written={class_written} skipped={class_skipped}"
        )

    mean_ratio = float(np.mean(total_area_ratios)) if total_area_ratios else math.nan
    print("Done")
    print(f"  total_written    : {total_written}")
    print(f"  total_skipped    : {total_skipped}")
    print(f"  mean_area_ratio  : {mean_ratio:.4f}")
    print(f"  vis_dir          : {args.vis_dir}")


if __name__ == "__main__":
    main()
