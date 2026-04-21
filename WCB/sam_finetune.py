#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


DEFAULT_TRAIN_ROOT = Path("/media/disk/kejunjie_only/dino_anomaly/seg_sam/paired_dataset")
DEFAULT_OUTPUT_ROOT = Path("/media/disk/kejunjie_only/dino_anomaly/seg_sam/outputs/sam_ft")


@dataclass(frozen=True)
class SamplePair:
    stem: str
    image_path: Path
    mask_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight SAM fine-tuning on 180 paired defect masks.")
    parser.add_argument("--train_root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--sam_ckpt", type=Path, required=True)
    parser.add_argument("--sam_model_type", choices=("vit_h", "vit_l", "vit_b"), default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--freeze_image_encoder", type=str2bool, default=True)
    parser.add_argument("--freeze_prompt_encoder", type=str2bool, default=True)
    parser.add_argument("--train_mask_decoder", type=str2bool, default=True)
    parser.add_argument("--box_mode", choices=("tight", "full"), default="tight")
    parser.add_argument("--num_pos_points", type=int, default=1)
    parser.add_argument("--num_neg_points", type=int, default=4)
    parser.add_argument("--save_dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--vis_dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--vis_max", type=int, default=200)
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def list_pairs(train_root: Path) -> List[SamplePair]:
    img_dir = train_root / "img"
    mask_dir = train_root / "seg_mask"
    if not img_dir.is_dir():
        raise SystemExit(f"img dir not found: {img_dir}")
    if not mask_dir.is_dir():
        raise SystemExit(f"seg_mask dir not found: {mask_dir}")

    image_map: Dict[str, Path] = {}
    for path in sorted(img_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            image_map[path.stem] = path

    mask_map: Dict[str, Path] = {}
    for path in sorted(mask_dir.iterdir()):
        if path.is_file() and path.suffix.lower() == ".png" and path.stem != "report":
            mask_map[path.stem] = path

    missing_masks = sorted(set(image_map) - set(mask_map))
    missing_images = sorted(set(mask_map) - set(image_map))
    if missing_masks or missing_images:
        message = [
            "img/ and seg_mask/ stems do not match.",
            f"missing masks: {len(missing_masks)}",
            f"missing images: {len(missing_images)}",
        ]
        if missing_masks:
            message.append(f"sample missing masks: {missing_masks[:10]}")
        if missing_images:
            message.append(f"sample missing images: {missing_images[:10]}")
        raise SystemExit(" ".join(message))

    pairs: List[SamplePair] = []
    for stem in sorted(mask_map):
        pairs.append(SamplePair(stem=stem, image_path=image_map[stem], mask_path=mask_map[stem]))
    return pairs


def split_pairs(pairs: Sequence[SamplePair], seed: int) -> Tuple[List[SamplePair], List[SamplePair]]:
    items = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(items)
    val_count = max(1, int(round(len(items) * 0.2)))
    val_pairs = items[:val_count]
    train_pairs = items[val_count:]
    return train_pairs, val_pairs


def load_image_and_mask(pair: SamplePair) -> Tuple[np.ndarray, np.ndarray]:
    with Image.open(pair.image_path) as img:
        image = np.array(img.convert("RGB"))
    with Image.open(pair.mask_path) as m:
        mask = np.array(m.convert("L"))
    mask = mask > 0
    return image, mask


def tight_box_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("empty mask")
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def full_box_for_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    return np.array([0.0, 0.0, float(max(w - 1, 0)), float(max(h - 1, 0))], dtype=np.float32)


def sample_points(
    mask: np.ndarray,
    box_xyxy: np.ndarray,
    num_pos: int,
    num_neg: int,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)
    pos_points: List[Tuple[float, float]] = []
    if len(xs) == 0:
        raise ValueError("cannot sample positive points from empty mask")
    indices = list(range(len(xs)))
    rng.shuffle(indices)
    for idx in indices[: max(num_pos, 1)]:
        pos_points.append((float(xs[idx]), float(ys[idx])))
    while len(pos_points) < num_pos:
        pos_points.append(pos_points[-1])

    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy.tolist()]
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, mask.shape[1] - 1)
    y2 = min(y2, mask.shape[0] - 1)

    yy, xx = np.mgrid[y1 : y2 + 1, x1 : x2 + 1]
    inside_box = np.ones_like(yy, dtype=bool)
    inside_gt = mask[y1 : y2 + 1, x1 : x2 + 1]
    neg_candidate = inside_box & (~inside_gt)
    if neg_candidate.any():
        dist_left = xx - x1
        dist_right = x2 - xx
        dist_top = yy - y1
        dist_bottom = y2 - yy
        edge_dist = np.minimum.reduce([dist_left, dist_right, dist_top, dist_bottom])
        preferred = neg_candidate & (edge_dist <= max(2, int(min(x2 - x1 + 1, y2 - y1 + 1) * 0.15)))
        if preferred.any():
            neg_candidate = preferred
    neg_y, neg_x = np.nonzero(neg_candidate)
    if len(neg_x) == 0:
        neg_points = [(float(x1), float(y1))] * max(num_neg, 1)
    else:
        order = list(range(len(neg_x)))
        rng.shuffle(order)
        neg_points = [
            (float(neg_x[idx] + x1), float(neg_y[idx] + y1))
            for idx in order[: max(num_neg, 1)]
        ]
        while len(neg_points) < num_neg:
            neg_points.append(neg_points[-1])

    point_coords = np.array(pos_points + neg_points, dtype=np.float32)
    point_labels = np.array([1] * len(pos_points) + [0] * len(neg_points), dtype=np.int64)
    return point_coords, point_labels


def binary_iou_batch(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    pred_bool = pred_mask > 0.5
    gt_bool = gt_mask > 0.5
    intersection = (pred_bool & gt_bool).sum(dim=(-2, -1)).float()
    union = (pred_bool | gt_bool).sum(dim=(-2, -1)).float().clamp_min(1.0)
    return intersection / union


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    numer = 2.0 * (probs * target).sum(dim=(-2, -1))
    denom = probs.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    dice = 1.0 - (numer + 1.0) / (denom + 1.0)
    return dice.mean()


def freeze_modules(
    sam: torch.nn.Module,
    freeze_image_encoder: bool,
    freeze_prompt_encoder: bool,
    train_mask_decoder: bool,
) -> None:
    for module in [sam.image_encoder]:
        for param in module.parameters():
            param.requires_grad = not freeze_image_encoder
    for module in [sam.prompt_encoder]:
        for param in module.parameters():
            param.requires_grad = not freeze_prompt_encoder
    for param in sam.mask_decoder.parameters():
        param.requires_grad = train_mask_decoder


def render_panel(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    box_xyxy: np.ndarray,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    out_path: Path,
) -> None:
    base = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    gt_img = Image.fromarray((gt_mask.astype(np.uint8) * 120), mode="L")
    pred_img = Image.fromarray((pred_mask.astype(np.uint8) * 180), mode="L")
    overlay.paste((0, 255, 0, 90), mask=gt_img)
    overlay.paste((0, 0, 255, 90), mask=pred_img)

    x1, y1, x2, y2 = [float(v) for v in box_xyxy.tolist()]
    draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0, 255), width=3)
    for (x, y), label in zip(point_coords.tolist(), point_labels.tolist()):
        color = (0, 255, 0, 255) if label == 1 else (255, 0, 0, 255)
        r = 5
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

    merged = Image.alpha_composite(base, overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save(out_path)


def prepare_prompts(
    transform: ResizeLongestSide,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    box_xyxy: np.ndarray,
    original_size: Tuple[int, int],
    device: torch.device,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    coords = transform.apply_coords(point_coords, original_size)
    boxes = transform.apply_boxes(box_xyxy[None, :], original_size)
    coord_t = torch.as_tensor(coords, dtype=torch.float32, device=device)[None, :, :]
    label_t = torch.as_tensor(point_labels, dtype=torch.int64, device=device)[None, :]
    box_t = torch.as_tensor(boxes, dtype=torch.float32, device=device)
    return (coord_t, label_t), box_t


def forward_sample(
    sam: torch.nn.Module,
    transform: ResizeLongestSide,
    image: np.ndarray,
    gt_mask: np.ndarray,
    args: argparse.Namespace,
    rng: random.Random,
    device: torch.device,
) -> Dict[str, object]:
    if not gt_mask.any():
        raise ValueError("empty target mask")

    box_xyxy = tight_box_from_mask(gt_mask) if args.box_mode == "tight" else full_box_for_mask(gt_mask)
    point_coords, point_labels = sample_points(
        gt_mask,
        box_xyxy=box_xyxy,
        num_pos=max(args.num_pos_points, 1),
        num_neg=max(args.num_neg_points, 0),
        rng=rng,
    )

    resized = transform.apply_image(image)
    input_size = resized.shape[:2]
    image_t = torch.as_tensor(resized, dtype=torch.float32, device=device).permute(2, 0, 1).contiguous()[None]
    image_t = sam.preprocess(image_t)

    if args.freeze_image_encoder:
        with torch.no_grad():
            image_embeddings = sam.image_encoder(image_t)
    else:
        image_embeddings = sam.image_encoder(image_t)

    prompt_inputs, box_t = prepare_prompts(
        transform=transform,
        point_coords=point_coords,
        point_labels=point_labels,
        box_xyxy=box_xyxy,
        original_size=image.shape[:2],
        device=device,
    )

    if args.freeze_prompt_encoder:
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=prompt_inputs,
                boxes=box_t,
                masks=None,
            )
    else:
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=prompt_inputs,
            boxes=box_t,
            masks=None,
        )

    low_res_masks, iou_preds = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
    )
    full_logits = sam.postprocess_masks(low_res_masks, input_size, image.shape[:2])
    gt_t = torch.from_numpy(gt_mask.astype(np.float32)).to(device)[None, None]

    with torch.no_grad():
        pred_prob_all = torch.sigmoid(full_logits)
        ious = binary_iou_batch(pred_prob_all[0], gt_t[0])
        best_idx = int(torch.argmax(ious).item())

    chosen_logits = full_logits[:, best_idx : best_idx + 1]
    chosen_prob = torch.sigmoid(chosen_logits)
    bce = F.binary_cross_entropy_with_logits(chosen_logits, gt_t)
    dice = dice_loss_from_logits(chosen_logits, gt_t)
    loss = bce + dice

    pred_mask = chosen_prob[0, 0].detach().cpu().numpy() > 0.5
    iou_value = float(binary_iou_batch(chosen_prob[0], gt_t[0])[0].item())
    return {
        "loss": loss,
        "iou": iou_value,
        "box": box_xyxy,
        "points": point_coords,
        "labels": point_labels,
        "pred_mask": pred_mask,
        "gt_mask": gt_mask,
        "pred_score": float(iou_preds[0, best_idx].detach().cpu().item()),
    }


def save_checkpoint(path: Path, sam: torch.nn.Module, args: argparse.Namespace, epoch: int, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized_args = {}
    for key, value in vars(args).items():
        serialized_args[key] = str(value) if isinstance(value, Path) else value
    payload = {
        "model_state": sam.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "args": serialized_args,
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.train_root = args.train_root.resolve()
    args.sam_ckpt = args.sam_ckpt.resolve()
    args.output_root = args.output_root.resolve()
    if args.sam_model_type is None:
        args.sam_model_type = infer_sam_model_type(args.sam_ckpt)
    args.save_dir = args.save_dir.resolve() if args.save_dir is not None else args.output_root / "ckpts"
    args.vis_dir = args.vis_dir.resolve() if args.vis_dir is not None else args.output_root / "vis_train"

    if not args.sam_ckpt.is_file():
        raise SystemExit(f"--sam_ckpt not found: {args.sam_ckpt}")

    pairs = list_pairs(args.train_root)
    if not pairs:
        raise SystemExit("no paired img/seg_mask samples found")
    train_pairs, val_pairs = split_pairs(pairs, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[args.sam_model_type](checkpoint=str(args.sam_ckpt))
    sam.to(device)
    freeze_modules(
        sam,
        freeze_image_encoder=args.freeze_image_encoder,
        freeze_prompt_encoder=args.freeze_prompt_encoder,
        train_mask_decoder=args.train_mask_decoder,
    )

    trainable_params = [param for param in sam.parameters() if param.requires_grad]
    if not trainable_params:
        raise SystemExit("no trainable parameters enabled")
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    transform = ResizeLongestSide(args.img_size)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.vis_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    vis_saved = 0
    best_val_iou = -1.0

    print("Fine-tuning SAM")
    print(f"  train_root            : {args.train_root}")
    print(f"  sam_ckpt              : {args.sam_ckpt}")
    print(f"  sam_model_type        : {args.sam_model_type}")
    print(f"  device                : {device}")
    print(f"  total_pairs           : {len(pairs)}")
    print(f"  train_pairs           : {len(train_pairs)}")
    print(f"  val_pairs             : {len(val_pairs)}")
    print(f"  freeze_image_encoder  : {args.freeze_image_encoder}")
    print(f"  freeze_prompt_encoder : {args.freeze_prompt_encoder}")
    print(f"  train_mask_decoder    : {args.train_mask_decoder}")

    for epoch in range(1, args.epochs + 1):
        shuffled = list(train_pairs)
        rng.shuffle(shuffled)

        sam.train()
        if args.freeze_image_encoder:
            sam.image_encoder.eval()
        if args.freeze_prompt_encoder:
            sam.prompt_encoder.eval()

        train_losses: List[float] = []
        train_ious: List[float] = []
        for batch_start in range(0, len(shuffled), args.batch_size):
            batch_pairs = shuffled[batch_start : batch_start + args.batch_size]
            batch_losses: List[torch.Tensor] = []

            optimizer.zero_grad(set_to_none=True)
            for pair in batch_pairs:
                image, gt_mask = load_image_and_mask(pair)
                result = forward_sample(sam, transform, image, gt_mask, args, rng, device)
                batch_losses.append(result["loss"])
                train_losses.append(float(result["loss"].detach().cpu().item()))
                train_ious.append(float(result["iou"]))

                if vis_saved < args.vis_max:
                    render_panel(
                        image=image,
                        gt_mask=result["gt_mask"],
                        pred_mask=result["pred_mask"],
                        box_xyxy=result["box"],
                        point_coords=result["points"],
                        point_labels=result["labels"],
                        out_path=args.vis_dir / f"train_epoch{epoch:03d}_{pair.stem}.jpg",
                    )
                    vis_saved += 1

            if batch_losses:
                loss = torch.stack(batch_losses).mean()
                loss.backward()
                optimizer.step()

        sam.eval()
        val_losses: List[float] = []
        val_ious: List[float] = []
        with torch.no_grad():
            for pair in val_pairs:
                image, gt_mask = load_image_and_mask(pair)
                result = forward_sample(sam, transform, image, gt_mask, args, rng, device)
                val_losses.append(float(result["loss"].detach().cpu().item()))
                val_ious.append(float(result["iou"]))
                if vis_saved < args.vis_max:
                    render_panel(
                        image=image,
                        gt_mask=result["gt_mask"],
                        pred_mask=result["pred_mask"],
                        box_xyxy=result["box"],
                        point_coords=result["points"],
                        point_labels=result["labels"],
                        out_path=args.vis_dir / f"val_epoch{epoch:03d}_{pair.stem}.jpg",
                    )
                    vis_saved += 1

        train_loss = float(np.mean(train_losses)) if train_losses else math.nan
        train_iou = float(np.mean(train_ious)) if train_ious else math.nan
        val_loss = float(np.mean(val_losses)) if val_losses else math.nan
        val_iou = float(np.mean(val_ious)) if val_ious else math.nan
        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_iou={train_iou:.4f} "
            f"val_loss={val_loss:.4f} val_iou={val_iou:.4f}"
        )

        metrics = {
            "train_loss": train_loss,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_iou": val_iou,
        }
        save_checkpoint(args.save_dir / "last.pth", sam, args, epoch, metrics)
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            save_checkpoint(args.save_dir / "best.pth", sam, args, epoch, metrics)
            print(f"  new_best: val_iou={val_iou:.4f} -> {args.save_dir / 'best.pth'}")

    print("Done")
    print(f"  best_ckpt : {args.save_dir / 'best.pth'}")
    print(f"  last_ckpt : {args.save_dir / 'last.pth'}")
    print(f"  vis_dir   : {args.vis_dir}")


if __name__ == "__main__":
    main()
