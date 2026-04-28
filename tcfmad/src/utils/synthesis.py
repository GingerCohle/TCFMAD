import math
import random
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from skimage import morphology


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def generate_target_foreground_mask(img: np.ndarray, subclass: str) -> np.ndarray:
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )

    img_tensor = inv_normalize(img)
    img_tensor = torch.clamp(img_tensor, 0, 1)
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np_uint8 = (img_np * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np_uint8, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if subclass in ["carpet", "leather", "tile", "wood", "cable", "transistor"]:
        target_foreground_mask = np.ones_like(img_gray)
    elif subclass == "pill":
        _, target_foreground_mask = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        target_foreground_mask = (target_foreground_mask > 0).astype(int)
    elif subclass in ["hazelnut", "metal_nut", "toothbrush"]:
        _, target_foreground_mask = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
        )
        target_foreground_mask = (target_foreground_mask > 0).astype(int)
    elif subclass in ["bottle", "capsule", "grid", "screw", "zipper"]:
        _, target_background_mask = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        target_background_mask = (target_background_mask > 0).astype(int)
        target_foreground_mask = 1 - target_background_mask
    elif subclass in ["capsules"]:
        target_foreground_mask = np.ones_like(img_gray)
    elif subclass in ["pcb1", "pcb2", "pcb3", "pcb4"]:
        _, target_foreground_mask = cv2.threshold(
            img_np_uint8[:, :, 2], 100, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE
        )
        target_foreground_mask = target_foreground_mask.astype(bool).astype(int)
        target_foreground_mask = morphology.closing(target_foreground_mask, morphology.square(8))
        target_foreground_mask = morphology.opening(target_foreground_mask, morphology.square(3))
    elif subclass in ["candle", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pipe_fryum"]:
        _, target_foreground_mask = cv2.threshold(
            img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        target_foreground_mask = target_foreground_mask.astype(bool).astype(int)
        target_foreground_mask = morphology.closing(target_foreground_mask, morphology.square(3))
        target_foreground_mask = morphology.opening(target_foreground_mask, morphology.square(3))
    elif subclass in ["bracket_black", "bracket_brown", "connector"]:
        img_seg = img_np_uint8[:, :, 1]
        _, target_background_mask = cv2.threshold(
            img_seg, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        target_background_mask = target_background_mask.astype(bool).astype(int)
        target_foreground_mask = 1 - target_background_mask
    elif subclass in ["bracket_white", "tubes"]:
        img_seg = img_np_uint8[:, :, 2]
        _, target_background_mask = cv2.threshold(
            img_seg, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        target_background_mask = target_background_mask.astype(bool).astype(int)
        target_foreground_mask = target_background_mask
    elif subclass in ["metal_plate"]:
        img_seg = cv2.cvtColor(img_np_uint8, cv2.COLOR_RGB2GRAY)
        _, target_background_mask = cv2.threshold(
            img_seg, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        target_background_mask = target_background_mask.astype(bool).astype(int)
        target_foreground_mask = 1 - target_background_mask
    else:
        raise NotImplementedError("Unsupported foreground segmentation category")

    target_foreground_mask = morphology.closing(target_foreground_mask, morphology.square(6))
    target_foreground_mask = morphology.opening(target_foreground_mask, morphology.square(6))
    return target_foreground_mask


class CutPaste(object):
    def __init__(self, colorJitter=0.1):
        if colorJitter is None:
            self.colorJitter = None
        else:
            self.colorJitter = transforms.ColorJitter(
                brightness=colorJitter,
                contrast=colorJitter,
                saturation=colorJitter,
                hue=colorJitter,
            )

    def __call__(self, imgs):
        return imgs, imgs


class CutPasteNormal(CutPaste):
    def __init__(self, area_ratio=[0.02, 0.25], aspect_ratio=0.3, **kwargs):
        super().__init__(**kwargs)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, imgs, subclass):
        batch_size, _, h, w = imgs.shape
        augmented_imgs = imgs.clone()

        for i in range(batch_size):
            img = imgs[i]
            augmented = self.process_image(img, subclass)
            augmented_imgs[i] = augmented

        return imgs, augmented_imgs

    def process_image(self, img, subclass):
        img = img.clone()
        _, h, w = img.shape

        target_foreground_mask = generate_target_foreground_mask(img, subclass)

        area = h * w
        target_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * area
        aspect_ratio = random.uniform(self.aspect_ratio, 1 / self.aspect_ratio)

        cut_w = int(round(math.sqrt(target_area * aspect_ratio)))
        cut_h = int(round(math.sqrt(target_area / aspect_ratio)))

        if cut_w <= 0 or cut_h <= 0:
            return img

        from_x = random.randint(0, w - cut_w)
        from_y = random.randint(0, h - cut_h)

        patch = img[:, from_y : from_y + cut_h, from_x : from_x + cut_w]

        if self.colorJitter is not None:
            patch = self.colorJitter(patch)

        mask_indices = np.argwhere(target_foreground_mask == 1)
        if len(mask_indices) == 0:
            return img

        valid_indices = []
        for y, x in mask_indices:
            if y + cut_h <= h and x + cut_w <= w:
                valid_indices.append((y, x))

        if len(valid_indices) == 0:
            return img

        to_y, to_x = random.choice(valid_indices)

        augmented = img.clone()
        augmented[:, to_y : to_y + cut_h, to_x : to_x + cut_w] = patch

        return augmented


class CutPasteScar(CutPaste):
    def __init__(self, width=[2, 16], height=[10, 25], rotation=[-45, 45], **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.height = height
        self.rotation = rotation

    def __call__(self, imgs, subclass):
        batch_size, _, h, w = imgs.shape
        augmented_imgs = imgs.clone()

        for i in range(batch_size):
            img = imgs[i]
            augmented = self.process_image(img, subclass)
            augmented_imgs[i] = augmented

        return imgs, augmented_imgs

    def process_image(self, img, subclass):
        img = img.clone()
        _, h, w = img.shape

        target_foreground_mask = generate_target_foreground_mask(img, subclass)

        cut_w = int(random.uniform(*self.width))
        cut_h = int(random.uniform(*self.height))

        if cut_w <= 0 or cut_h <= 0:
            return img

        from_x = random.randint(0, w - cut_w)
        from_y = random.randint(0, h - cut_h)

        patch = img[:, from_y : from_y + cut_h, from_x : from_x + cut_w]

        if self.colorJitter is not None:
            patch = self.colorJitter(patch)

        rot_deg = random.uniform(*self.rotation)
        patch = TF.rotate(
            patch,
            angle=rot_deg,
            interpolation=TF.InterpolationMode.BILINEAR,
            expand=True,
        )

        _, patch_h, patch_w = patch.shape

        to_x = random.randint(0, w - patch_w)
        to_y = random.randint(0, h - patch_h)

        mask_indices = np.argwhere(target_foreground_mask == 1)
        if len(mask_indices) == 0:
            return img

        valid_indices = []
        for y, x in mask_indices:
            if y + patch_h <= h and x + patch_w <= w:
                valid_indices.append((y, x))

        if len(valid_indices) == 0:
            return img

        to_y, to_x = random.choice(valid_indices)

        augmented = img.clone()
        mask = torch.ones_like(patch)
        augmented = self.paste_with_mask(augmented, patch, mask, to_y, to_x)

        return augmented

    def paste_with_mask(self, img, patch, mask, top, left):
        _, h, w = img.shape
        _, patch_h, patch_w = patch.shape

        if top + patch_h > h or left + patch_w > w:
            return img

        img_region = img[:, top : top + patch_h, left : left + patch_w]
        mask = mask.to(img_region.device)
        img_region = img_region * (1 - mask) + patch * mask
        img[:, top : top + patch_h, left : left + patch_w] = img_region

        return img


class CutPasteUnion(object):
    def __init__(self, **kwargs):
        self.cutpaste_normal = CutPasteNormal(**kwargs)
        self.cutpaste_scar = CutPasteScar(**kwargs)

    def __call__(self, imgs, subclasses):
        batch_size = imgs.shape[0]
        augmented_imgs = imgs.clone()

        for i in range(batch_size):
            img = imgs[i].unsqueeze(0)
            subclass = subclasses[i]
            if random.random() < 0.5:
                _, augmented = self.cutpaste_normal(img, subclass)
            else:
                _, augmented = self.cutpaste_scar(img, subclass)
            augmented_imgs[i] = augmented.squeeze(0)

        return imgs, augmented_imgs


class SegPatchFolderPaste:
    def __init__(self, cfg, train_root, seed: int):
        synthesis_cfg = cfg.get("synthesis", {})
        self.train_root = Path(train_root).resolve()
        self.area_ratio = synthesis_cfg.get("area_ratio", [0.005, 0.05])
        self.feather_px = int(synthesis_cfg.get("feather_px", 2))
        self.color_match = bool(synthesis_cfg.get("color_match", True))
        self.max_tries = int(synthesis_cfg.get("max_tries", 20))
        self.seg_patch_dirname = synthesis_cfg.get("seg_patch_dirname", "seg_patch")
        self.paste_k = int(synthesis_cfg.get("paste_k", 1))
        if self.paste_k != 1:
            raise AssertionError(f"segpatch_folder currently requires paste_k=1, got {self.paste_k}")

        self.rng = random.Random(int(seed))
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.patch_cache: Dict[str, torch.Tensor] = {}
        self.patch_pool = self.build_index()

    def build_index(self) -> Dict[str, List[Path]]:
        pool: Dict[str, List[Path]] = {}
        if not self.train_root.is_dir():
            return pool

        for class_dir in sorted(self.train_root.iterdir()):
            if not class_dir.is_dir():
                continue
            patch_dir = class_dir / self.seg_patch_dirname
            patch_paths: List[Path] = []
            if patch_dir.is_dir():
                for pattern in ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"):
                    patch_paths.extend(sorted(patch_dir.glob(pattern)))
            pool[class_dir.name] = sorted(set(patch_paths))
        return pool

    def __call__(self, imgs_ok: torch.Tensor, paths: List[str]):
        augmented_imgs = imgs_ok.clone()
        for i, path in enumerate(paths):
            augmented_imgs[i] = self._paste_single(imgs_ok[i], path)
        return imgs_ok, augmented_imgs

    def _infer_class_name(self, image_path: str) -> str:
        path = Path(image_path).resolve()
        try:
            rel = path.relative_to(self.train_root)
            if rel.parts:
                return rel.parts[0]
        except ValueError:
            pass
        if path.parent.name == self.seg_patch_dirname:
            return path.parent.parent.name
        return path.parent.name

    def _load_patch(self, patch_path: Path, ref: torch.Tensor) -> torch.Tensor:
        cache_key = str(patch_path)
        cached = self.patch_cache.get(cache_key)
        if cached is None:
            patch = Image.open(patch_path).convert("RGB")
            cached = self.normalize(TF.to_tensor(patch))
            self.patch_cache[cache_key] = cached
        return cached.to(device=ref.device, dtype=ref.dtype)

    def _resize_patch(self, patch: torch.Tensor, out_h: int, out_w: int):
        _, patch_h, patch_w = patch.shape
        if patch_h <= 0 or patch_w <= 0:
            return None

        area_min, area_max = self.area_ratio
        target_area = self.rng.uniform(float(area_min), float(area_max)) * float(out_h * out_w)
        scale = math.sqrt(max(target_area, 1.0) / float(max(patch_h * patch_w, 1)))
        scale = min(scale, out_h / float(patch_h), out_w / float(patch_w))

        resize_h = max(1, min(out_h, int(round(patch_h * scale))))
        resize_w = max(1, min(out_w, int(round(patch_w * scale))))
        if resize_h <= 0 or resize_w <= 0:
            return None

        return F.interpolate(
            patch.unsqueeze(0),
            size=(resize_h, resize_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _match_patch_stats(self, patch: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
        patch_mean = patch.mean(dim=(1, 2), keepdim=True)
        region_mean = region.mean(dim=(1, 2), keepdim=True)
        patch_std = patch.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(1e-6)
        region_std = region.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(1e-6)
        return (patch - patch_mean) * (region_std / patch_std) + region_mean

    def _build_alpha(self, patch_h: int, patch_w: int, ref: torch.Tensor) -> torch.Tensor:
        alpha_y = torch.ones(patch_h, device=ref.device, dtype=ref.dtype)
        alpha_x = torch.ones(patch_w, device=ref.device, dtype=ref.dtype)

        feather_y = min(self.feather_px, patch_h // 2)
        feather_x = min(self.feather_px, patch_w // 2)

        for i in range(feather_y):
            value = float(i + 1) / float(feather_y + 1)
            alpha_y[i] = value
            alpha_y[-(i + 1)] = value

        for i in range(feather_x):
            value = float(i + 1) / float(feather_x + 1)
            alpha_x[i] = value
            alpha_x[-(i + 1)] = value

        return torch.outer(alpha_y, alpha_x).unsqueeze(0)

    def _paste_single(self, img_ok: torch.Tensor, image_path: str) -> torch.Tensor:
        class_name = self._infer_class_name(image_path)
        patch_candidates = self.patch_pool.get(class_name, [])
        if not patch_candidates:
            return img_ok

        img_abn = img_ok.clone()
        _, out_h, out_w = img_abn.shape

        for _ in range(self.paste_k):
            patch_path = Path(self.rng.choice(patch_candidates))
            patch = self._load_patch(patch_path, img_abn)
            patch = self._resize_patch(patch, out_h, out_w)
            if patch is None:
                continue

            _, patch_h, patch_w = patch.shape
            if patch_h > out_h or patch_w > out_w:
                continue

            for _ in range(self.max_tries):
                top = self.rng.randint(0, out_h - patch_h)
                left = self.rng.randint(0, out_w - patch_w)
                region = img_abn[:, top : top + patch_h, left : left + patch_w]
                patch_to_paste = patch.clone()
                if self.color_match:
                    patch_to_paste = self._match_patch_stats(patch_to_paste, region)
                alpha = self._build_alpha(patch_h, patch_w, img_abn)
                img_abn[:, top : top + patch_h, left : left + patch_w] = region * (1 - alpha) + patch_to_paste * alpha
                break

        return img_abn
