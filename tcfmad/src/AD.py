import os
import logging
import math
import time
import resource
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from matplotlib import cm, pyplot as plt
from PIL import Image

from src.datasets.dataset import build_dataloader
from src.utils.metrics import (
    calculate_pro,
    compute_imagewise_retrieval_metrics,
    compute_pixelwise_retrieval_metrics,
)
from src.helper import save_segmentation_grid
from src.utils.logging import CSVLogger
from src.tcfmad import VisionModule          

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("evaluator")


def _build_model(meta: Dict[str, Any]) -> VisionModule:
    return VisionModule(
        model_name=meta["model"],
        pred_depth=meta["pred_depth"],
        pred_emb_dim=meta["pred_emb_dim"],
        if_pe=meta.get("if_pred_pe", True),
        feat_normed=meta.get("feat_normed", False),
    )


def _load_checkpoint_modules(model: VisionModule, state: Dict[str, Any], ckpt: Path) -> None:
    model.predictor.load_state_dict(state["predictor"])
    projector_state = state.get("projector")
    if model.projector is not None and projector_state is not None:
        model.projector.load_state_dict(projector_state)

    skipped_fusions = []
    for key in ("fuse_h", "fuse_z"):
        fusion_state = state.get(key)
        if fusion_state is None:
            skipped_fusions.append(f"{key} (missing)")
            continue
        try:
            getattr(model, key).load_state_dict(fusion_state, strict=False)
        except RuntimeError:
            skipped_fusions.append(f"{key} (incompatible shape)")

    if skipped_fusions:
        logger.warning(
            "Checkpoint %s skipped %s; using default uniform fusion weights.",
            ckpt,
            ", ".join(skipped_fusions),
        )


def _resolve_eval_setup(cfg: Dict[str, Any]) -> tuple[str, List[str], int]:
    dataset_name = cfg["data"].get("dataset", "mvtec")
    if dataset_name == "mvtec":
        classnames = cfg["data"]["mvtec_classnames"]
        topk = cfg["testing"]["K_top_mvtec"]
    elif dataset_name == "visa":
        classnames = cfg["data"]["visa_classnames"]
        topk = cfg["testing"]["K_top_visa"]
    else:
        raise NotImplementedError
    return dataset_name, classnames, topk

@torch.inference_mode()
def _evaluate_single_ckpt(ckpt: Path, cfg: Dict[str, Any]) -> None:
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = (device.type == "cuda")

    def _sync():
        if use_cuda:
            torch.cuda.synchronize()

    model = _build_model(cfg["meta"])
    state = torch.load(ckpt, map_location="cpu")
    _load_checkpoint_modules(model, state, ckpt)
    model.to(device)
    model.eval()

    crop = cfg["meta"]["crop_size"]
    n_layer = cfg["meta"].get("n_layer", 3)
    profile = bool(cfg.get("profile", False))
    profile_max_images = int(cfg.get("profile_max_images", 32))
    profile_warmup_images = 1
    testing_cfg = cfg.get("testing", {})
    multi_layer_mean = bool(testing_cfg.get("multi_layer_mean", False))
    testing_layers = testing_cfg.get("layers")
    test_batch_size = int(testing_cfg.get("batch_size", cfg["data"].get("batch_size", 1)))
    test_num_workers = int(testing_cfg.get("num_workers", 0))
    segmentation_vis = bool(testing_cfg.get("segmentation_vis", False))
    rank = int(cfg.get("rank", 0))
    world_size = int(cfg.get("world_size", 1))

    if profile:
        test_batch_size = max(1, min(test_batch_size, profile_max_images))
    else:
        test_batch_size = max(1, test_batch_size)

    if profile:
        logger.info(
            "[PROFILE][infer] enabled profile_max_images=%d warmup_images=%d",
            profile_max_images, profile_warmup_images
        )
        logger.info(
            "[PROFILE][infer] model=%s crop=%s n_layer=%s batch_size=%d num_workers=%d",
            cfg["meta"].get("model"), crop, n_layer, test_batch_size, test_num_workers
        )
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
    if multi_layer_mean and rank == 0:
        alpha_h_mean, _ = model.fusion_alpha_means()
        logger.info(
            "learned fusion enabled | layers=%s | alpha_h_mean=%s",
            model.resolve_feature_layers(n_layer=n_layer, layers=testing_layers),
            [round(x, 4) for x in alpha_h_mean.cpu().tolist()],
        )

    # error = cfg["meta"].get("loss_mode", "l2")

    dataset_name, classnames, K = _resolve_eval_setup(cfg)
    class_order = {cls_name: idx for idx, cls_name in enumerate(classnames)}
    local_classnames = [cls_name for idx, cls_name in enumerate(classnames) if idx % world_size == rank]
    assert dataset_name in cfg["data"]["test_root"] # check if eval on the same dataset the ckpt trained on

    logger.info(
        "Evaluating %s on %s | rank=%d/%d | local_classes=%s",
        ckpt.name,
        dataset_name,
        rank,
        world_size,
        local_classnames,
    )
    
    os.makedirs(Path(cfg["logging"]["folder"]), exist_ok=True)

    enc_ms_total = 0.0
    pred_ms_total = 0.0
    heatmap_ms_total = 0.0
    metrics_ms_total = 0.0
    per_image_ms: List[float] = []
    processed_images = 0
    local_rows: List[tuple[str, float, float, float, float]] = []

    if segmentation_vis:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    for cls in local_classnames:
        if profile and processed_images >= profile_max_images:
            break
        _, loader, _ = build_dataloader(
            mode="test",
            root=cfg["data"]["test_root"],
            batch_size=test_batch_size,
            pin_mem=cfg["data"].get("pin_mem", True),
            num_workers=test_num_workers,
            classname=cls,
            resize=crop,
            datasetname=dataset_name,
        )

        print(f"Evaluating {cls}...")

        patch_scores, labels = [], []
        pix_buf, mask_buf, name_buf = [], [], []
        img_buf = [] if segmentation_vis else None

        for batch in loader:
            if profile and processed_images >= profile_max_images:
                break
            img = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            paths = batch["image_path"]; labels.extend(batch["is_anomaly"]); name_buf.extend(batch["image_name"])

            _sync()
            iter_t0 = time.perf_counter()

            _sync()
            t0 = time.perf_counter()
            if multi_layer_mean:
                enc_layers = model.target_multi_layer_features(img, paths, n_layer=n_layer, layers=testing_layers)
                enc, _ = model.fuse_h_layers(enc_layers)
            else:
                enc = model.target_features(img, paths, n_layer=n_layer)
            _sync()
            enc_ms = (time.perf_counter() - t0) * 1000.0

            _sync()
            t1 = time.perf_counter()
            pred = model.predict(enc)
            _sync()
            pred_ms = (time.perf_counter() - t1) * 1000.0

            _sync()
            t2 = time.perf_counter()
            l = F.mse_loss(enc, pred, reduction="none").mean(dim=2)

            topk = torch.topk(l, K, dim=1).values.mean(dim=1)
            patch_scores.extend(topk.cpu())
            h = w = int(math.sqrt(l.size(1)))
            pix = F.interpolate(l.view(-1,1,h,w), size=img.shape[2:], mode="bilinear", align_corners=False)
            pix_buf.append(pix.squeeze(1).cpu()); mask_buf.append(mask.cpu())
            if segmentation_vis:
                img_buf.append(img.cpu())
            _sync()
            heatmap_ms = (time.perf_counter() - t2) * 1000.0
            iter_ms = (time.perf_counter() - iter_t0) * 1000.0

            batch_size = int(img.shape[0])
            processed_images += batch_size
            if profile:
                enc_ms_total += enc_ms
                pred_ms_total += pred_ms
                heatmap_ms_total += heatmap_ms
                per_image_ms.extend([iter_ms / batch_size] * batch_size)

        p_np = torch.tensor(patch_scores).numpy()
        p_np = (p_np - p_np.min()) / (p_np.max() - p_np.min() + 1e-8) # normed

        pix_all = torch.cat(pix_buf)
        gmin, gmax = pix_all.min(), pix_all.max()
        pix_norm = ((pix_all - gmin) / (gmax - gmin + 1e-8)).numpy()
        mask_np  = torch.cat(mask_buf).squeeze(1).numpy()

        _sync()
        t3 = time.perf_counter()
        inst = compute_imagewise_retrieval_metrics(p_np, np.array(labels))
        pix  = compute_pixelwise_retrieval_metrics(pix_norm, mask_np)
        pro  = calculate_pro(mask_np, pix_norm,
                             max_steps=cfg["testing"]["max_steps"], expect_fpr=cfg["testing"]["expect_fpr"])
        _sync()
        metrics_ms_total += (time.perf_counter() - t3) * 1000.0

        logger.info("%s | AUROC_i %.4f | AUPR_i %.4f | AUROC_p %.4f | PRO-AUC %.4f",
                    cls, inst["auroc"], inst["aupr"], pix["auroc"], pro)
        local_rows.append((cls, inst["auroc"], inst["aupr"], pix["auroc"], pro))

        # Generate visualizations
        if segmentation_vis:
            std_cpu, mean_cpu = std.cpu(), mean.cpu()
            imgs_un = (torch.cat(img_buf) * std_cpu + mean_cpu).permute(0,2,3,1).numpy()
            out_dir = Path(cfg["logging"]["folder"]) / "segmentation" / cls
            save_segmentation_grid(out_dir, name_buf, imgs_un, mask_np, pix_norm)

    if dist.is_available() and dist.is_initialized() and world_size > 1:
        gathered_rows: List[List[tuple[str, float, float, float, float]]] = [None] * world_size
        dist.all_gather_object(gathered_rows, local_rows)
    else:
        gathered_rows = [local_rows]

    if rank == 0:
        all_rows = [row for rows in gathered_rows for row in rows]
        all_rows.sort(key=lambda row: class_order[row[0]])

        csv_path = Path(cfg["logging"]["folder"]) / f"{cfg['logging']['write_tag']}_eval.csv"
        csv_logger = CSVLogger(
            csv_path,
            ("%s", "checkpoint"), ("%s", "class"),
            ("%.8f", "inst_auroc"), ("%.8f", "inst_aupr"),
            ("%.8f", "pix_auroc"),  ("%.8f", "pro_auc"),
        )

        for cls, inst_auroc, inst_aupr, pix_auroc, pro_auc in all_rows:
            csv_logger.log(ckpt.name, cls, inst_auroc, inst_aupr, pix_auroc, pro_auc)

        inst_auc = [row[1] for row in all_rows]
        inst_aupr = [row[2] for row in all_rows]
        pix_auc = [row[3] for row in all_rows]
        pro_auc = [row[4] for row in all_rows]

        logger.info("Mean | AUROC_i %.4f | AUPR_i %.4f | AUROC_p %.4f | PRO-AUC %.4f",
                    np.mean(inst_auc), np.mean(inst_aupr), np.mean(pix_auc), np.mean(pro_auc))
        csv_logger.log(ckpt.name, "Mean", np.mean(inst_auc), np.mean(inst_aupr),
                       np.mean(pix_auc), np.mean(pro_auc))

    if profile and per_image_ms:
        n = len(per_image_ms)
        warm = min(profile_warmup_images, n)
        eval_slice = per_image_ms[warm:] if warm < n else per_image_ms
        avg_ms = float(np.mean(eval_slice))
        throughput = (1000.0 / avg_ms) if avg_ms > 0 else 0.0
        avg_enc = enc_ms_total / n
        avg_pred = pred_ms_total / n
        avg_heat = heatmap_ms_total / n
        avg_metrics = metrics_ms_total / max(len(local_rows), 1)
        cpu_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

        if use_cuda:
            peak_alloc_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024.0 / 1024.0
            logger.info(
                "[PROFILE][infer] peak_gpu_mem_allocated_mb=%.2f peak_gpu_mem_reserved_mb=%.2f",
                peak_alloc_mb, peak_reserved_mb
            )
        else:
            logger.info("[PROFILE][infer] CUDA unavailable; GPU memory stats not available.")

        logger.info("[PROFILE][infer] cpu_max_rss_mb=%.2f", cpu_rss_mb)
        logger.info("[PROFILE][infer] processed_images=%d warmup_images=%d", n, warm)
        logger.info("[PROFILE][infer] avg_latency_ms_per_image=%.3f throughput_img_per_s=%.3f", avg_ms, throughput)
        logger.info(
            "[PROFILE][infer] avg_stage_ms_per_image encoder=%.3f predictor=%.3f heatmap=%.3f metrics_per_class=%.3f",
            avg_enc, avg_pred, avg_heat, avg_metrics
        )
    

@torch.inference_mode()
def _demo(ckpt: Path, cfg: Dict[str, Any]) -> None:
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = _build_model(cfg["meta"])
    state = torch.load(ckpt, map_location="cpu")
    _load_checkpoint_modules(model, state, ckpt)
    model.to(device)
    model.eval()

    crop = cfg["meta"]["crop_size"]
    n_layer = cfg["meta"].get("n_layer", 3)
    testing_cfg = cfg.get("testing", {})
    multi_layer_mean = bool(testing_cfg.get("multi_layer_mean", False))
    testing_layers = testing_cfg.get("layers")
    out_root = Path(cfg["logging"]["folder"]) / "heatmaps"
    out_root.mkdir(parents=True, exist_ok=True)

    dataset_name = cfg["data"].get("dataset", "mvtec")
    assert dataset_name in cfg["data"]["test_root"] # check if eval on the same dataset the ckpt trained on
    
    test_root = Path(cfg["data"]["test_root"])
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIF", "*.TIFF", "*.WEBP")
    img_paths: List[Path] = []
    for ext in exts:
        img_paths += list(test_root.rglob(ext))
    img_paths = sorted(set(img_paths))
    if not img_paths:
        raise FileNotFoundError(f"No images found under: {test_root}")
    print(f"[INFO] Found {len(img_paths)} images under {test_root}")
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    def _load_and_preprocess(path: Path):
        pil = Image.open(path).convert("RGB")

        W0, H0 = pil.size

        pil_resized = pil.resize((crop, crop), Image.BILINEAR)

        img = torch.from_numpy(np.array(pil_resized)).float() / 255.0   # [H,W,3], 0~1
        img = img.permute(2, 0, 1).unsqueeze(0).to(device)              # [1,3,H,W]
        img = (img - mean) / std

        return pil, (W0, H0), img
    
    def _to_numpy_image(t_img: torch.Tensor):
        # t_img: [1,3,H,W]
        x = (t_img * std + mean).clamp(0, 1)
        x = x[0].permute(1, 2, 0).detach().cpu().numpy()  # [H,W,3]
        return (x * 255.0).astype(np.uint8)
    
    def _save_overlay_heatmap(rgb_uint8: np.ndarray, heat: np.ndarray, save_path: Path, alpha: float = 0.5):
        """
        rgb_uint8: [H,W,3] 0~255
        heat:      [H,W]   0~1
        """
        import cv2
        H, W = heat.shape

        heat_255 = (heat * 255.0).clip(0, 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_255, cv2.COLORMAP_JET)      # BGR
        rgb_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)            # RGB->BGR
        overlay = cv2.addWeighted(heat_color, alpha, rgb_bgr, 1 - alpha, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        Image.fromarray(overlay_rgb).save(save_path)

    for i, path in enumerate(img_paths, 1):
        pil_orig, (W0, H0), img = _load_and_preprocess(path)

        if multi_layer_mean:
            enc_layers = model.target_multi_layer_features(img, [str(path)], n_layer=n_layer, layers=testing_layers)
            enc, _ = model.fuse_h_layers(enc_layers)
        else:
            enc = model.target_features(img, [str(path)], n_layer=n_layer)  # [1, P, D]
        pred = model.predict(enc)                                       # [1, P, D]

        l = F.mse_loss(enc, pred, reduction="none").mean(dim=2)         # [1, P]

        h = w = int(math.sqrt(l.size(1)))
        pix = F.interpolate(l.view(1, 1, h, w), size=img.shape[2:], mode="bilinear", align_corners=False)  # [1,1,H,W]
        pix = pix.squeeze(0).squeeze(0)  # [H,W]

        pmin, pmax = pix.min(), pix.max()
        pix_norm = (pix - pmin) / (pmax - pmin + 1e-8)                  # [H,W], 0~1

        img_uint8 = _to_numpy_image(img)                                 # [H,W,3] @ crop

        rel = path.relative_to(test_root)
        save_dir = (out_root / rel.parent)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{path.stem}_heatmap.png"

        _save_overlay_heatmap(img_uint8, pix_norm.detach().cpu().numpy(), save_path)
        print(f"[{i}/{len(img_paths)}] Saved: {save_path}")


def main(args: Dict[str, Any]) -> None:
    ckpt = Path(args["ckpt_path"])
    print(f"loading {ckpt}...")
    _evaluate_single_ckpt(ckpt, args)
    logger.info("Finished. Metrics appended to CSV.")

if __name__ == "__main__":
    main()
