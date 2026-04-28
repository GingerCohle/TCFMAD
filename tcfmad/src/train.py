
from __future__ import annotations

import os, sys, random, logging
import time
import resource
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import yaml, numpy as np, torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler

from src.utils.logging import CSVLogger, gpu_timer, grad_logger, AverageMeter
from src.datasets.dataset import build_dataloader
from src.utils.synthesis import CutPasteUnion, SegPatchFolderPaste
from src.tcfmad import VisionModule

_GLOBAL_SEED = 0
random.seed(42); np.random.seed(0); torch.manual_seed(0)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

def cuda_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024.0 / 1024.0

def cuda_reserved_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_reserved() / 1024.0 / 1024.0

def reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

class Trainer:
    def __init__(self, args: Dict[str, Any]):
        # ---------- basic ----------
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)

        # ---------- model ----------
        mcfg = args["meta"]
        self.model = VisionModule(
            mcfg["model"], mcfg["pred_depth"], mcfg["pred_emb_dim"], if_pe=mcfg.get("if_pred_pe", True), feat_normed=mcfg.get("feat_normed", False),
        )
        self.n_layer = args["meta"].get("n_layer", 3)
        self.model.predictor.requires_grad_(True)
        if self.model.projector:
            self.model.projector.requires_grad_(True)
        self.loss_mode = args["meta"].get("loss_mode", "l2") # l2 or smooth_l1
        logger.info(f"Loss mode {self.loss_mode}")
        training_cfg = args.get("training", {})
        self.multi_layer_mean = bool(training_cfg.get("multi_layer_mean", False))
        self.consistency_loss = bool(training_cfg.get("consistency_loss", False))
        self.cons_lambda = float(training_cfg.get("cons_lambda", 0.1))
        self.training_layers = training_cfg.get("layers")
        self.model.fuse_h.requires_grad_(self.multi_layer_mean)
        self.model.fuse_z.requires_grad_(self.multi_layer_mean)
        if self.multi_layer_mean:
            alpha_h_mean, alpha_z_mean = self.model.fusion_alpha_means()
            logger.info(
                "[LEARNED-FUSION][train] enabled layers=%s alpha_h_mean=%s alpha_z_mean=%s consistency_loss=%s cons_lambda=%.4f",
                self.model.resolve_feature_layers(self.n_layer, self.training_layers),
                [round(x, 4) for x in alpha_h_mean.cpu().tolist()],
                [round(x, 4) for x in alpha_z_mean.cpu().tolist()],
                self.consistency_loss,
                self.cons_lambda,
            )

        # ---------- data ----------
        dcfg = args["data"]
        assert dcfg["dataset"] in dcfg["data_name"] # check if the dataset aligns with the few-shot folder
        _, self.loader, self.sampler = build_dataloader(
            mode="train",
            root=dcfg["train_root"],
            batch_size=dcfg["batch_size"],
            pin_mem=dcfg["pin_mem"],
            resize=mcfg["crop_size"],
            use_hflip=dcfg.get("use_hflip",False),
            use_vflip=dcfg.get("use_vflip",False),
            use_rotate90=dcfg.get("use_rotate90",False),
            use_color_jitter=dcfg.get("use_color_jitter",False),
            use_gray=dcfg.get("use_gray",False),
            use_blur=dcfg.get("use_blur",False),
        )
        synthesis_cfg = args.get("synthesis", {})
        self.synthesis_mode = synthesis_cfg.get("mode", "cutpaste")
        self.synthesis_seed = int(synthesis_cfg.get("seed", 0)) + int(args.get("rank", 0))
        if self.synthesis_mode == "cutpaste":
            self.synth = CutPasteUnion(colorJitter=0.5)
        elif self.synthesis_mode == "segpatch_folder":
            train_image_root = Path(dcfg["train_root"]) / "train"
            self.synth = SegPatchFolderPaste(args, train_image_root, self.synthesis_seed)
            patch_classes = sum(1 for patches in self.synth.patch_pool.values() if patches)
            patch_count = sum(len(patches) for patches in self.synth.patch_pool.values())
            logger.info(
                "[SYNTH][segpatch_folder] train_root=%s classes_with_patches=%d total_patches=%d seed=%d",
                train_image_root,
                patch_classes,
                patch_count,
                self.synthesis_seed,
            )
        else:
            raise ValueError(f"Unknown synthesis.mode: {self.synthesis_mode}")
        self.batch_size = dcfg["batch_size"]

        # ---------- optimization ----------
        from src.helper import init_opt

        ocfg = args["optimization"]
        self.optimizer, self.scheduler, self.scaler = init_opt(
            predictor=self.model.predictor,
            extra_modules=[self.model.fuse_h, self.model.fuse_z] if self.multi_layer_mean else None,
            wd=float(ocfg["weight_decay"]),
            lr=ocfg["lr"],
            lr_config=ocfg.get("lr_config", "const"),
            max_epoch=ocfg["epochs"],                         # for cosine_warmup
            min_lr=ocfg.get("min_lr", 1e-6),                  # for cosine_warmup
            warmup_epoch=ocfg.get("warmup_epoch", 5),         # for cosine_warmup
            step_size=ocfg.get("step_size", 300),             # for step
            gamma=ocfg.get("gamma", 0.1),                     # for step
        )
        self.epochs = ocfg["epochs"]
        self.use_bf16 = mcfg["use_bfloat16"]
        self.profile = bool(args.get("profile", False))
        self.profile_max_steps = int(args.get("profile_max_steps", 20))
        self.profile_warmup_steps = 3
        trainer_cfg = args.get("trainer", {})
        self.max_steps = trainer_cfg.get("max_steps")
        if self.max_steps is not None:
            self.max_steps = int(self.max_steps)
            logger.info("[TRAIN] max_steps=%d", self.max_steps)

        # ---------- logging ----------
        lcfg: Dict[str, Any] = args.get("logging", {})
        log_dir = Path(lcfg.get("folder", "logs"))
        # log_dir.mkdir(parents=True, exist_ok=True)     
        self.ckpt_dir = log_dir

        self.tag = lcfg.get("write_tag", "train")      
        
        self.csv_logger = CSVLogger(
            str(self.ckpt_dir / f"{self.tag}.csv"),
            ("%d", "epoch"),
            ("%d", "itr"),
            ("%.5f", "loss"),
            ("%d", "time (ms)"),
        )

        if self.profile:
            trainable = [(n, p.numel()) for n, p in self.model.named_parameters() if p.requires_grad]
            pred_numel = sum(num for n, num in trainable if n.startswith("predictor."))
            enc_numel = sum(num for n, num in trainable if n.startswith("encoder."))
            proj_numel = sum(num for n, num in trainable if n.startswith("projector."))
            total_numel = sum(num for _, num in trainable)
            logger.info(
                "[PROFILE][train] enabled profile_max_steps=%d warmup_steps=%d",
                self.profile_max_steps,
                self.profile_warmup_steps
            )
            logger.info(
                "[PROFILE][train] trainable_params_total=%d predictor=%d encoder=%d projector=%d",
                total_numel, pred_numel, enc_numel, proj_numel
            )

    def _sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _make_abnormal_images(self, imgs, labels, paths) -> torch.Tensor:
        if self.synthesis_mode == "segpatch_folder":
            _, imgs_abn = self.synth(imgs, paths)
        else:
            _, imgs_abn = self.synth(imgs, labels)
        return imgs_abn

    def _loss_fn(self, h, p) -> torch.Tensor:
        if self.loss_mode == 'l2':
            return F.mse_loss(h.flatten(0,1), p.flatten(0,1), reduction="mean")
        elif self.loss_mode == 'smooth_l1':
            return F.smooth_l1_loss(h.flatten(0,1), p.flatten(0,1), reduction="mean")
        else:
            raise NotImplementedError(f"Loss mode {self.loss_mode} not implemented")

    def _consistency_loss_fn(self, layer_preds: List[torch.Tensor], mean_pred: torch.Tensor) -> torch.Tensor:
        losses = [F.mse_loss(pred, mean_pred, reduction="mean") for pred in layer_preds]
        return torch.stack(losses).mean()

    def _save_ckpt(self, ep, step=None):
        name = f"{self.tag}-step{step}.pth.tar" if step else f"{self.tag}-ep{ep}.pth.tar"
        torch.save({"predictor": self.model.predictor.state_dict(),
                    "projector": self.model.projector.state_dict() if self.model.projector else None,
                    "fuse_h": self.model.fuse_h.state_dict(),
                    "fuse_z": self.model.fuse_z.state_dict(),
                    "epoch": ep, "lr": self.optimizer.param_groups[0]["lr"]}, self.ckpt_dir/name)

    def train(self):
        mp.set_start_method("spawn", force=True); gstep = 0
        step_times_ms: List[float] = []
        profile_segments = (
            "h2d",
            "encoder_fwd",
            "predictor_fwd",
            "loss",
            "backward",
            "optimizer_step",
            "zero_grad",
        )
        profile_stats = {
            name: {
                "delta_alloc_mb": [],
                "segment_peak_mb": [],
            }
            for name in profile_segments
        }
        overall_peak = {"allocated_mb": 0.0, "reserved_mb": 0.0}

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        def run_segment(name: str, fn, step_idx: int, collect: bool):
            if not (self.profile and torch.cuda.is_available()):
                return fn()
            self._sync()
            before_alloc = cuda_mem_mb()
            reset_peak()
            out = fn()
            self._sync()
            after_alloc = cuda_mem_mb()
            seg_peak = peak_mb()
            reserved = cuda_reserved_mb()
            overall_peak["allocated_mb"] = max(overall_peak["allocated_mb"], seg_peak)
            overall_peak["reserved_mb"] = max(overall_peak["reserved_mb"], reserved)
            delta_alloc = after_alloc - before_alloc
            if collect:
                profile_stats[name]["delta_alloc_mb"].append(delta_alloc)
                profile_stats[name]["segment_peak_mb"].append(seg_peak)
                logger.info(
                    "[PROFILE][segment][step=%d] %s before_alloc_mb=%.2f after_alloc_mb=%.2f delta_alloc_mb=%.2f segment_peak_mb=%.2f reserved_mb=%.2f",
                    step_idx,
                    name,
                    before_alloc,
                    after_alloc,
                    delta_alloc,
                    seg_peak,
                    reserved,
                )
            return out

        for ep in range(self.epochs):
            logger.info("Epoch %d", ep+1); self.sampler.set_epoch(ep); loss_m, time_m = AverageMeter(), AverageMeter()
            for itr, (imgs, labels, paths) in enumerate(self.loader):
                if self.max_steps is not None and gstep >= self.max_steps:
                    break
                if self.profile and gstep >= self.profile_max_steps:
                    break
                step_idx = gstep + 1
                collect_stats = step_idx > self.profile_warmup_steps
                if self.profile:
                    self._sync()
                    t0 = time.perf_counter()

                    imgs = run_segment("h2d", lambda: imgs.to(self.device, non_blocking=True), step_idx, collect_stats)
                    imgs_abn = self._make_abnormal_images(imgs, labels, paths)
                    overall_peak["allocated_mb"] = max(overall_peak["allocated_mb"], cuda_mem_mb())
                    overall_peak["reserved_mb"] = max(overall_peak["reserved_mb"], cuda_reserved_mb())
                    use_clean_context = np.random.rand() < 0.5

                    with autocast(dtype=torch.bfloat16, enabled=self.use_bf16):
                        def _encoder_fwd():
                            if not self.multi_layer_mean:
                                h_local = self.model.target_features(imgs, paths, n_layer=self.n_layer)
                                if use_clean_context:
                                    z_local = self.model._extract(imgs, paths, n_layer=self.n_layer)
                                else:
                                    z_local = self.model._extract(imgs_abn, paths, n_layer=self.n_layer)
                                return h_local, z_local

                            h_layers_local = self.model.target_multi_layer_features(
                                imgs, paths, n_layer=self.n_layer, layers=self.training_layers
                            )
                            if use_clean_context:
                                z_layers_local = self.model.multi_layer_features(
                                    imgs, paths, n_layer=self.n_layer, layers=self.training_layers
                                )
                            else:
                                z_layers_local = self.model.multi_layer_features(
                                    imgs_abn, paths, n_layer=self.n_layer, layers=self.training_layers
                                )
                            return h_layers_local, z_layers_local

                        h_or_layers, z_or_layers = run_segment("encoder_fwd", _encoder_fwd, step_idx, collect_stats)

                        def _predictor_fwd():
                            if not self.multi_layer_mean:
                                return self.model.predictor(self.model.dropout(z_or_layers))

                            z_fused_local, _ = self.model.fuse_z_layers(z_or_layers)
                            p_fused_local = self.model.predictor(self.model.dropout(z_fused_local))
                            if not self.consistency_loss:
                                return p_fused_local, None
                            layer_preds_local = [
                                self.model.predictor(self.model.dropout(z_layer))
                                for z_layer in z_or_layers
                            ]
                            return p_fused_local, layer_preds_local

                        pred_out = run_segment("predictor_fwd", _predictor_fwd, step_idx, collect_stats)

                        def _loss_fwd():
                            if not self.multi_layer_mean:
                                return self._loss_fn(h_or_layers, pred_out)

                            h_fused_local, _ = self.model.fuse_h_layers(h_or_layers)
                            p_fused_local, layer_preds_local = pred_out
                            main_loss = self._loss_fn(h_fused_local, p_fused_local)
                            if not self.consistency_loss:
                                return main_loss
                            cons_loss = self._consistency_loss_fn(layer_preds_local, p_fused_local)
                            return main_loss + self.cons_lambda * cons_loss

                        loss = run_segment("loss", _loss_fwd, step_idx, collect_stats)

                    self._sync()
                    t = (time.perf_counter() - t0) * 1000.0
                else:
                    imgs = imgs.to(self.device, non_blocking=True)
                    imgs_abn = self._make_abnormal_images(imgs, labels, paths)
                    def _step():
                        with autocast(dtype=torch.bfloat16, enabled=self.use_bf16):
                            if not self.multi_layer_mean:
                                if np.random.rand() < 0.5:
                                    h = self.model.target_features(imgs, paths, n_layer=self.n_layer); _, p = self.model.context_features(imgs, paths, n_layer=self.n_layer)
                                else:
                                    h = self.model.target_features(imgs, paths, n_layer=self.n_layer); _, p = self.model.context_features(imgs_abn, paths, n_layer=self.n_layer)
                                return self._loss_fn(h, p,)

                            context_imgs = imgs if np.random.rand() < 0.5 else imgs_abn
                            h_layers = self.model.target_multi_layer_features(
                                imgs, paths, n_layer=self.n_layer, layers=self.training_layers
                            )
                            z_layers = self.model.multi_layer_features(
                                context_imgs, paths, n_layer=self.n_layer, layers=self.training_layers
                            )
                            h_fused, _ = self.model.fuse_h_layers(h_layers)
                            z_fused, _ = self.model.fuse_z_layers(z_layers)
                            p_fused = self.model.predictor(self.model.dropout(z_fused))
                            loss = self._loss_fn(h_fused, p_fused)
                            if self.consistency_loss:
                                layer_preds = [
                                    self.model.predictor(self.model.dropout(z_layer))
                                    for z_layer in z_layers
                                ]
                                loss = loss + self.cons_lambda * self._consistency_loss_fn(layer_preds, p_fused)
                            return loss
                    (loss,), t = gpu_timer(lambda: [_step()])

                if self.profile:
                    if self.use_bf16:
                        run_segment("backward", lambda: self.scaler.scale(loss).backward(), step_idx, collect_stats)
                        run_segment("optimizer_step", lambda: self.scaler.step(self.optimizer), step_idx, collect_stats)
                        self.scaler.update()
                    else:
                        run_segment("backward", lambda: loss.backward(), step_idx, collect_stats)
                        run_segment("optimizer_step", lambda: self.optimizer.step(), step_idx, collect_stats)
                    self._sync()
                    grad_stats = grad_logger(self.model.predictor.named_parameters())
                    run_segment("zero_grad", lambda: self.optimizer.zero_grad(), step_idx, collect_stats)
                else:
                    if self.use_bf16: self.scaler.scale(loss).backward(); self.scaler.step(self.optimizer); self.scaler.update()
                    else: loss.backward(); self.optimizer.step()
                    self._sync()
                    grad_stats = grad_logger(self.model.predictor.named_parameters()); self.optimizer.zero_grad()

                loss_m.update(loss.item()); time_m.update(t); gstep += 1
                if self.profile:
                    step_times_ms.append(t)
                if gstep % 100 == 0: self._save_ckpt(ep, gstep)
                self.csv_logger.log(ep+1, itr, loss.item(), t)
                if itr % 100 == 0:
                    mem_mb = (torch.cuda.max_memory_allocated()/1024**2) if torch.cuda.is_available() else 0.0
                    logger.info("[E %d I %d] loss %.6f (avg %.6f) mem %.2fMB (%.1fms)", ep+1, itr, loss.item(), loss_m.avg, mem_mb, time_m.avg)
                    if grad_stats:
                        logger.info("    grad: [%.2e %.2e] (%.2e %.2e)", grad_stats.first_layer, grad_stats.last_layer, grad_stats.min, grad_stats.max)
            logger.info(
                "Epoch %d complete. Avg loss %.6f, lr %.6f",
                ep + 1,
                loss_m.avg,
                self.optimizer.param_groups[0]['lr']
            )
            if self.scheduler is not None:
                self.scheduler.step()
            if self.max_steps is not None and gstep >= self.max_steps:
                break
            if self.profile and gstep >= self.profile_max_steps:
                break

        if self.profile and step_times_ms:
            warm = min(self.profile_warmup_steps, len(step_times_ms))
            eval_slice = step_times_ms[warm:] if warm < len(step_times_ms) else step_times_ms
            avg_step_ms = float(np.mean(eval_slice))
            cpu_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            logger.info("[PROFILE][train] steps=%d warmup_steps=%d", len(step_times_ms), warm)
            logger.info("[PROFILE][train] avg_step_time_ms=%.3f", avg_step_ms)
            measured_steps = len(profile_stats["h2d"]["delta_alloc_mb"])
            logger.info("[PROFILE][train] measured_steps_excluding_warmup=%d", measured_steps)
            logger.info("[PROFILE][summary] segment,delta_alloc_mb_mean,delta_alloc_mb_std,segment_peak_mb_mean,segment_peak_mb_std")
            for name in profile_segments:
                deltas = profile_stats[name]["delta_alloc_mb"]
                peaks = profile_stats[name]["segment_peak_mb"]
                if deltas and peaks:
                    d_mean = float(np.mean(deltas)); d_std = float(np.std(deltas))
                    p_mean = float(np.mean(peaks)); p_std = float(np.std(peaks))
                else:
                    d_mean = d_std = p_mean = p_std = float("nan")
                logger.info(
                    "[PROFILE][summary] %s,%.4f,%.4f,%.4f,%.4f",
                    name,
                    d_mean,
                    d_std,
                    p_mean,
                    p_std,
                )
            if torch.cuda.is_available():
                activation_peak_mb = 0.0
                for name in ("encoder_fwd", "predictor_fwd", "loss"):
                    peaks = profile_stats[name]["segment_peak_mb"]
                    if peaks:
                        activation_peak_mb = max(activation_peak_mb, float(max(peaks)))
                logger.info(
                    "[PROFILE][train] peak_gpu_mem_allocated_mb=%.2f peak_gpu_mem_reserved_mb=%.2f",
                    overall_peak["allocated_mb"],
                    overall_peak["reserved_mb"],
                )
                logger.info("[PROFILE][train] activation_related_peak_mb=%.2f", activation_peak_mb)
            else:
                logger.info("[PROFILE][train] CUDA unavailable; GPU memory stats not available.")
            logger.info("[PROFILE][train] cpu_max_rss_mb=%.2f", cpu_rss_mb)

def main(args: Dict[str, Any]) -> None:
    if args is None:
        cfg_path = Path(__file__).with_name("params.yaml");
        if not cfg_path.exists(): raise FileNotFoundError("No args provided and default parameter file does not exist")
        with open(cfg_path) as f: args = yaml.safe_load(f)
    Trainer(args).train()

if __name__ == "__main__":
    main()
