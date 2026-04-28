# TCFMAD Pipeline Trace
Generated on: 2026-03-05

This traces the exact call path in code, from dataset loading to scoring/localization/metrics.

## 1) Global orchestration and mode routing

Call order:

1. `tcfmad/main.py::main(cfg)`
2. `OmegaConf.to_container` -> plain dict
3. Spawn one process per `devices` entry
4. `tcfmad/main.py::process_main(rank, cfg_dict, world_size)`
5. Route by `mode`:
   - `train` -> `src.train.main(args=params)`
   - `AD` -> load saved `params.yaml`, build `ckpt_path=train-step<ckpt_step>.pth.tar`, then `src.AD.main(args=params)`
   - `demo` -> load saved `params.yaml`, build `ckpt_path=pretrained.pth.tar`, then `src.AD._demo(...)`

## 2) Dataset and dataloader flow

## 2.1 Few-shot training data generation (pre-step)

- Entrypoint: `tcfmad/src/sample.py::main` -> `sample_images(...)`
- Reads from full dataset class folders:
  - checks `train/good` or `train/ok`
- Writes few-shot set to:
  - `<target>/train/<class_name>/<image files>`

This `target` becomes `data.train_root` (via `data.data_path + data.data_name`).

## 2.2 Train dataloader

Call order:

1. `src.train.Trainer.__init__`
2. `build_dataloader(mode="train", root=data.train_root, ...)`
3. `TrainDataset(root)` where ImageFolder root is `<root>/train`
4. `build_train_transform_staged(...)`
5. `DataLoader(dataset, sampler=DistributedSampler(...), drop_last=True)`

Train sample format from `TrainDataset.__getitem__`:
- `image_train`: tensor `[3, H, W]` (normalized)
- `target`: class-name string (after remap via `self.classes[target]`)
- `path_train`: source image path

## 2.3 Test dataloader (both MVTec and VisA)

Call order:

1. `src.AD._evaluate_single_ckpt` (or `_demo` image scan path)
2. `build_dataloader(mode="test", root=data.test_root, classname=<one class>, datasetname=<mvtec|visa>)`
3. `TestDataset(source, classname, datasetname)`
4. `TestDataset.get_image_data()` builds list of `(classname, anomaly, image_path, mask_path)`

Dataset-specific branch in `get_image_data()`:
- MVTec normal folder token: `good`
- VisA normal folder token: `ok`
- Non-normal samples load `ground_truth/<anomaly>/*` masks; normal samples get zero mask

Test sample dict:
- `image`: `[3, H, W]`
- `mask`: `[1, H, W]`
- `is_anomaly`: `int(anomaly not in ("good", "ok"))`
- plus `classname`, `anomaly`, `image_name`, `image_path`

## 3) Foundation encoder and feature extraction

Main class: `tcfmad/src/tcfmad.py::VisionModule`

## 3.1 Encoder construction (`VisionModule._build_encoder`)

Supported encoders by `meta.model`:
- `dinov3` (default): `torch.hub.load("facebookresearch/dinov3", "dinov3_vitb16")`
- `dinov2`: `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")`
- `dino`
- `siglip`
- `clip`
- `dinosiglip`

Frozen vs trainable:
- For all non-`dinosiglip` models: encoder parameters are set `requires_grad=False`.
- Predictor (`vit_predictor`) is trainable.
- Optional `projector` exists for `dinosiglip` only.

## 3.2 Feature extraction path

- `target_features(images, paths, n_layer)` -> no-grad `_extract(...)`
- `context_features(images, paths, n_layer)` -> `_extract(...)` then predictor

For default `dinov3`:
- `_extract` calls `encoder.get_intermediate_layers(imgs, n=n_layer, return_class_token=False)[0]`
- output feature tensor `h`: shape `[B, P, D]`

`P` is patch count, `D` is encoder embed dim. With 512 crop and ViT-B/16, expected `P` is typically patch-grid sized (square), and later code assumes `sqrt(P)` is integer for heatmap reshape.

## 3.3 Predictor

- Built in `VisionModule.__init__` using `vit_predictor(...)`
- `predict(z)` returns same token layout `[B, P, D]`

## 4) Training pipeline (few-shot)

Call order:

1. `main.py mode=train` -> `src.train.main` -> `Trainer(args).train()`
2. For each batch:
   - load `imgs, labels, paths`
   - synthesize anomaly views: `CutPasteUnion(imgs, labels)` -> `imgs_abn`
   - compute target features: `h = model.target_features(imgs, paths, n_layer)`
   - compute context prediction:
     - branch A (50%): `p` from clean `imgs`
     - branch B (50%): `p` from augmented `imgs_abn`
   - loss:
     - `l2`: `MSE(h.flatten(0,1), p.flatten(0,1))`
     - or `smooth_l1`
   - optimize AdamW on predictor params
3. Save checkpoint every 100 global steps:
   - `train-step<step>.pth.tar`
4. Log to CSV:
   - `train.csv` in `logging.folder`

## 5) Inference / evaluation pipeline (`mode=AD`)

Call order:

1. `main.py mode=AD app=test app.ckpt_step=<N>`
2. Load saved train `params.yaml` and inject `meta`
3. Build ckpt path: `train-step<N>.pth.tar`
4. `src.AD.main` -> `_evaluate_single_ckpt(ckpt, cfg)`
5. For each class (dataset-specific class list):
   - build class test loader
   - for each sample:
     - `enc = model.target_features(img, paths, n_layer)` `[B,P,D]`
     - `pred = model.predict(enc)` `[B,P,D]`
     - patch error: `l = mse(enc, pred, reduction="none").mean(dim=2)` `[B,P]`
     - image anomaly score:
       - `topk = torch.topk(l, K, dim=1).values.mean(dim=1)`
       - `K = testing.K_top_mvtec` or `testing.K_top_visa`
     - pixel map:
       - reshape `l -> [B,1,h,w]` with `h=w=sqrt(P)`
       - bilinear upsample to input image size `[B,1,H,W]`
6. Class-level normalization and metrics:
   - image scores min-max normalized
   - pixel map min-max normalized
   - image metrics: `compute_imagewise_retrieval_metrics` (AUROC/AUPR/F1 stats)
   - pixel metrics: `compute_pixelwise_retrieval_metrics` (AUROC/AUPR/F1 stats)
   - PRO-AUC: `calculate_pro(..., max_steps, expect_fpr)`
7. Save outputs:
   - `<logging.folder>/AD_eval.csv`
   - optional segmentation figures under `<logging.folder>/segmentation/<class>/...` if `testing.segmentation_vis=True`

## 6) Demo pipeline (`mode=demo`)

Call order:

1. `main.py mode=demo app=test`
2. Load `pretrained.pth.tar`
3. `src.AD._demo(ckpt, cfg)`:
   - recursively scan all images under `data.test_root`
   - preprocess each to crop size + ImageNet normalization
   - run same `enc -> pred -> mse per patch` path
   - build heatmap, normalize per-image
   - overlay and save:
     - `<logging.folder>/heatmaps/<relative_subdir>/<stem>_heatmap.png`

## 7) Few-shot meaning in this repo

- Few-shot K (number of clean train images per class) is controlled by `sample.py num_samples` and folder contents.
- It is not a learnable prototype count in model code.
- `testing.K_top_mvtec/visa` is a scoring hyperparameter (top-K patch errors), not few-shot sample count.

## 8) What is trained vs frozen

Trained:
- `VisionModule.predictor` (always optimized in `init_opt(... predictor=...)`)

Frozen:
- Encoder for `dinov3/dinov2/dino/siglip/clip` (`requires_grad=False`)

Special case:
- `dinosiglip` creates a projector module; code marks it trainable, but optimizer is built only on predictor params.

## 9) Metrics and output artifacts

Computed metrics:
- image-level: AUROC, AUPR (+ F1 stats internally)
- pixel-level: AUROC, AUPR (+ F1 stats internally)
- localization: PRO-AUC

Saved artifacts:
- train: `train.csv`, `train-step*.pth.tar`, `params.yaml`
- eval: `AD_eval.csv`, optional segmentation grids
- demo: heatmap overlays
