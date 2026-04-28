# TCFMAD Repo Map
Generated on: 2026-03-05

## Top-level tree

```text
/media/disk/kejunjie_only/dino_anomaly/TCFMAD
├── README.md
├── requirements.txt
├── setup.py
├── tcfmad/
│   ├── main.py
│   ├── configs/
│   │   ├── config.yaml
│   │   ├── app/train_dinov3.yaml
│   │   ├── app/test.yaml
│   │   └── sample_few_shot.yaml
│   └── src/
│       ├── train.py
│       ├── AD.py
│       ├── tcfmad.py
│       ├── sample.py
│       ├── datasets/dataset.py
│       └── utils/{metrics.py,synthesis.py,logging.py}
├── mvtec/                # full MVTec dataset root
├── visa/                 # full VisA dataset root
├── logs/                 # checkpoints/results (and released pretrained projectors)
└── assets/               # small demo assets
```

## README summary (what it says)

- Supported benchmarks: MVTec AD and VisA.
- Quick-start uses `python tcfmad/main.py mode=demo app=test ...`.
- Few-shot sampling uses `python tcfmad/src/sample.py`.
- Training uses `python tcfmad/main.py mode=train ... app=train_dinov3`.
- Inference/eval uses `python tcfmad/main.py mode=AD ... app=test app.ckpt_step=<step>`.
- README requires DINOv3 usage rights and released "trained manifold projectors" downloaded into `./logs/`.

## Required weights / checkpoints

From README + code behavior:

1. Foundation encoder weights
- Loaded in code by `torch.hub` / HuggingFace in `tcfmad/src/tcfmad.py` (`VisionModule._build_encoder`).
- If not cached, these may be downloaded on first run.

2. TCFMAD predictor/projector checkpoints
- `mode=demo` expects:
  - `logs/<data_name>/<model_name><diy_name>/params.yaml`
  - `logs/<data_name>/<model_name><diy_name>/pretrained.pth.tar`
- `mode=AD` expects:
  - `logs/<data_name>/<model_name><diy_name>/params.yaml`
  - `logs/<data_name>/<model_name><diy_name>/train-step<ckpt_step>.pth.tar`

## Key modules

- `tcfmad/main.py`: Hydra entrypoint + mode router (`train`, `AD`, `demo`) and DDP process spawn.
- `tcfmad/src/train.py`: training loop (`Trainer`), augmentation with CutPaste, checkpoint saving.
- `tcfmad/src/AD.py`: evaluation (`_evaluate_single_ckpt`) and demo heatmap export (`_demo`).
- `tcfmad/src/tcfmad.py`: `VisionModule`, encoder construction, feature extraction, predictor forward.
- `tcfmad/src/datasets/dataset.py`: train/test datasets + transforms + dataloader builder.
- `tcfmad/src/utils/metrics.py`: image AUROC/AUPR, pixel AUROC/AUPR, PRO-AUC.
- `tcfmad/src/sample.py`: few-shot subset creation script.

## Entrypoints

### Train / eval / infer entrypoints

1. Primary CLI entrypoint
- `tcfmad/main.py`
- Mode switch in `process_main(...)`:
  - `mode=train` -> `src.train.main(args=params)`
  - `mode=AD` -> `src.AD.main(args=params)`
  - `mode=demo` -> `src.AD._demo(ckpt_path, params)`

2. Few-shot data prep entrypoint
- `tcfmad/src/sample.py`

3. Internal modules (not standalone CLI parsers)
- `tcfmad/src/train.py`
- `tcfmad/src/AD.py`

## CLI args and defaults

Hydra-style overrides are used (`key=value`).

### `python tcfmad/main.py --help`

Main config defaults (`tcfmad/configs/config.yaml` + app config):

- `diy_name=_pretrained`
- `mode=train`
- `data.batch_size=8`
- `data.num_workers=0`
- `data.pin_mem=true`
- `data.dataset=dataset-name`
- `data.data_name=fewshot-folder-name`
- `data.data_path=/path/to/fewshot-folder`
- `data.train_root=${data.data_path}/${data.data_name}`
- `data.test_root=/path/to/dataset`
- `data.mvtec_classnames=[bottle,...,zipper]`
- `data.visa_classnames=[candle,...,pipe_fryum]`
- `data.use_hflip=true`
- `data.use_vflip=true`
- `data.use_rotate90=true`
- `data.use_color_jitter=true`
- `data.use_gray=true`
- `data.use_blur=true`
- `optimization.weight_decay=1e-4`
- `optimization.epochs=2000`
- `optimization.lr=0.001`
- `optimization.lr_config=const`
- `optimization.min_lr=1e-4`
- `optimization.warmup_epoch=100`
- `optimization.step_size=300`
- `optimization.gamma=0.1`
- `testing.K_top_mvtec=10`
- `testing.K_top_visa=6`
- `testing.expect_fpr=0.3`
- `testing.max_steps=200`
- `testing.segmentation_vis=true`
- `devices=[cuda:0]`
- `dist.master_addr=localhost`
- `dist.master_port=40112`
- `dist.backend=nccl`

App group options:
- `app=train_dinov3` (default)
- `app=test`

`app=train_dinov3` defaults:
- `app.meta.model=dinov3`
- `app.meta.crop_size=512`
- `app.meta.pred_depth=6`
- `app.meta.pred_emb_dim=384`
- `app.meta.use_bfloat16=false`
- `app.meta.if_pred_pe=false`
- `app.meta.n_layer=3`
- `app.meta.feat_normed=false`
- `app.meta.loss_mode=l2`
- `app.logging.folder=logs/${data.data_name}/${app.meta.model}${oc.select:diy_name,""}`
- `app.logging.write_tag=train`

`app=test` defaults:
- `app.model_name=dinov3`
- `app.ckpt_step=''`
- `app.logging.write_tag=AD`

### `python tcfmad/src/sample.py --help`

Defaults (`tcfmad/configs/sample_few_shot.yaml`):

- `source=/path/to/dataset`
- `target=/path/to/few_shot_folder`
- `num_samples=1`
- `train_subpaths=[train/good, train/ok]`
- `allowed_exts=[.png, .jpg, .jpeg]`
- `rename_images=false`
- `seed=42`
- `user_config=''`

### Hydra common flags (both CLIs)

- `--help/-h`
- `--hydra-help`
- `--cfg/-c [job|hydra|all]`
- `--resolve`
- `--run/-r`
- `--multirun/-m`
- `--config-path/-cp`
- `--config-name/-cn`
- `--config-dir/-cd`
- plus arbitrary `key=value` overrides

## Dataset folder expectations

### Full dataset root for testing/eval (`data.test_root`)

Expected by `TestDataset.get_image_data()`:

- MVTec class folder:
  - `<test_root>/<class>/test/good/*.png`
  - `<test_root>/<class>/test/<defect>/*.png`
  - `<test_root>/<class>/ground_truth/<defect>/*`
- VisA class folder:
  - `<test_root>/<class>/test/ok/*.JPG`
  - `<test_root>/<class>/test/ko/*.JPG` (or defect folders)
  - `<test_root>/<class>/ground_truth/<defect>/*`

### Few-shot train root (`data.train_root`)

Expected by `TrainDataset` (ImageFolder on `<train_root>/train`):

- `<train_root>/train/<class_name>/*`

This is exactly what `tcfmad/src/sample.py` produces.

## Notes

- `data.num_workers` exists in config/help but is not passed into `DataLoader` in `build_dataloader`.
- `mode=AD` and `mode=demo` both assert dataset keyword is contained in `data.test_root` path string (`"mvtec"` or `"visa"`).
