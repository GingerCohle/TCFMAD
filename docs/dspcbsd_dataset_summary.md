# DsPCBSD Dataset Summary

## Goal
Summarize the structure, annotation format, and quick visualization sanity-check status of the COCO-format dataset at:

`/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/DsPCBSD`

## Dataset Root

```text
DsPCBSD/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
├── train2017/
└── val2017/
```

This is a standard COCO-style object detection layout:
- `annotations/instances_train.json`: train split annotations
- `annotations/instances_val.json`: val split annotations
- `train2017/`: training images
- `val2017/`: validation images

## Annotation Discovery

Auto-discovered COCO JSON candidates:
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/DsPCBSD/annotations/instances_train.json`
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/DsPCBSD/annotations/instances_val.json`

Preferred file for the quick visualization run:
- `ann_json used`: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/DsPCBSD/annotations/instances_val.json`

Inferred image root:
- `img_root used`: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/DsPCBSD/val2017`

Validation result:
- At least 5 images could be opened successfully from the inferred image root.

## Annotation Format

The annotation files follow the normal COCO detection schema:
- `images`: image metadata
- `annotations`: bounding boxes and per-instance metadata
- `categories`: category definitions

Bounding box format:
- `bbox = [x, y, w, h]`

Example image filenames referenced by the JSON:
- `Y_000900.jpg`
- `S_11463370.jpg`

These files live directly under `train2017/` or `val2017/`.

## Dataset Statistics

### Train Split
- annotation file: `instances_train.json`
- images: `8208`
- annotations: `16184`
- categories: `9`

### Val Split
- annotation file: `instances_val.json`
- images: `2051`
- annotations: `4092`
- categories: `9`

### Total
- images: `10259`
- annotations: `20276`
- categories: `9`

## Categories

The dataset contains 9 categories:
- `SP`
- `HB`
- `MB`
- `CS`
- `CFO`
- `BMFO`
- `OP`
- `SC`
- `SH`

Validation-set category counts from the quick scan:
- `SP`: `929`
- `HB`: `608`
- `MB`: `546`
- `CS`: `448`
- `CFO`: `423`
- `BMFO`: `346`
- `OP`: `338`
- `SC`: `285`
- `SH`: `169`

## Visualization Sanity Check

Visualization script:
- [tools/vis_coco_samples.py](/media/disk/kejunjie_only/dino_anomaly/TCFMAD/tools/vis_coco_samples.py)

Smoke test command:

```bash
cd /media/disk/kejunjie_only/dino_anomaly/TCFMAD
python tools/vis_coco_samples.py \
  --dataset_root /media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/DsPCBSD \
  --out_dir ./coco_vis_dspcbsd \
  --num 8 --seed 0 --only_images_with_boxes true
```

Smoke test result:
- images generated: `8`
- boxes visible: `yes`
- labels visible: `yes`
- gallery written: `coco_vis_dspcbsd/index.html`
- log written: `coco_vis_dspcbsd/run.log`

Run summary:
- dataset_root: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/DsPCBSD`
- ann_json used: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/DsPCBSD/annotations/instances_val.json`
- img_root used: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/DsPCBSD/val2017`
- total images scanned: `2051`
- images saved: `8`
- avg boxes per image: `1.75`

## Quick Takeaways

- This is a clean COCO-style PCB defect detection dataset.
- The dataset structure is directly compatible with standard COCO tooling.
- Visualization works without `pycocotools`; direct JSON parsing is sufficient.
- The current sample visualization pipeline is already validated on this dataset.
