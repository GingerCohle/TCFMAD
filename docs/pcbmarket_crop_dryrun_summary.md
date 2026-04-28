# PCBMarket Crop Dry-Run Summary

## Goal

Run a small VisA-like dry-run on PCBMarket before full conversion:
- sample a small subset from `train` and `val`
- generate defect-present `KO` square crops around bbox centers
- generate defect-free `OK` square crops from the same source image with `IoU <= 0`
- save overlays and per-image panels for visual inspection

## Inputs

- dataset root: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/pcbmarket`
- train json: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/pcbmarket/pcb_cocoanno/instances_train.json`
- val json: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/pcbmarket/pcb_cocoanno/instances_val.json`
- train image root: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/pcbmarket/images`
- val image root: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/pcbmarket/images`

Candidate discovery:
- json candidates:
  - `pcb_cocoanno/instances_train.json`
  - `pcb_cocoanno/instances_val.json`
- image dir candidates:
  - `images/`

## Dry-Run Settings

Command used:

```bash
cd /media/disk/kejunjie_only/dino_anomaly/TCFMAD
python tools/pcbmarket_crop_dryrun.py \
  --dataset_root /media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/pcbmarket \
  --split both \
  --num_images_train 20 \
  --num_images_val 10 \
  --seed 0 \
  --alpha 8.0 --min_side 384 --max_side 768 \
  --ok_per_ko 1 --iou_thr 0.0 --max_try 50 \
  --out_dir outputs/pcbmarket_dryrun
```

Crop policy:
- `KO side = clip(alpha * max(w, h), min_side, max_side)`
- `alpha = 8.0`
- `min_side = 384`
- `max_side = 768`
- `OK` uses the same side length as the paired `KO`
- `OK` acceptance rule: `IoU(crop, any bbox) <= 0.0`

## Output Layout

- output root: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_dryrun`
- log: [pcbmarket_crop_dryrun.log](/media/disk/kejunjie_only/dino_anomaly/TCFMAD/logs/pcbmarket_crop_dryrun.log)
- structured summary: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_dryrun/dryrun_summary.json`

Directory structure:

```text
outputs/pcbmarket_dryrun/
├── train/
│   ├── ko/
│   ├── ok/
│   └── panels/
└── val/
    ├── ko/
    ├── ok/
    └── panels/
```

Total generated images:
- `290`

This stays under the requested `<= 300` image budget.

## Split Summary

| Split | Sampled source images | KO crops | OK crops | Panels | KO bbox/crop area mean | Clamp fraction | OK try mean | OK failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 20 | 86 | 86 | 20 | 0.0125 | 0.2558 | 2.02 | 0 |
| val | 10 | 44 | 44 | 10 | 0.0124 | 0.2045 | 2.14 | 0 |

Additional notes:
- `KO bbox/crop area` is small, around `1.24% - 1.25%`, which means the crop keeps a lot of context around the defect.
- Around `20% - 26%` of KO crops needed boundary clamping.
- `OK` crops were easy to find in this dry-run; average tries stayed close to `2`, with zero failures.

## Sampled Source Images

Examples from train:
- `04_spur_09.jpg`
- `08_spur_06.jpg`
- `06_missing_hole_04.jpg`

Examples from val:
- `10_spur_04.jpg`
- `04_mouse_bite_06.jpg`
- `05_spur_10.jpg`

## Example Outputs

KO examples:
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_dryrun/train/ko/train_img491_ann978_spur_side688_ko.jpg`
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_dryrun/train/ko/train_img538_ann66_spur_side528_ko.jpg`
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_dryrun/val/ko/val_img556_ann386_spur_side768_ko.jpg`

OK examples:
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_dryrun/train/ok/train_img491_ok0_0_side688.jpg`
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_dryrun/train/ok/train_img538_ok0_0_side528.jpg`
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_dryrun/val/ok/val_img556_ok0_0_side768.jpg`

Panel examples:
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_dryrun/train/panels/train_img121_panel.jpg`
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_dryrun/train/panels/train_img181_panel.jpg`
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_dryrun/val/panels/val_img556_panel.jpg`

## Visual Inspection Notes

Spot-check result from sampled panels:
- `original` view shows all source-image boxes
- `KO` crop correctly centers on a selected defect bbox and overlays a white rectangle mask plus green bbox
- `OK` crop appears free of defect intersections under the `IoU == 0` rule

One inspected panel:
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_dryrun/train/panels/train_img121_panel.jpg`

Observed behavior:
- `KO` crop keeps generous PCB context around the defect
- `OK` crop looks visually clean and does not show intersecting defect boxes

## Reproduce

Script:
- [pcbmarket_crop_dryrun.py](/media/disk/kejunjie_only/dino_anomaly/TCFMAD/tools/pcbmarket_crop_dryrun.py)

Run:

```bash
cd /media/disk/kejunjie_only/dino_anomaly/TCFMAD
python tools/pcbmarket_crop_dryrun.py \
  --dataset_root /media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/pcbmarket \
  --split both \
  --num_images_train 20 \
  --num_images_val 10 \
  --seed 0 \
  --alpha 8.0 --min_side 384 --max_side 768 \
  --ok_per_ko 1 --iou_thr 0.0 --max_try 50 \
  --out_dir outputs/pcbmarket_dryrun
```
