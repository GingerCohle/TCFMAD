# PCBMarket to VisA Summary

## Goal

Convert the PCBMarket COCO dataset into a VisA-style anomaly detection dataset at:

`/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/pcbmarket_visa`

Target structure per class:

```text
<class_name>/
  train/ok/
  test/ok/
  test/ko/
  ground_truth/ko/
```

## Source and Output Paths

- source root: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/pcbmarket`
- output root: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/pcbmarket_visa`
- verify visuals: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_visa_verify`
- log: [pcbmarket_to_visa.log](/media/disk/kejunjie_only/dino_anomaly/TCFMAD/logs/pcbmarket_to_visa.log)
- converter: [pcbmarket_coco_to_visa.py](/media/disk/kejunjie_only/dino_anomaly/TCFMAD/tools/pcbmarket_coco_to_visa.py)

## Source Layout Verification

The source dataset does **not** match the expected `annotations/ + train2017/val2017/` layout.

Expected paths checked:
- `annotations/instances_train.json`: missing
- `annotations/instances_val.json`: missing
- `train2017/`: missing
- `val2017/`: missing

Actual discovered source layout:
- train json: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/pcbmarket/pcb_cocoanno/instances_train.json`
- val json: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/pcbmarket/pcb_cocoanno/instances_val.json`
- shared image root for both splits:
  `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/pcbmarket/images`

## Categories

Classes used for VisA-style output:
- `missing_hole`
- `mouse_bite`
- `open_circuit`
- `short`
- `spur`
- `spurious_copper`

## One-Category-Per-Image Check

Result:
- train violations: `0`
- val violations: `0`

Interpretation:
- every source image contains boxes from exactly one category
- the fallback multi-category routing logic was implemented, but it was not needed for this dataset

Violation examples:
- none

## Conversion Settings

Command used:

```bash
cd /media/disk/kejunjie_only/dino_anomaly/TCFMAD
python tools/pcbmarket_coco_to_visa.py \
  --src_root /media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/pcbmarket \
  --out_root /media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/pcbmarket_visa \
  --alpha 8.0 --min_side 384 --max_side 768 --resize 512 \
  --ok_per_ko 1 --iou_thr 0.0 --max_try 50 \
  --max_anns_per_image 6 --seed 0 \
  --write_verify true --verify_max 50
```

Policy summary:
- `KO` crop: square crop around bbox center, side = `clip(8.0 * max(w, h), 384, 768)`
- `KO` crop resized to `512 x 512`
- `KO` mask: projected bbox rectangle, binary `0/255`, resized with nearest-neighbor
- `OK` crop: same side as paired `KO`
- `OK` acceptance: `IoU(crop, any bbox) <= 0.0`

## Final Output Counts Per Class

| Class | train/ok | test/ok | test/ko | ground_truth/ko |
|---|---:|---:|---:|---:|
| missing_hole | 400 | 97 | 97 | 97 |
| mouse_bite | 386 | 106 | 106 | 106 |
| open_circuit | 393 | 89 | 89 | 89 |
| short | 406 | 85 | 85 | 85 |
| spur | 380 | 108 | 108 | 108 |
| spurious_copper | 398 | 105 | 105 | 105 |

Totals:
- `train/ok`: `2363`
- `test/ok`: `590`
- `test/ko`: `590`
- `ground_truth/ko`: `590`

File count note:
- dataset files under `pcbmarket_visa`: `4134`
- this includes `4133` data files plus `_conversion_summary.json`

## Crop and Sampling Statistics

Aggregate crop anchors processed:
- `2953`

BBox-to-crop area ratio:
- count: `2953`
- mean: `0.0125`
- min: `0.0042`
- max: `0.0639`

Interpretation:
- on average, the defect bbox occupies about `1.25%` of the square crop area
- crops keep substantial visual context around the defect

OK sampling:
- sampled OK crops: `2953`
- average tries per OK crop: `2.13`
- min tries: `1`
- max tries: `32`
- OK failure count: `0`

Boundary clamping:
- clamped crop-anchor fraction: `0.1422`

Missing source images:
- `0`

## Verification Visuals

Verification output folder:
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_visa_verify`

Number of debug panels:
- `50`

These panels are now val-driven, so each one includes:
- original image with all source bboxes
- one KO crop overlay
- one OK crop overlay

Example verify panels:
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_visa_verify/val_img100_panel.jpg`
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_visa_verify/val_img110_panel.jpg`
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_visa_verify/val_img113_panel.jpg`

## Output Structure Check

Per-class directories were created successfully:
- `missing_hole`
- `mouse_bite`
- `open_circuit`
- `short`
- `spur`
- `spurious_copper`

Each class now contains:
- `train/ok/`
- `test/ok/`
- `test/ko/`
- `ground_truth/ko/`

## Notes

- The source dataset was left unchanged.
- No full-dataset intermediate copy was created.
- The conversion is reproducible under the same seed and parameters.
