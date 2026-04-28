# PCBMarket Dataset Summary

## Goal

Summarize the structure, annotation format, split layout, class inventory, and small visual sanity-check results for:

`/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/pcbmarket`

## Dataset Root

- dataset root: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/pcbmarket`
- disk usage: `915.8MB` (`960258069` bytes)

## Directory Layout

```text
pcbmarket/
├── images/
│   ├── 01_missing_hole_01.jpg
│   ├── 01_mouse_bite_01.jpg
│   ├── 01_open_circuit_01.jpg
│   ├── ...
├── pcb_cocoanno/
│   ├── instances_train.json
│   └── instances_val.json
```

Notes:
- There is a single shared image directory: `images/`
- Train and val membership are defined by COCO JSON files, not by separate image folders
- No `test` split directory or `instances_test.json` file was found

## Format Detection

Detected dataset type:
- `COCO detection`

Why:
- found `instances_train.json` and `instances_val.json`
- both files contain standard COCO keys:
  - `images`
  - `annotations`
  - `categories`
- image filenames from JSON resolve under:
  `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/pcbmarket/images`

Conflicts checked:
- VOC XML: none
- VOC ImageSets: none
- segmentation mask folders/files: none
- custom `pkl/npz/csv`: none

## Annotation Files

- train json: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/pcbmarket/pcb_cocoanno/instances_train.json`
- val json: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/pcbmarket/pcb_cocoanno/instances_val.json`
- shared image root: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/pcbmarket/images`

COCO fields observed:
- image fields: `id`, `file_name`, `width`, `height`
- annotation fields: `id`, `image_id`, `category_id`, `bbox`, `area`, `iscrowd`
- bbox format: `[x, y, w, h]`

## Class Inventory

Number of classes: `6`

Classes:
- `missing_hole`
- `mouse_bite`
- `open_circuit`
- `short`
- `spur`
- `spurious_copper`

## Split Statistics

| Split | Images | Annotations | Annotated images | Avg boxes/image | Sample open success |
|---|---:|---:|---:|---:|---:|
| train | 555 | 2363 | 555 | 4.2577 | 5/5 |
| val | 138 | 590 | 138 | 4.2754 | 5/5 |

Additional notes:
- total image files found under `images/`: `693`
- `555 + 138 = 693`, so the two JSON splits fully cover the shared image pool
- every image in both splits has at least one annotation

## Image Size Summary

The dataset does not use a single fixed resolution. Both splits contain `10` distinct image sizes.

### Train image sizes
- width min / mean / max: `2240 / 2786.88 / 3056`
- height min / mean / max: `1586 / 2133.89 / 2530`
- top resolutions:
  - `3034 x 1586`: `99`
  - `3056 x 2464`: `95`
  - `2759 x 2154`: `53`
  - `2775 x 2159`: `53`
  - `2868 x 2316`: `50`
  - `2904 x 1921`: `48`
  - `2529 x 2530`: `46`
  - `2282 x 2248`: `45`
  - `2544 x 2156`: `43`
  - `2240 x 2016`: `23`

### Val image sizes
- width min / mean / max: `2240 / 2741.25 / 3056`
- height min / mean / max: `1586 / 2156.24 / 2530`
- top resolutions:
  - `3056 x 2464`: `25`
  - `3034 x 1586`: `21`
  - `2544 x 2156`: `17`
  - `2282 x 2248`: `15`
  - `2529 x 2530`: `14`
  - `2904 x 1921`: `12`
  - `2868 x 2316`: `11`
  - `2240 x 2016`: `9`
  - `2759 x 2154`: `7`
  - `2775 x 2159`: `7`

### Combined note
- The most common resolutions across the whole dataset are:
  - `3056 x 2464`: `120`
  - `3034 x 1586`: `120`
  - `2868 x 2316`: `61`
  - `2759 x 2154`: `60`
  - `2904 x 1921`: `60`
  - `2544 x 2156`: `60`
  - `2282 x 2248`: `60`
  - `2775 x 2159`: `60`
  - `2529 x 2530`: `60`
  - `2240 x 2016`: `32`

## Class Distribution

### Train
- `short`: `406`
- `missing_hole`: `400`
- `spurious_copper`: `398`
- `open_circuit`: `393`
- `mouse_bite`: `386`
- `spur`: `380`

### Val
- `spur`: `108`
- `mouse_bite`: `106`
- `spurious_copper`: `105`
- `missing_hole`: `97`
- `open_circuit`: `89`
- `short`: `85`

## Bounding Box Size Summary

Numbers below are in pixel units from COCO `bbox = [x, y, w, h]`.

### Combined per-class bbox sizes

| Class | Boxes | Width min/mean/max | Height min/mean/max | Area min/median/mean/max |
|---|---:|---:|---:|---:|
| missing_hole | 497 | 40 / 67.76 / 140 | 39 / 69.10 / 141 | 1764 / 4221 / 4836.18 / 19740 |
| mouse_bite | 492 | 26 / 64.07 / 123 | 27 / 62.43 / 121 | 858 / 3849 / 4009.72 / 9717 |
| open_circuit | 482 | 25 / 53.36 / 110 | 24 / 50.79 / 93 | 676 / 2534 / 2818.41 / 9114 |
| short | 491 | 34 / 87.74 / 243 | 35 / 79.75 / 174 | 1326 / 6241 / 7396.90 / 37665 |
| spur | 488 | 31 / 76.41 / 159 | 25 / 70.36 / 155 | 1054 / 5037.5 / 5267.02 / 15965 |
| spurious_copper | 503 | 30 / 81.21 / 284 | 26 / 78.48 / 216 | 1482 / 5355 / 5619.06 / 16500 |

### Interpretation
- `open_circuit` boxes are the smallest on average.
- `short` boxes are the largest on average, and also have the largest maximum area.
- `spurious_copper` has the widest max width range, up to `284` pixels.
- Most categories cluster in roughly `50-90` pixels mean width and height, so these are relatively small defects compared with the full image size.

## Visual Sanity Check

Sample size:
- up to `5` images per split
- fixed seed
- COCO boxes and class labels drawn onto images

Output directory:
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_vis`

Saved example visualizations:
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_vis/train_306_08_open_circuit_06.jpg`
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_vis/train_112_12_missing_hole_07.jpg`
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_vis/train_376_04_short_10.jpg`
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_vis/train_367_04_short_01.jpg`
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_vis/train_277_05_open_circuit_07.jpg`
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_vis/val_85_09_missing_hole_05.jpg`
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_vis/val_237_01_open_circuit_07.jpg`
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_vis/val_79_08_missing_hole_09.jpg`
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_vis/val_438_10_short_01.jpg`
- `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_vis/val_691_12_spurious_copper_08.jpg`

Result:
- boxes are present and render correctly
- labels are present and readable
- sampled images from both splits opened successfully

## Reproduce

One-liner:

```bash
cd /media/disk/kejunjie_only/dino_anomaly/TCFMAD
python tools/pcbmarket_audit.py --dataset_root /media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/pcbmarket --out_dir outputs/pcbmarket_vis
```

## Audit Outputs

- script: [pcbmarket_audit.py](/media/disk/kejunjie_only/dino_anomaly/TCFMAD/tools/pcbmarket_audit.py)
- log: [pcbmarket_audit.log](/media/disk/kejunjie_only/dino_anomaly/TCFMAD/logs/pcbmarket_audit.log)
- visualization dir: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_vis`
- structured summary: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/outputs/pcbmarket_vis/audit_summary.json`
