# TCFMAD Data Format

This document summarizes the dataset layout expected by the current loaders in this repository.

## 1. Training Data Root

`data.train_root` should point to a directory that contains a `train/` folder.

Expected structure:

```text
<data.train_root>/
└── train/
    ├── <class_1>/
    │   ├── normal_image_001.jpg
    │   ├── normal_image_002.png
    │   └── seg_patch/
    │       ├── 000.png
    │       └── 001.png
    ├── <class_2>/
    │   ├── normal_image_001.jpg
    │   └── seg_patch/
    │       └── 000.png
    └── ...
```

Requirements:
- normal few-shot training images are placed directly under each class folder
- optional defect patches for synthesis are placed under `train/<class>/seg_patch/`
- class names are inferred from the folder names under `train/`

## 2. Segmentation Patch Folder

When `synthesis.mode=segpatch_folder` is enabled, the trainer looks for defect patches here:

```text
train/<class>/seg_patch/*
```

Assumptions:
- patches are sampled from the same class as the current normal training image
- one patch is pasted per synthesized abnormal image in the default workflow
- if a class has no `seg_patch/`, synthesis for that class falls back to no paste

Typical VisA-style example:

```text
visa_tmp/
└── train/
    └── candle/
        ├── 0001.JPG
        ├── 0002.JPG
        └── seg_patch/
            ├── 000.png
            └── 001.png
```

## 3. Test / Evaluation Data Root

`data.test_root` should point to the dataset root that contains class folders.

For each class, the loader expects:

```text
<data.test_root>/
└── <class>/
    ├── test/
    │   ├── <anomaly_type_1>/
    │   │   ├── image_001.png
    │   │   └── image_002.png
    │   ├── <anomaly_type_2>/
    │   └── <normal_folder>/
    │       ├── image_001.png
    │       └── image_002.png
    └── ground_truth/
        ├── <anomaly_type_1>/
        │   ├── image_001.png
        │   └── image_002.png
        └── <anomaly_type_2>/
```

The normal folder name depends on `data.dataset`:
- `mvtec`: normal test folder must be `good`
- `visa`: normal test folder must be `ok`

For the normal test folder:
- images exist under `test/good/` or `test/ok/`
- no mask files are required

For anomalous test folders:
- images exist under `test/<anomaly_type>/`
- masks exist under `ground_truth/<anomaly_type>/`
- image and mask file ordering is matched by sorted filename order

## 4. Loader Mapping

Current config-to-path mapping is:
- `data.train_root` -> used by the training loader, which reads from `<data.train_root>/train/...`
- `data.test_root` -> used by the test loader, which reads from `<data.test_root>/<class>/test/...`
- `data.dataset` -> controls whether the normal test folder is `good` or `ok`

## 5. Minimal Examples

### Training root example

```text
visa_tmp/
└── train/
    ├── candle/
    │   ├── 0001.JPG
    │   ├── 0002.JPG
    │   └── seg_patch/
    │       ├── 000.png
    │       └── 001.png
    └── capsules/
        ├── 0001.JPG
        └── seg_patch/
            └── 000.png
```

### Test root example for `data.dataset=visa`

```text
visa/
└── candle/
    ├── test/
    │   ├── ok/
    │   │   ├── 0001.JPG
    │   │   └── 0002.JPG
    │   └── crack/
    │       ├── 0101.JPG
    │       └── 0102.JPG
    └── ground_truth/
        └── crack/
            ├── 0101.png
            └── 0102.png
```

## 6. Practical Notes

- keep normal training images directly under each class folder
- keep `seg_patch/` names consistent across classes
- use matching sorted filenames between anomalous test images and masks
- avoid mixing unrelated helper files into `train/<class>/` and `test/<anomaly_type>/`
- if you prepare another dataset, align the normal test folder naming with `data.dataset`

## 7. Related Docs

- `README.md`
- `docs/segpatch_folder_synthesis.md`
- `docs/tcfmad_pipeline_trace.md`
