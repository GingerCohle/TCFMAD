# COCO Unlabeled Image Summary

## Dataset

- dataset root: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/DsPCBSD`
- train json: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/DsPCBSD/annotations/instances_train.json`
- val json: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/DsPCBSD/annotations/instances_val.json`
- train image root: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/DsPCBSD/train2017`
- val image root: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/datadet/deeppcb/DsPCBSD/val2017`

## Method

The scan logic is:
- collect all `image_id` values from `images`
- collect all `image_id` values referenced by `annotations`
- compute `unlabeled_ids = image_ids - annotated_ids`
- map unlabeled ids back to `file_name`
- verify whether the corresponding image file exists on disk

Script used:
- [coco_find_unlabeled_images.py](/media/disk/kejunjie_only/dino_anomaly/TCFMAD/tools/coco_find_unlabeled_images.py)

## Summary

| Split | Total images | Annotated images | Unlabeled images | File existence pass rate |
|---|---:|---:|---:|---:|
| train | 8208 | 8208 | 0 | 100.00% |
| val | 2051 | 2051 | 0 | 100.00% |

## Example Unlabeled Filenames

Train:
- none

Val:
- none

## Conclusion

No unlabeled images were found in either split.

This means:
- every image listed in `images` is referenced by at least one annotation
- there are currently no direct "negative-only" image candidates available from `images - annotations`

## Fallback Strategy

If you still need negative or normal regions for downstream experiments, a practical fallback is:
- generate `ok` crops from annotated images by sampling regions whose IoU is approximately `0` against all bounding boxes in the same image

This avoids changing the dataset itself while still producing candidate background or non-defect patches.

## Outputs

- log: [coco_find_unlabeled_images.log](/media/disk/kejunjie_only/dino_anomaly/TCFMAD/logs/coco_find_unlabeled_images.log)
- summary json: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/coco_unlabeled_report/summary.json`
- train list: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/coco_unlabeled_report/train_unlabeled.txt`
- val list: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/coco_unlabeled_report/val_unlabeled.txt`
