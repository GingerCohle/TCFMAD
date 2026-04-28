#!/usr/bin/env bash
set -euo pipefail

source ~/.bashrc || true
if [[ -f "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
  conda activate "${ENV_NAME:-tcfmad}" || true
fi

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WORKDIR}"

DATASET="${DATASET:-visa}"
DATA_NAME="${DATA_NAME:-visa_tmp}"
DIY_NAME="${DIY_NAME:-_2shot_0.5}"
MODEL_NAME="${MODEL_NAME:-dinov3}"
TEST_ROOT="${TEST_ROOT:-data/visa}"
TRAIN_DIR="${TRAIN_DIR:-logs/${DATA_NAME}/${MODEL_NAME}${DIY_NAME}}"
PARAMS_PATH="${TRAIN_DIR}/params.yaml"
LOG_FILE="${LOG_FILE:-logs/ad_${DATA_NAME}_fast.log}"

TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
TEST_NUM_WORKERS="${TEST_NUM_WORKERS:-4}"
SEGMENTATION_VIS="${SEGMENTATION_VIS:-false}"
TESTING_MULTI_LAYER_MEAN="${TESTING_MULTI_LAYER_MEAN:-true}"
TESTING_LAYERS="${TESTING_LAYERS:-}"
DEVICE_LIST="${DEVICE_LIST:-cuda:0}"
DIST_MASTER_PORT="${DIST_MASTER_PORT:-29501}"

mkdir -p "$(dirname "${LOG_FILE}")"
exec > >(tee "${LOG_FILE}") 2>&1

if [[ ! -d "${TRAIN_DIR}" ]]; then
  echo "Training output directory not found: ${TRAIN_DIR}" >&2
  exit 1
fi

if [[ ! -f "${PARAMS_PATH}" ]]; then
  echo "Training params not found: ${PARAMS_PATH}" >&2
  exit 1
fi

if [[ ! -d "${TEST_ROOT}" ]]; then
  echo "Test root not found: ${TEST_ROOT}" >&2
  exit 1
fi

if [[ -n "${CKPT_STEP_OVERRIDE:-}" ]]; then
  CKPT_STEP="${CKPT_STEP_OVERRIDE}"
else
  CKPT_STEP="$(
    find "${TRAIN_DIR}" -maxdepth 1 -type f -name 'train-step*.pth.tar' -printf '%f\n' \
      | sed -E 's/^train-step([0-9]+)\.pth\.tar$/\1/' \
      | sort -n \
      | tail -n 1
  )"
fi

if [[ -z "${CKPT_STEP}" ]]; then
  echo "No train-step checkpoint found under ${TRAIN_DIR}" >&2
  exit 1
fi

CKPT_PATH="${TRAIN_DIR}/train-step${CKPT_STEP}.pth.tar"
RESULTS_DIR="${TRAIN_DIR}/eval/${CKPT_STEP}"

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Resolved checkpoint does not exist: ${CKPT_PATH}" >&2
  exit 1
fi

CMD=(
  python tcfmad/main.py
  mode=AD
  app=test
  data.dataset="${DATASET}"
  data.data_name="${DATA_NAME}"
  diy_name="${DIY_NAME}"
  data.test_root="${TEST_ROOT}"
  app.ckpt_step="${CKPT_STEP}"
  "devices=[${DEVICE_LIST}]"
  "dist.master_port=${DIST_MASTER_PORT}"
  "testing.batch_size=${TEST_BATCH_SIZE}"
  "testing.num_workers=${TEST_NUM_WORKERS}"
  "testing.segmentation_vis=${SEGMENTATION_VIS}"
  "testing.multi_layer_mean=${TESTING_MULTI_LAYER_MEAN}"
)

if [[ -n "${TESTING_LAYERS}" ]]; then
  CMD+=("testing.layers=${TESTING_LAYERS}")
fi

echo "Selected ckpt_step: ${CKPT_STEP}"
echo "Resolved checkpoint: ${CKPT_PATH}"
echo "Resolved test_root: ${TEST_ROOT}"
echo "Device list: ${DEVICE_LIST}"
echo "Dist master port: ${DIST_MASTER_PORT}"
echo "Testing batch size: ${TEST_BATCH_SIZE}"
echo "Testing num_workers: ${TEST_NUM_WORKERS}"
echo "Segmentation visualization: ${SEGMENTATION_VIS}"
echo "Testing multi-layer mean: ${TESTING_MULTI_LAYER_MEAN}"
if [[ -n "${TESTING_LAYERS}" ]]; then
  echo "Testing layers override: ${TESTING_LAYERS}"
fi
echo "Results will be written under: ${RESULTS_DIR}"
echo "Log will be written to: ${LOG_FILE}"
printf 'Final command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}" "$@"
