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

DATA_NAME="${DATA_NAME:-visa_tmp}"
DIY_NAME="${DIY_NAME:-_2shot}"
MODEL_NAME="${MODEL_NAME:-dinov3}"
TRAIN_DIR="${TRAIN_DIR:-logs/${DATA_NAME}/${MODEL_NAME}${DIY_NAME}}"

START_STEP="${START_STEP:-4000}"
END_STEP="${END_STEP:-6000}"
STEP_STRIDE="${STEP_STRIDE:-}"
RANGE_LOG_DIR="${RANGE_LOG_DIR:-logs/test_${DATA_NAME}_range}"
DIST_MASTER_PORT_BASE="${DIST_MASTER_PORT_BASE:-29501}"

mkdir -p "${RANGE_LOG_DIR}"

if [[ ! -d "${TRAIN_DIR}" ]]; then
  echo "Training output directory not found: ${TRAIN_DIR}" >&2
  exit 1
fi

mapfile -t ALL_STEPS < <(
  find "${TRAIN_DIR}" -maxdepth 1 -type f -name 'train-step*.pth.tar' -printf '%f\n' \
    | sed -E 's/^train-step([0-9]+)\.pth\.tar$/\1/' \
    | sort -n
)

if [[ "${#ALL_STEPS[@]}" -eq 0 ]]; then
  echo "No train-step checkpoint found under ${TRAIN_DIR}" >&2
  exit 1
fi

SELECTED_STEPS=()
for step in "${ALL_STEPS[@]}"; do
  if (( step < START_STEP || step > END_STEP )); then
    continue
  fi
  if [[ -n "${STEP_STRIDE}" ]] && (( (step - START_STEP) % STEP_STRIDE != 0 )); then
    continue
  fi
  SELECTED_STEPS+=("${step}")
done

if [[ "${#SELECTED_STEPS[@]}" -eq 0 ]]; then
  echo "No checkpoints matched range ${START_STEP}-${END_STEP} under ${TRAIN_DIR}" >&2
  exit 1
fi

echo "Training directory: ${TRAIN_DIR}"
echo "Selected checkpoint steps: ${SELECTED_STEPS[*]}"
echo "Per-step logs will be written to: ${RANGE_LOG_DIR}"

run_idx=0
for step in "${SELECTED_STEPS[@]}"; do
  port=$((DIST_MASTER_PORT_BASE + run_idx))
  log_file="${RANGE_LOG_DIR}/step${step}.log"

  echo "===== evaluating step ${step} ====="
  echo "log_file=${log_file}"
  echo "dist_master_port=${port}"

  LOG_FILE="${log_file}" \
  CKPT_STEP_OVERRIDE="${step}" \
  DIST_MASTER_PORT="${port}" \
  bash scripts/test_visa_tmp_2shot_fast.sh "$@"

  run_idx=$((run_idx + 1))
done
