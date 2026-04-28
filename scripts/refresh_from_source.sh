#!/usr/bin/env bash
set -euo pipefail

DEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-}"
STAGE_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "${STAGE_DIR}"
}
trap cleanup EXIT

printf 'Source: %s
' "${SOURCE_ROOT}"
printf 'Destination: %s
' "${DEST_ROOT}"

if [[ -z "${SOURCE_ROOT}" || ! -d "${SOURCE_ROOT}" ]]; then
  echo "Set SOURCE_ROOT=/path/to/source_repo when running this script." >&2
  exit 1
fi

rsync -a   --exclude '.git/'   --exclude '.idea/'   --exclude '.vscode/'   --exclude '__pycache__/'   --exclude '.ipynb_checkpoints/'   --exclude '*.pyc'   --exclude '*.pyo'   --exclude '*.pyd'   --exclude 'logs/'   --exclude 'outputs/'   --exclude 'visa_tmp/'   --exclude 'visa_tmp_bak/'   --exclude 'tcfmad/visa_tmp/'   --exclude '*.pth'   --exclude '*.pth.tar'   --exclude '*.pt'   --exclude '*.ckpt'   --exclude '*.onnx'   --exclude 'main.log'   --exclude 'sample.log'   --exclude 'tar_err.log'   --exclude 'train+val'   --exclude '.DS_Store'   --exclude '*.swp'   --exclude '*.swo'   "${SOURCE_ROOT}/" "${STAGE_DIR}/"

rsync -a --delete   --exclude '.git/'   --exclude '.gitignore'   --exclude 'README.md'   --exclude 'DATA_FORMAT.md'   --exclude 'docs/forgithub_snapshot.md'   --exclude 'docs/forgithub_report.md'   --exclude 'scripts/refresh_from_source.sh'   "${STAGE_DIR}/" "${DEST_ROOT}/"

if [[ ! -d "${DEST_ROOT}/tcfmad" ]]; then
  for candidate in "${DEST_ROOT}"/*; do
    [[ -d "${candidate}" ]] || continue
    [[ "$(basename "${candidate}")" == "tcfmad" ]] && continue
    if [[ -f "${candidate}/main.py" && -d "${candidate}/src" && -d "${candidate}/configs" ]]; then
      mv "${candidate}" "${DEST_ROOT}/tcfmad"
      break
    fi
  done
fi

if [[ ! -f "${DEST_ROOT}/tcfmad/src/tcfmad.py" && -d "${DEST_ROOT}/tcfmad/src" ]]; then
  candidate="$(python - "${DEST_ROOT}/tcfmad/src" <<'PY2'
from pathlib import Path
import sys
base = Path(sys.argv[1])
for p in sorted(base.glob('*.py')):
    if p.name in {'__init__.py', 'tcfmad.py'}:
        continue
    try:
        text = p.read_text(encoding='utf-8')
    except Exception:
        continue
    if 'class VisionModule' in text:
        print(p)
        break
PY2
)"
  if [[ -n "${candidate}" ]]; then
    mv "${candidate}" "${DEST_ROOT}/tcfmad/src/tcfmad.py"
  fi
fi

for suffix in repo_map profile_modules_report pipeline_trace run_commands; do
  current="$(find "${DEST_ROOT}/docs" -maxdepth 1 -type f -name "*${suffix}.md" | head -n 1)"
  target="${DEST_ROOT}/docs/tcfmad_${suffix}.md"
  if [[ -n "${current}" && "${current}" != "${target}" ]]; then
    mv "${current}" "${target}"
  fi
done

printf 'Refresh complete.
'
printf 'Managed files preserved: %s
' '.gitignore README.md DATA_FORMAT.md docs/forgithub_snapshot.md docs/forgithub_report.md scripts/refresh_from_source.sh'
