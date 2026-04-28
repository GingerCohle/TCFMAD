# TCFMAD Report

## Result

This repository is a cleaned, shareable copy prepared for GitHub publishing.

## What Was Kept

- `tcfmad/` package and configs
- shell helpers under `scripts/`
- utilities under `tools/`
- project documentation under `docs/`
- top-level `README.md` and `DATA_FORMAT.md`

## What Was Removed

- `.git/`, `.idea/`, `.vscode/`
- `__pycache__/`, `.pyc`, notebook checkpoints
- `logs/`, `outputs/`
- dataset folders such as `visa_tmp/`, `visa_tmp_bak/`, and nested cached data folders
- checkpoints such as `*.pth`, `*.pth.tar`, `*.pt`, `*.ckpt`, `*.onnx`
- local notes file `train+val`

## Package Rename

The clean repo was fully rebranded from the previous package naming to `tcfmad`:
- top-level package directory renamed to `tcfmad/`
- core module renamed to `tcfmad/src/tcfmad.py`
- script entrypoints updated to `python tcfmad/main.py`
- setup metadata updated to `tcfmad`
- outward-facing docs renamed to `tcfmad_*` where applicable

## Validation

Static validation completed:
- shell scripts pass `bash -n`
- key Python files pass `python -m py_compile`
- no `tcfmad` / `TCFMAD` strings remain in the repository
- no `__pycache__` or `.pyc` files remain

## Manual Review Notes

Some inherited docs and utility defaults still include workstation-specific absolute paths. Those are now rebranded, but they are still local-path assumptions and should be reviewed before public release.

## Refresh Script

Use:
```bash
SOURCE_ROOT=/path/to/source_repo bash scripts/refresh_from_source.sh
```
