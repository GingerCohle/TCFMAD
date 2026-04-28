# TCFMAD Snapshot

## Source

- source working repo: provided externally via `SOURCE_ROOT` during refresh
- clean repo destination: this repository root
- snapshot method: read-only filesystem inspection with `ls`, `find`, `rg`, `file`, and size scans

## Preserved Structure

The clean repo preserves the project materials that are useful for code review and sharing:
- `tcfmad/`
- `scripts/`
- `tools/`
- `docs/`
- `assets/`
- `requirements.txt`
- `setup.py`
- `CODE_REVIEW_SUMMARY.md`
- `sample_segpatch.py`

## Excluded Content

The clean repo excludes local or non-source artifacts:
- VCS metadata and editor folders
- caches and compiled Python files
- logs and outputs
- checkpoints and large binary artifacts
- local sampled datasets such as `visa_tmp/` and `visa_tmp_bak/`
- local command notes such as `train+val`

## Project-Specific Ideas

This repo exposes two main method additions:
- defect-patch synthesis via `segpatch_folder`
- multi-layer consistency learning

## Refresh Workflow

The repo can be refreshed from an external working tree with:
- `scripts/refresh_from_source.sh`

The script is conservative and preserves destination-managed files such as:
- `README.md`
- `DATA_FORMAT.md`
- `docs/forgithub_snapshot.md`
- `docs/forgithub_report.md`
- `scripts/refresh_from_source.sh`
