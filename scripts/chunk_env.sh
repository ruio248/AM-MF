#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AM_MF_PYTHON="${AM_MF_PYTHON:-/home/lrh/AM-MF-4090/.venv/bin/python}"
export D4RL_DATASET_DIR="${D4RL_DATASET_DIR:-/home/lrh/d4rl_datasets}"
export AM_MF_RUNTIME_ROOT="${AM_MF_RUNTIME_ROOT:-/data/lrh/am-mf-experiments/chunk-size/runtime}"
exec bash "$project_root/scripts/experiment_env.sh" "$@"
