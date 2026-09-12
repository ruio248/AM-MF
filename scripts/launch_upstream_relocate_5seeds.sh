#!/usr/bin/env bash
set -euo pipefail

# GPU scheduler only: every experiment still executes the untouched upstream
# main_meanflowql.py through run_upstream_relocate.sh.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_root="${1:-$project_root/artifacts/relocate_b0_n}"
log_root="${2:-$project_root/logs/relocate_b0_n}"

mkdir -p "$artifact_root" "$log_root" \
  "$project_root/runtime/pycache" "$project_root/runtime/cache" \
  "$project_root/runtime/tmp"

export PATH="$project_root/.venv/bin:$PATH"
export D4RL_DATASET_DIR="${D4RL_DATASET_DIR:-$project_root/runtime/d4rl-datasets}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPYCACHEPREFIX="$project_root/runtime/pycache"
export XDG_CACHE_HOME="$project_root/runtime/cache"
export TMPDIR="$project_root/runtime/tmp"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:$HOME/.local/am-mf-mujoco-build-deps/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python -m unittest tests.test_upstream_integrity

if pgrep -f "$project_root.*main_meanflowql.py" >/dev/null; then
  echo "A formal run from this source tree is already active." >&2
  exit 1
fi

nohup bash -c "cd '$project_root' && CUDA_VISIBLE_DEVICES=0 bash scripts/run_upstream_relocate.sh b0 1 '$artifact_root/b0_seed001' && CUDA_VISIBLE_DEVICES=0 bash scripts/run_upstream_relocate.sh b0 5 '$artifact_root/b0_seed005'" >"$log_root/gpu0_b0_seed001_then_005.log" 2>&1 &
pid0=$!
nohup bash -c "cd '$project_root' && CUDA_VISIBLE_DEVICES=1 bash scripts/run_upstream_relocate.sh n 1 '$artifact_root/n_seed001' && CUDA_VISIBLE_DEVICES=1 bash scripts/run_upstream_relocate.sh n 5 '$artifact_root/n_seed005'" >"$log_root/gpu1_n_seed001_then_005.log" 2>&1 &
pid1=$!
nohup bash -c "cd '$project_root' && CUDA_VISIBLE_DEVICES=2 bash scripts/run_upstream_relocate.sh b0 2 '$artifact_root/b0_seed002'" >"$log_root/gpu2_b0_seed002.log" 2>&1 &
pid2=$!
nohup bash -c "cd '$project_root' && CUDA_VISIBLE_DEVICES=3 bash scripts/run_upstream_relocate.sh n 2 '$artifact_root/n_seed002'" >"$log_root/gpu3_n_seed002.log" 2>&1 &
pid3=$!
nohup bash -c "cd '$project_root' && CUDA_VISIBLE_DEVICES=4 bash scripts/run_upstream_relocate.sh b0 3 '$artifact_root/b0_seed003'" >"$log_root/gpu4_b0_seed003.log" 2>&1 &
pid4=$!
nohup bash -c "cd '$project_root' && CUDA_VISIBLE_DEVICES=5 bash scripts/run_upstream_relocate.sh n 3 '$artifact_root/n_seed003'" >"$log_root/gpu5_n_seed003.log" 2>&1 &
pid5=$!
nohup bash -c "cd '$project_root' && CUDA_VISIBLE_DEVICES=6 bash scripts/run_upstream_relocate.sh b0 4 '$artifact_root/b0_seed004'" >"$log_root/gpu6_b0_seed004.log" 2>&1 &
pid6=$!
nohup bash -c "cd '$project_root' && CUDA_VISIBLE_DEVICES=7 bash scripts/run_upstream_relocate.sh n 4 '$artifact_root/n_seed004'" >"$log_root/gpu7_n_seed004.log" 2>&1 &
pid7=$!

{
  printf "gpu\tpid\tqueue\n"
  printf "0\t%s\tb0_seed001,b0_seed005\n" "$pid0"
  printf "1\t%s\tn_seed001,n_seed005\n" "$pid1"
  printf "2\t%s\tb0_seed002\n" "$pid2"
  printf "3\t%s\tn_seed002\n" "$pid3"
  printf "4\t%s\tb0_seed003\n" "$pid4"
  printf "5\t%s\tn_seed003\n" "$pid5"
  printf "6\t%s\tb0_seed004\n" "$pid6"
  printf "7\t%s\tn_seed004\n" "$pid7"
} | tee "$log_root/workers.tsv"
