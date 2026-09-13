#!/usr/bin/env bash
# Platform-only launcher for the frozen upstream MeanFlowQL runner.
# It intentionally contains no task, method, or training-budget decisions.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${AM_MF_PYTHON:-}" ]]; then
  project_python="$AM_MF_PYTHON"
elif [[ -x "$project_root/.venv/bin/python" ]]; then
  project_python="$project_root/.venv/bin/python"
else
  echo "Set AM_MF_PYTHON or create $project_root/.venv." >&2
  exit 1
fi

if [[ ! -x "$project_python" ]]; then
  echo "AM_MF_PYTHON is not executable: $project_python" >&2
  exit 1
fi

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

runtime_root="${AM_MF_RUNTIME_ROOT:-$project_root/runtime}"
export PYTHONPYCACHEPREFIX="$runtime_root/pycache"
export XDG_CACHE_HOME="$runtime_root/cache"
export JAX_COMPILATION_CACHE_DIR="$runtime_root/jax-cache"
export TMPDIR="$runtime_root/tmp"

if [[ -z "${MUJOCO_PY_MUJOCO_PATH:-}" && -d "$HOME/.mujoco/mujoco210" ]]; then
  export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
fi

library_paths=()
if [[ -n "${MUJOCO_PY_MUJOCO_PATH:-}" && -d "$MUJOCO_PY_MUJOCO_PATH/bin" ]]; then
  library_paths+=("$MUJOCO_PY_MUJOCO_PATH/bin")
fi

# A user-local GLVND/EGL runtime is needed on some shared container images.
egl_prefix="${AM_MF_EGL_PREFIX:-$HOME/.local/am-mf-egl}"
if [[ -d "$egl_prefix/usr/lib/x86_64-linux-gnu" ]]; then
  library_paths+=("$egl_prefix/usr/lib/x86_64-linux-gnu")
fi

mujoco_build_deps="${AM_MF_MUJOCO_BUILD_DEPS:-$HOME/.local/am-mf-mujoco-build-deps}"
if [[ -d "$mujoco_build_deps/usr/bin" ]]; then
  export PATH="$mujoco_build_deps/usr/bin:$PATH"
fi
if [[ -d "$mujoco_build_deps/usr/include" ]]; then
  export CPATH="$mujoco_build_deps/usr/include${CPATH:+:$CPATH}"
fi
if [[ -d "$mujoco_build_deps/usr/lib/x86_64-linux-gnu" ]]; then
  export LIBRARY_PATH="$mujoco_build_deps/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}"
  library_paths+=("$mujoco_build_deps/usr/lib/x86_64-linux-gnu")
fi

for system_library_path in /usr/lib/x86_64-linux-gnu /usr/lib/nvidia; do
  if [[ -d "$system_library_path" ]]; then
    library_paths+=("$system_library_path")
  fi
done

if [[ ${#library_paths[@]} -gt 0 ]]; then
  joined_library_paths="$(IFS=:; printf '%s' "${library_paths[*]}")"
  export LD_LIBRARY_PATH="$joined_library_paths${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$PYTHONPYCACHEPREFIX" "$XDG_CACHE_HOME" "$TMPDIR" "$JAX_COMPILATION_CACHE_DIR"
cd "$project_root"
exec "$project_python" -u "$@"
