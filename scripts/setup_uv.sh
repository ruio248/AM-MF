#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-cpu}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found. Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

SYNC_ARGS=(--frozen)
VERIFY_ARGS=()

case "${PROFILE}" in
  cpu)
    ;;
  cuda)
    if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
      echo "The reproducible CUDA profile requires Linux x86_64." >&2
      exit 1
    fi
    SYNC_ARGS+=(--extra cuda12)
    VERIFY_ARGS+=(--require-gpu)
    ;;
  d4rl)
    SYNC_ARGS+=(--extra d4rl)
    VERIFY_ARGS+=(--check-d4rl)
    ;;
  full)
    if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
      echo "The full profile includes CUDA 12 and requires Linux x86_64." >&2
      exit 1
    fi
    SYNC_ARGS+=(--extra cuda12 --extra d4rl --extra toy)
    VERIFY_ARGS+=(--require-gpu --check-d4rl --check-toy)
    ;;
  toy)
    SYNC_ARGS+=(--extra toy)
    VERIFY_ARGS+=(--check-toy)
    ;;
  *)
    echo "Usage: $0 {cpu|cuda|d4rl|toy|full}" >&2
    exit 2
    ;;
esac

echo "Creating the '${PROFILE}' environment from uv.lock..."
uv sync "${SYNC_ARGS[@]}"

echo "Verifying the '${PROFILE}' environment..."
if [[ "${PROFILE}" == "cpu" ]]; then
  uv run --frozen --no-sync python scripts/verify_uv_environment.py
else
  uv run --frozen --no-sync python scripts/verify_uv_environment.py "${VERIFY_ARGS[@]}"
fi
