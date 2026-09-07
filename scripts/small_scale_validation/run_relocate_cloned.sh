#!/usr/bin/env bash
# Launch the MeanFlowQL paper-style baseline for D4RL Relocate Cloned.
#
# Usage:
#   bash scripts/small_scale_validation/run_relocate_cloned.sh offline [seed]
#   bash scripts/small_scale_validation/run_relocate_cloned.sh online  [seed]
#
# Appendix E.2 fixes alpha=10000, C5, and time_steps=50 for this task.
# The D4RL offline-only budget and the paper's online-table budget are exposed
# as environment variables because the paper describes them in two places.
set -euo pipefail

stage="${1:-}"
seed="${2:-1}"

if [[ "${stage}" != "offline" && "${stage}" != "online" ]]; then
  echo "Usage: $0 {offline|online} [seed]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

# A strategy can be changed without editing this script, for example:
# AGENT_PATH=agents/am_meanflow_target_changed.py \
# EXTRA_AGENT_FLAGS='--agent.adjoint_eta=0.1' \
# bash scripts/small_scale_validation/run_relocate_cloned.sh online 1
agent_path="${AGENT_PATH:-agents/meanflowql.py}"
alpha="${ALPHA:-10000}"
num_candidates="${NUM_CANDIDATES:-5}"
time_steps="${TIME_STEPS:-50}"

if [[ "${stage}" == "offline" ]]; then
  # Appendix D says D4RL offline training uses 500K gradient steps.
  offline_steps="${OFFLINE_STEPS:-500000}"
  online_steps=0
else
  # Table 2 reports the online protocol at 1M and 2M total gradient steps.
  offline_steps="${OFFLINE_STEPS:-1000000}"
  online_steps="${ONLINE_STEPS:-1000000}"
fi

extra_agent_flags=()
if [[ -n "${EXTRA_AGENT_FLAGS:-}" ]]; then
  # Intended for trusted local launch arguments such as --agent.adjoint_eta=0.1.
  read -r -a extra_agent_flags <<< "${EXTRA_AGENT_FLAGS}"
fi

agent_args=(
  --agent="${agent_path}"
  --agent.discount=0.99
  --agent.num_candidates="${num_candidates}"
)
case "$(basename "${agent_path}")" in
  meanflowql.py|am_meanflow_target_changed.py)
    # These two agents implement the reformulated MeanFlowQL target.
    agent_args+=(--agent.alpha="${alpha}" --agent.time_steps="${time_steps}")
    ;;
  native_meanflow.py|am_meanflow.py)
    # Native MeanFlow and the original AM target use their own loss controls;
    # alpha/time_steps are deliberately not passed because they are not config keys.
    ;;
  *)
    echo "Unsupported AGENT_PATH: ${agent_path}" >&2
    echo "Use one of agents/{meanflowql,native_meanflow,am_meanflow,am_meanflow_target_changed}.py" >&2
    exit 2
    ;;
esac

cmd=(
  python main_meanflowql.py
  --env_name=relocate-cloned-v1
  --seed="${seed}"
  --offline_steps="${offline_steps}"
  --online_steps="${online_steps}"
  --balanced_sampling=0
  --use_observation_normalization=True
  --eval_episodes="${EVAL_EPISODES:-50}"
  --eval_interval="${EVAL_INTERVAL:-100000}"
  --save_interval="${SAVE_INTERVAL:-100000}"
  --enable_early_stopping=False
  --proj_wandb="${PROJ_WANDB:-meanflowql_small_scale}"
  --run_group="${RUN_GROUP:-relocate_cloned_${stage}}"
  --wandb_save_dir="${WANDB_SAVE_DIR:-wandb_offline}"
  --wandb_online="${WANDB_ONLINE:-False}"
)

cmd+=("${agent_args[@]}")
cmd+=("${extra_agent_flags[@]}")
printf 'Launching: '
printf '%q ' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
