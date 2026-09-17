#!/usr/bin/env bash
# Thin launcher only: all optimisation and evaluation remain in the frozen
# upstream main_meanflowql.py.  Do not add task-specific training loops here.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 {b0|n|n_offline_gated|n_path_local} {humanoid_large_task1|relocate_cloned|pen_cloned_v1|hammer_cloned_v1} SEED OUTPUT_ROOT" >&2
  exit 2
fi

arm="$1"
task_profile="$2"
seed="$3"
output_root="$4"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$arm" in
  b0) agent="agents/meanflowql.py" ;;
  n|n_offline_gated|n_path_local) agent="agents/am_meanflow_note.py" ;;
  *)
    echo "arm must be b0, n, n_offline_gated, or n_path_local" >&2
    exit 2
    ;;
esac

case "$task_profile" in
  humanoid_large_task1)
    env_name="humanoidmaze-large-navigate-singletask-task1-v0"
    alpha="6000"
    discount="0.995"
    time_steps="50"
    metric="evaluation/success"
    project_name="meanflowql_upstream_humanoid_large"
    ;;
  relocate_cloned)
    env_name="relocate-cloned-v1"
    alpha="10000"
    discount="0.99"
    time_steps="50"
    metric="evaluation/episode.normalized_return"
    project_name="meanflowql_upstream_relocate"
    ;;
  # D4RL Adroit profiles.  alpha, num_candidates and time_steps follow the
  # per-task values in MeanFlowQL Appendix Table 7; the D4RL discount is the
  # default 0.99 and the offline-to-online budget is 1M offline + 1M online as
  # described in Appendix D and reported at 1M and 2M gradient steps.
  pen_cloned_v1)
    env_name="pen-cloned-v1"
    alpha="10000"
    discount="0.99"
    time_steps="100"
    metric="evaluation/episode.normalized_return"
    project_name="meanflowql_upstream_pen"
    ;;
  hammer_cloned_v1)
    env_name="hammer-cloned-v1"
    alpha="11000"
    discount="0.99"
    time_steps="100"
    metric="evaluation/episode.normalized_return"
    project_name="meanflowql_upstream_hammer"
    ;;
  *)
    echo "unknown task profile: $task_profile" >&2
    exit 2
    ;;
esac

offline_steps="${AM_MF_OFFLINE_STEPS:-1000000}"
online_steps="${AM_MF_ONLINE_STEPS:-1000000}"
eval_interval="${AM_MF_EVAL_INTERVAL:-100000}"
eval_episodes="${AM_MF_EVAL_EPISODES:-50}"
save_interval="${AM_MF_SAVE_INTERVAL:-100000}"
buffer_size="${AM_MF_BUFFER_SIZE:-2000000}"
log_interval="${AM_MF_LOG_INTERVAL:-5000}"
run_group="${arm}_${task_profile}_seed${seed}"
# The formal N protocol uses the agent default (500k updates).  A short smoke
# run can lower this *only* through an explicit environment override so that
# the actual adjoint-control branch, rather than just behavior initialization,
# is exercised before formal jobs are launched.
agent_extra_flags=()
if [[ ( "$arm" == "n" || "$arm" == "n_offline_gated" || "$arm" == "n_path_local" ) && -n "${AM_MF_BEHAVIOR_WARMUP_UPDATES:-}" ]]; then
  agent_extra_flags+=("--agent.behavior_warmup_updates=${AM_MF_BEHAVIOR_WARMUP_UPDATES}")
fi
if [[ "$arm" == "n_path_local" ]]; then
  agent_extra_flags+=(
    "--agent.transport_target_mode=path_local"
    "--agent.control_eta=0.1"
    "--agent.control_eta_ramp_updates=0"
    "--agent.control_adjoint_clip=0"
    "--agent.control_uncertainty_scale=0"
    "--agent.control_uncertainty_end_update=-1"
  )
fi
if [[ "$arm" == "n_offline_gated" ]]; then
  # Conservative offline-AM ablation. The original N default remains exact
  # unless this explicitly selected arm is used.
  agent_extra_flags+=(
    "--agent.control_eta_ramp_updates=${AM_MF_CONTROL_ETA_RAMP_UPDATES:-500000}"
    "--agent.control_adjoint_clip=${AM_MF_CONTROL_ADJOINT_CLIP:-1.0}"
    "--agent.control_uncertainty_scale=${AM_MF_CONTROL_UNCERTAINTY_SCALE:-0.25}"
    "--agent.control_uncertainty_end_update=${AM_MF_CONTROL_UNCERTAINTY_END_UPDATE:-$offline_steps}"
  )
fi

mkdir -p "$output_root"
{
  printf 'git_commit='; git -C "$project_root" rev-parse HEAD
  printf 'git_status='; git -C "$project_root" status --short
  printf 'arm=%s\n' "$arm"
  printf 'task_profile=%s\n' "$task_profile"
  printf 'env_name=%s\n' "$env_name"
  printf 'seed=%s\n' "$seed"
  printf 'offline_steps=%s\n' "$offline_steps"
  printf 'online_steps=%s\n' "$online_steps"
  printf 'alpha=%s\n' "$alpha"
  printf 'discount=%s\n' "$discount"
  printf 'time_steps=%s\n' "$time_steps"
  printf 'num_candidates=5\n'
  printf 'log_interval=%s\n' "$log_interval"
  printf 'early_stopping=false\n'
  if [[ "$arm" == "n" || "$arm" == "n_offline_gated" || "$arm" == "n_path_local" ]]; then
    printf 'behavior_warmup_updates=%s\n' "${AM_MF_BEHAVIOR_WARMUP_UPDATES:-500000}"
  fi
  if [[ "$arm" == "n_path_local" ]]; then
    printf 'transport_target_mode=path_local\n'
    printf 'control_eta=0.1\n'
    printf 'control_eta_ramp_updates=0\n'
    printf 'control_adjoint_clip=0\n'
    printf 'control_uncertainty_scale=0\n'
    printf 'control_uncertainty_end_update=-1\n'
  fi
  if [[ "$arm" == "n_offline_gated" ]]; then
    printf 'control_eta_ramp_updates=%s\n' "${AM_MF_CONTROL_ETA_RAMP_UPDATES:-500000}"
    printf 'control_adjoint_clip=%s\n' "${AM_MF_CONTROL_ADJOINT_CLIP:-1.0}"
    printf 'control_uncertainty_scale=%s\n' "${AM_MF_CONTROL_UNCERTAINTY_SCALE:-0.25}"
    printf 'control_uncertainty_end_update=%s\n' "${AM_MF_CONTROL_UNCERTAINTY_END_UPDATE:-$offline_steps}"
  fi
} > "$output_root/launch_manifest.txt"

bash "$project_root/scripts/experiment_env.sh" -m unittest tests.test_upstream_integrity

exec bash "$project_root/scripts/experiment_env.sh" main_meanflowql.py \
  --env_name="$env_name" \
  --agent="$agent" \
  --agent.alpha="$alpha" \
  --agent.time_steps="$time_steps" \
  --agent.discount="$discount" \
  --agent.num_candidates=5 \
  --agent.consistency_alpha=0 \
  "${agent_extra_flags[@]}" \
  --seed="$seed" \
  --offline_steps="$offline_steps" \
  --online_steps="$online_steps" \
  --buffer_size="$buffer_size" \
  --pretrain_factor=0 \
  --balanced_sampling=0 \
  --use_observation_normalization=True \
  --eval_episodes="$eval_episodes" \
  --eval_interval="$eval_interval" \
  --log_interval="$log_interval" \
  --save_interval="$save_interval" \
  --video_episodes=0 \
  --enable_early_stopping=False \
  --early_stopping_metric="$metric" \
  --wandb_online=False \
  --proj_wandb="$project_name" \
  --run_group="$run_group" \
  --save_dir="$output_root/checkpoints" \
  --wandb_save_dir="$output_root/wandb"
