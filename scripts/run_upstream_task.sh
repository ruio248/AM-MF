#!/usr/bin/env bash
# Thin launcher only: all optimisation and evaluation remain in the frozen
# upstream main_meanflowql.py.  Do not add task-specific training loops here.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 {b0|n|n_offline_gated|n_path_local|n_online_only} {humanoid_large_task1|humanoid_large_task1_paper|relocate_cloned|pen_cloned_v1|hammer_cloned_v1|door_cloned_v1|antmaze_umaze_v2|antmaze_umaze_diverse_v2|antmaze_medium_play_v2|antmaze_medium_diverse_v2|antmaze_large_play_v2|humanoidmaze_medium_task1|cube_double_play_task2} SEED OUTPUT_ROOT" >&2
  exit 2
fi

arm="$1"
task_profile="$2"
seed="$3"
output_root="$4"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$arm" in
  b0) agent="agents/meanflowql.py" ;;
  n|n_offline_gated|n_path_local|n_online_only) agent="agents/am_meanflow_note.py" ;;
  *)
    echo "arm must be b0, n, n_offline_gated, n_path_local, or n_online_only" >&2
    exit 2
    ;;
esac

case "$task_profile" in
  humanoid_large_task1)
    env_name="humanoidmaze-large-navigate-singletask-task1-v0"
    alpha="6000"
    num_candidates="5"
    discount="0.995"
    time_steps="50"
    metric="evaluation/success"
    project_name="meanflowql_upstream_humanoid_large"
    ;;
  # Same task as humanoid_large_task1 but with the per-task alpha from the
  # paper's Table 7 (10000).  The upstream README example uses 6000, so the two
  # are kept side by side instead of silently changing the historical profile.
  humanoid_large_task1_paper)
    env_name="humanoidmaze-large-navigate-singletask-task1-v0"
    alpha="10000"
    num_candidates="5"
    discount="0.995"
    time_steps="50"
    metric="evaluation/success"
    project_name="meanflowql_upstream_humanoid_large"
    ;;
  relocate_cloned)
    env_name="relocate-cloned-v1"
    alpha="10000"
    num_candidates="5"
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
    num_candidates="5"
    discount="0.99"
    time_steps="100"
    metric="evaluation/episode.normalized_return"
    project_name="meanflowql_upstream_pen"
    ;;
  hammer_cloned_v1)
    env_name="hammer-cloned-v1"
    alpha="11000"
    num_candidates="5"
    discount="0.99"
    time_steps="100"
    metric="evaluation/episode.normalized_return"
    project_name="meanflowql_upstream_hammer"
    ;;
  door_cloned_v1)
    env_name="door-cloned-v1"
    alpha="9000"
    num_candidates="5"
    discount="0.99"
    time_steps="50"
    metric="evaluation/episode.normalized_return"
    project_name="meanflowql_upstream_door"
    ;;
  # D4RL antmaze profiles.  alpha and time_steps are the per-task values from
  # Table 7.  D4RL antmaze returns 0/1 reward per step and ends on success, and
  # d4rl normalises its score by x100, so evaluation/episode.normalized_return
  # *is* the success percentage reported in the paper (the evaluator only emits
  # evaluation/success for OGBench environments).
  antmaze_umaze_v2)
    env_name="antmaze-umaze-v2"
    alpha="100"
    num_candidates="5"
    discount="0.99"
    time_steps="10000"
    metric="evaluation/episode.normalized_return"
    project_name="meanflowql_upstream_antmaze"
    ;;
  antmaze_umaze_diverse_v2)
    env_name="antmaze-umaze-diverse-v2"
    alpha="130"
    num_candidates="5"
    discount="0.99"
    time_steps="100"
    metric="evaluation/episode.normalized_return"
    project_name="meanflowql_upstream_antmaze"
    ;;
  antmaze_medium_play_v2)
    env_name="antmaze-medium-play-v2"
    alpha="10"
    num_candidates="5"
    discount="0.99"
    time_steps="50"
    metric="evaluation/episode.normalized_return"
    project_name="meanflowql_upstream_antmaze"
    ;;
  antmaze_medium_diverse_v2)
    env_name="antmaze-medium-diverse-v2"
    alpha="50"
    num_candidates="5"
    discount="0.99"
    time_steps="50"
    metric="evaluation/episode.normalized_return"
    project_name="meanflowql_upstream_antmaze"
    ;;
  antmaze_large_play_v2)
    env_name="antmaze-large-play-v2"
    alpha="10"
    num_candidates="5"
    discount="0.99"
    time_steps="100"
    metric="evaluation/episode.normalized_return"
    project_name="meanflowql_upstream_antmaze"
    ;;
  # OGBench profiles (state-based; discount 0.995 as in the upstream command).
  humanoidmaze_medium_task1)
    env_name="humanoidmaze-medium-navigate-singletask-task1-v0"
    alpha="150"
    num_candidates="5"
    discount="0.995"
    time_steps="50"
    metric="evaluation/success"
    project_name="meanflowql_upstream_humanoid_medium"
    ;;
  cube_double_play_task2)
    env_name="cube-double-play-singletask-task2-v0"
    alpha="120"
    num_candidates="5"
    discount="0.995"
    time_steps="50"
    metric="evaluation/success"
    project_name="meanflowql_upstream_cube_double"
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
# Optional overrides used by the online-phase ablations.  Defaults keep the
# upstream values (alpha from Table 7, uniform sampling over the offline+online
# buffer, unnormalised Q loss).
alpha="${AM_MF_ALPHA:-$alpha}"
balanced_sampling="${AM_MF_BALANCED_SAMPLING:-0}"
normalize_q_loss="${AM_MF_NORMALIZE_Q_LOSS:-False}"
# Audit flags (audit-fixes branch only): see docs/AUDIT_PATCHES.md.
strict_norm_stats="${AM_MF_STRICT_NORM_STATS:-False}"
critic_lr="${AM_MF_CRITIC_LR:-3e-4}"
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
if [[ "$arm" == "n_online_only" ]]; then
  # AM is off for the entire offline phase and activates on the first online
  # update.  The host loop runs offline updates for i <= offline_steps while the
  # agent picks its phase with `current_step < behavior_warmup_updates`, so the
  # warmup must be offline_steps + 1 for AM to never touch an offline update.
  online_only_warmup=$((offline_steps + 1))
  agent_extra_flags+=(
    "--agent.behavior_warmup_updates=${online_only_warmup}"
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
  printf 'num_candidates=%s\n' "$num_candidates"
  printf 'log_interval=%s\n' "$log_interval"
  printf 'early_stopping=false\n'
  printf 'balanced_sampling=%s\n' "$balanced_sampling"
  printf 'normalize_q_loss=%s\n' "$normalize_q_loss"
  printf 'strict_norm_stats=%s\n' "$strict_norm_stats"
  printf 'critic_lr=%s\n' "$critic_lr"
  if [[ "$arm" == "n" || "$arm" == "n_offline_gated" || "$arm" == "n_path_local" ]]; then
    printf 'behavior_warmup_updates=%s\n' "${AM_MF_BEHAVIOR_WARMUP_UPDATES:-500000}"
  fi
  if [[ "$arm" == "n_online_only" ]]; then
    printf 'behavior_warmup_updates=%s\n' "$online_only_warmup"
    printf 'am_phase=online_only\n'
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
  --agent.num_candidates="$num_candidates" \
  --agent.consistency_alpha=0 \
  "${agent_extra_flags[@]}" \
  --seed="$seed" \
  --offline_steps="$offline_steps" \
  --online_steps="$online_steps" \
  --buffer_size="$buffer_size" \
  --pretrain_factor=0 \
  --balanced_sampling="$balanced_sampling" \
  --agent.normalize_q_loss="$normalize_q_loss" \
  --agent.critic_lr="$critic_lr" \
  --strict_norm_stats="$strict_norm_stats" \
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
