#!/usr/bin/env bash
set -euo pipefail

# This is a launcher, not a training runner.  Both arms execute the untouched
# upstream main_meanflowql.py and differ only in the selected agent config.

if [[ $# -lt 3 || $# -gt 3 ]]; then
  echo "usage: $0 {b0|n} SEED OUTPUT_ROOT" >&2
  exit 2
fi

arm="$1"
seed="$2"
output_root="$3"

case "$arm" in
  b0)
    agent="agents/meanflowql.py"
    ;;
  n)
    agent="agents/am_meanflow_note.py"
    ;;
  *)
    echo "arm must be b0 or n" >&2
    exit 2
    ;;
esac

python -m unittest discover -s tests -p 'test_upstream_integrity.py'

exec python main_meanflowql.py \
  --env_name=relocate-cloned-v1 \
  --agent="$agent" \
  --agent.alpha=10000 \
  --agent.time_steps=50 \
  --agent.discount=0.99 \
  --agent.num_candidates=5 \
  --agent.consistency_alpha=0 \
  --seed="$seed" \
  --offline_steps=1000000 \
  --online_steps=1000000 \
  --buffer_size=2000000 \
  --pretrain_factor=0 \
  --balanced_sampling=0 \
  --use_observation_normalization=True \
  --eval_episodes=50 \
  --eval_interval=100000 \
  --save_interval=1000000 \
  --video_episodes=0 \
  --wandb_online=False \
  --proj_wandb=meanflowql_upstream_relocate \
  --run_group="${arm}_seed${seed}" \
  --save_dir="$output_root/checkpoints" \
  --wandb_save_dir="$output_root/wandb"
