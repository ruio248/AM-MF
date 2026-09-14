# One-Step Generative Policies with Q-Learning: A Reformulation of MeanFlow
[![AAAI](https://img.shields.io/badge/AAAI-Accepted-87CEEB.svg)](https://aaai.org/)
[![arXiv](https://img.shields.io/badge/Paper-arXiv-FFB6C1.svg)](https://arxiv.org/abs/2511.13035)

<p align="center">
  <img src="./toy_example/teaser.png" width="65%">
</p>

## V2: N algorithm and upstream-faithful evaluation

This repository's **V2** release contains the clean implementation of the N
adjoint-matching method and a reproducible B0/N experiment path that retains
the original MeanFlowQL training main and evaluator. The method performs
prior-constrained local velocity control using the endpoint-Q adjoint, then
integrates one controlled path and distills consistent interval-average
velocity targets from it.

The complete method definition, exact Relocate command, final three-seed
result, and the limits of the comparison are documented in
[`docs/V2_UPSTREAM_RESULTS.md`](docs/V2_UPSTREAM_RESULTS.md). In particular,
the reported result is a same-upstream-path B0/N comparison; it is not yet a
paper-level Relocate replication or a QAM-FQL protocol-aligned comparison.

To start a V2 Relocate seed, use the thin upstream launcher rather than a
task-specific trainer:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_upstream_task.sh \
  n relocate_cloned 1 /path/to/results/n_seed1
```

The launcher runs the frozen-upstream integrity test before training. Dataset,
checkpoint, and W&B artifacts are intentionally external to Git.



## Setup

### Environment Setup

```bash
conda env create -f environment.yml
conda activate flowrl
```

### Dataset Setup

```
python download_all_datasets.py
```

## Toy Experiments

```
python toy_example/verify_fql_flow_fit.py
python toy_example/verify_meanflow_uat.py
```

If you wish to evaluate the performance of our different variants on the toy example, please modify the training objective in the `meanflow_loss` section and the action sampling logic in the `sample_action` function within the `meanflowql_beta.py` file accordingly.

## Offline Experiments

```
Example:
python main_meanflowql.py --env_name=humanoidmaze-large-navigate-singletask-task1-v0 --agent=agents/meanflowql.py --agent.time_steps=50 --early_stopping_metric=evaluation/success --offline_steps=1000005 --early_stopping_patience=10 --proj_wandb=0726_humanoidmaze-large-navigate-singletask-task1-v0 --wandb_save_dir=07281632_meanflowRL_param_search --run_group=meanflow_param_search --seed=1 --agent.alpha=6000 --agent.discount=0.995 --agent.num_candidates=5 --wandb_online=True
```

## Offline to Online Experiments

```
Example:
python main_meanflowql.py --env_name=humanoidmaze-medium-navigate-singletask-task1-v0 --agent=agents/meanflowql.py --online_steps=1000000 --offline_steps=1000000 --proj_wandb=0729online-humanoidmaze-medium-navigate-singletask-task1-v0 --wandb_save_dir=meanflowRL_online --run_group=meanflow_online --seed=1 --early_stopping_patience=100 --wandb_online=True --agent.time_steps=100 --agent.alpha=2000 --agent.discount=0.995 --agent.num_candidates=5
```
## License
This project is released under the MIT License.
