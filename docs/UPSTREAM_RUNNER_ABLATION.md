# B0/N upstream-runner ablation

This experiment freezes the imported MeanFlowQL execution path at local commit
`ce953a8` (upstream `3fab100`).  The formal B0 and N runs both execute the
unchanged `main_meanflowql.py`, `utils/evaluation.py`, `utils/datasets.py`, and
`envs/env_utils.py`.

The only method switch is the agent configuration:

- B0: `agents/meanflowql.py`.
- N: `agents/am_meanflow_note.py`.

N contains behavior warmup, prior freezing, EMA, interval transport, endpoint
adjoint control, and Jacobian matching inside its `update()` implementation. It
implements the original runner's `create`, `update`, and `sample_actions`
interface. There is no task-specific Python trainer and no replacement evaluator.

The formal Relocate command is wrapped only for repeatability:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_upstream_relocate.sh b0 1 /path/to/results/b0_seed1
CUDA_VISIBLE_DEVICES=1 bash scripts/run_upstream_relocate.sh n 1 /path/to/results/n_seed1
```

Both commands preserve the upstream replay construction, padded-array
normalization, global NumPy sampling, sequential online JAX RNG, unseeded reset,
evaluation timing, and evaluator behavior.  No historical checkpoint is loaded.

The integrity test must pass before the launcher starts training:

```bash
python -m unittest discover -s tests -p 'test_upstream_integrity.py'
```

Additional deterministic evaluation or diagnostics must be performed after
training and reported separately; they are not part of the upstream-faithful
training trajectory.

`launch_upstream_relocate_5seeds.sh` is a historical five-seed convenience
scheduler. The completed V2 Relocate protocol uses seeds 1--3; its exact
configuration, aggregate result, and reporting boundary are in
[`V2_UPSTREAM_RESULTS.md`](V2_UPSTREAM_RESULTS.md). If the historical scheduler
is used on an eight-GPU host, seeds 1--4 start immediately and seed 5 is queued
behind seed 1 on GPUs 0 and 1:

```bash
bash scripts/launch_upstream_relocate_5seeds.sh
```

This file only schedules shell commands. Each child still invokes
`run_upstream_relocate.sh`, which in turn invokes the original Python main.
