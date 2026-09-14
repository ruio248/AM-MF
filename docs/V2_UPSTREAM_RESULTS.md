# V2: N adjoint matching under the frozen MeanFlowQL execution path

## Scope of this release

V2 is the clean, reproducible code release for the current **N** method:
endpoint-Q adjoint credit assignment, local prior-constrained velocity control,
one controlled teacher trajectory, and interval-average-velocity distillation.
It also contains the launcher and the result card for the completed B0/N
Relocate comparison.

The release deliberately contains **code, tests, launchers, configuration, and
aggregate results only**. D4RL data, MuJoCo assets, W&B files, checkpoints, and
raw experiment outputs are not versioned.

## N algorithm

N does not optimize Q directly through the final action prediction. Given a
frozen behavior prior `pre_actor` and an EMA endpoint map, it first assigns the
terminal-Q signal to an intermediate state:

\[
\lambda_t = \nabla_{x_t} Q_{\rm tar}\!\left(
o,\operatorname{clip}(F_{\bar\theta}^{0\leftarrow t}(o,x_t))\right).
\]

Under the repository time convention (`t=1` is noise and `t=0` is action), the
local proximal control is:

\[
v_\star(o,x_t,t)=v_{\rm pre}(o,x_t,t)-\eta\lambda_t.
\]

It integrates one RK2 controlled path and regenerates every interval target
from that same path:

\[
\Phi_\star^{r\leftarrow t}(o,x_t)
=x_t-\int_r^t v_\star(o,x_\tau,\tau)\,d\tau,
\qquad
U_\star(o,x_t,r,t)
=\frac{x_t-\Phi_\star^{r\leftarrow t}(o,x_t)}{t-r}.
\]

Thus the interval targets share one teacher trajectory. This is not the old
AlphaFlow-style target-mixing consistency regularizer: V2 sets
`consistency_alpha=0`. The retained structural principle is interval
composability, obtained by construction of the controlled path rather than by
an additional consistency-loss target.

The implementation is [`agents/am_meanflow_note.py`](../agents/am_meanflow_note.py).
Its endpoint-adjoint, RK2 integration, and finite-difference directional
utilities are in [`utils/note_flow.py`](../utils/note_flow.py).

## Frozen upstream execution boundary

The imported MeanFlowQL pathway is frozen at local commit `ce953a8` (upstream
`3fab100`). Both formal arms execute the unchanged:

- `main_meanflowql.py` training main;
- `utils/evaluation.py` evaluator;
- `utils/datasets.py` replay/data handling; and
- `envs/env_utils.py` environment creation.

`tests/test_upstream_integrity.py` hashes these files and the canonical launcher
runs that test before training. No task-specific Python training loop,
replacement evaluator, historical baseline actor, or historical critic
checkpoint is used.

| Arm | Agent | Method-specific lifecycle |
| --- | --- | --- |
| B0 | `agents/meanflowql.py` | Original MeanFlowQL actor and Direct-Q/MFI training. |
| N | `agents/am_meanflow_note.py` | 500k-update behavior initialization, frozen prior, EMA map, controlled-path transport matching, and Jacobian matching. |

B0 and N use the same upstream main/evaluator, data path, replay construction,
normalization option, offline/online budgets, candidate count, evaluation
cadence, and no-checkpoint initialization. This is a controlled **end-to-end
method** comparison. It is not an adjoint-only attribution: a matching
interval-actor Direct-Q control (`B1`) is required for that narrower claim.

The upstream evaluator seeds its action-noise key from NumPy and calls
`env.reset()` without a per-episode fixed seed. B0 and N use that identical
evaluator pathway, but their 50-episode evaluation trajectories are not
retrospectively paired by a separate fixed evaluation seed.

## Canonical Relocate protocol

The completed V2 run uses `relocate-cloned-v1`, training seeds `1, 2, 3`, 50
evaluation episodes, and C5 candidate action selection.

| Setting | Value |
| --- | ---: |
| Offline updates | 1,000,001 actual updates in the upstream loop |
| Online interaction / updates | 1,000,000 / 1,000,000 |
| `agent.alpha` | 10,000 |
| `agent.time_steps` | 50 |
| discount | 0.99 |
| candidate actions | 5 |
| replay capacity | 2,000,000 |
| evaluation cadence | every 100,000 updates |
| early stopping | disabled |
| N behavior warmup | 500,000 updates, included in N's offline budget |
| N controlled teacher | 8-step midpoint RK2 |
| N `consistency_alpha` | 0 |

Run one seed through the canonical launcher:

```bash
# Run in the configured project environment; see scripts/experiment_env.sh.
CUDA_VISIBLE_DEVICES=0 bash scripts/run_upstream_task.sh \
  b0 relocate_cloned 1 /path/to/results/b0_seed1

CUDA_VISIBLE_DEVICES=0 bash scripts/run_upstream_task.sh \
  n relocate_cloned 1 /path/to/results/n_seed1
```

The launcher writes a `launch_manifest.txt` alongside each result directory.
Raw result files remain external to Git.

## Completed Relocate result

This table is the final upstream-evaluator record at total training step
2,000,000, not a selected intermediate checkpoint. The primary quantity is
`evaluation/episode.normalized_return`, on the native D4RL normalized-return
scale used by `utils/evaluation.py`.

| Method | Seed 1 | Seed 2 | Seed 3 | Mean ± sample std. |
| --- | ---: | ---: | ---: | ---: |
| B0: MeanFlowQL | 0.723 | 0.031 | -0.026 | **0.243 ± 0.417** |
| N: controlled-path adjoint matching | 21.061 | 47.139 | 39.048 | **35.749 ± 13.348** |

For context, the raw upstream CSV's terminal `evaluation/goal_achieved` field
was `0.007 ± 0.012` for B0 and `0.613 ± 0.214` for N. It is **not** a strict
episode-success metric: it is the evaluator's final-info aggregate. A strict
Relocate success claim requires trajectory-level recomputation, for example
`sum_t(goal_achieved_t) > 25`, and is not inferred from this field.

The offline endpoint was near zero for both arms: B0 `-0.025 ± 0.025`, N
`-0.007 ± 0.068`. The observed separation emerged during online adaptation;
the final table should not be read as an offline improvement.

## Interpretation and non-claims

This run provides a positive three-seed end-to-end signal for N **against B0
under the shared frozen upstream execution path**. It does not yet establish:

1. a strict numerical replication of the paper's published Relocate result:
   this B0 run is lower than the reported public MeanFlowQL Relocate reference;
2. that the adjoint mechanism alone, separate from N's interval architecture
   and lifecycle, causes the improvement; or
3. a comparison or win against QAM-FQL, which uses its own task/action-chunking
   and evaluation protocol.

Those claims require, respectively, an explained B0 reproduction gap, the B1
mechanism control, and a separately protocol-aligned QAM comparison.

## Verification

Run the tests before a new experiment:

```bash
python -m unittest discover -s tests -p 'test_upstream_integrity.py'
python -m unittest discover -s tests -p 'test_note_flow.py'
python -m unittest discover -s tests -p 'test_am_meanflow_note.py'
```

The test suite covers the frozen-file integrity guard, time direction,
controlled-velocity sign, interval composition, adjoint finite-difference
agreement, stop-gradient boundaries, warmup/prior lifecycle, and the
upstream-compatible agent interface.
