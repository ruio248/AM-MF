# Run provenance: Relocate B0/N and the offline-AM ablations

Scope: every results directory produced for the B0/N comparison on
`relocate-cloned-v1`, the offline-AM ablations, and the humanoid large-task
runs on the A800 host, with the exact code state and configuration each one
used. Compiled 2026-09-17 from the per-run `launch_manifest.txt` files, the
W&B `wandb-metadata.json` argument vectors, and the per-run `eval.csv` records
on that host.

Why this file exists: most of these runs were launched by ad-hoc shell commands
that are not in the repository. Without this record, a results directory can no
longer be tied to a code state, and the negative ablations cannot be
reproduced or cited.

## 1. Code states

| commit | date | subject |
| --- | --- | --- |
| `239edb7` | 2026-09-13 | checkpoint formal runs at each evaluation interval (V2 result card parent) |
| `0d09290` | 2026-09-17 | add gated conservative offline AM ablation: `control_eta_ramp_updates`, `control_adjoint_clip`, `control_uncertainty_scale`, `control_terms()`, control diagnostics |
| `dffbc6f` | 2026-09-17 | gate critic uncertainty during offline only: `control_uncertainty_end_update`, `scheduled_uncertainty_scale()`, schedule threaded through teacher/Jacobian teacher |
| `2047b66` | 2026-09-17 | add path-local transport ablation: `transport_target_mode` in `{interval_mean, path_local}` |
| blob `84e1cda` (uncommitted) | 2026-09-15 | `239edb7` plus a relaxed `behavior_warmup_updates` guard; see below |

### 1.1 The uncommitted blob used by the warmup-0 and online-only runs

Three run groups were launched while `agents/am_meanflow_note.py` was modified
in place, so their `git_commit` alone does not determine their code. The
modification is only the guard relaxation needed to allow
`behavior_warmup_updates=0`:

```diff
@@ -64,8 +64,9 @@
         if config["encoder"] is not None or config["consistency_alpha"] != 0:
             raise ValueError("Note v1 supports vector observations and consistency_alpha=0")
+        if config["behavior_warmup_updates"] < 0:
+            raise ValueError("behavior_warmup_updates must be nonnegative")
         for key in (
-            "behavior_warmup_updates",
             "teacher_steps",
             "teacher_batch_size",
             "jacobian_interval",
```

`239edb7` blob: `80dd3a1`. Working-tree blob at launch: `84e1cda`. No
algorithmic behaviour is changed by this diff, and the same relaxation is
included in `0d09290`; runs that need exact reproduction should therefore use
`0d09290` or later instead of `239edb7` plus the dirty blob.

### 1.2 Attempts that never produced a result

| directory | failure |
| --- | --- |
| `relocate_warmup0/20260915_124158` | `AttributeError: 'NoneType' object has no attribute 'eglQueryString'` (host lacked the EGL runtime) |
| `relocate_warmup0/20260915_130233` | `ValueError: behavior_warmup_updates must be positive` (guard not yet relaxed) |
| `relocate_warmup500k_offline/20260915_232137` | EGL `eglQueryString` error, all three seeds |
| `relocate_warmup500k_offline/20260915_233228` | `OSError: Unable to synchronously open file (truncated file: eof = 8388608, stored_eof = 581319494)` — truncated D4RL dataset, all three seeds |

The preceding item is why `scripts/install_user_egl.sh` and the dataset
download helper exist in this repository.

## 2. Run groups

All rows are `relocate-cloned-v1` with `alpha=10000`, `discount=0.99`,
`time_steps=50`, `num_candidates=5`, `eval_episodes=50`,
`eval_interval=100000`, early stopping disabled, and 1,000,000 offline /
1,000,000 online updates unless stated. `W` is `behavior_warmup_updates`.
Final column is the terminal `evaluation/episode.normalized_return` per seed,
at the step shown when the run did not reach 2,000,000.

| # | directory (under `artifacts/`) | arm | seeds | launched | code | tree | distinguishing flags | final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `formal/relocate_cloned/20260914_113322/b0_seed{1,2,3}` | `b0` | 1,2,3 | 2026-09-14 11:33 | `239edb7` | clean | upstream MeanFlowQL actor | 0.723 / 0.031 / -0.026 |
| 2 | `formal/relocate_cloned/20260914_113322/n_seed{1,2,3}` | `n` | 1,2,3 | 2026-09-14 11:33 | `239edb7` | clean | `W=500000` | 21.061 / 47.139 / 39.048 |
| 3 | `formal/relocate_warmup0/20260915_130940/n_seed{1,2,3}` | `n` | 1,2,3 | 2026-09-15 13:09 | `239edb7` + `84e1cda` | dirty | `W=0` | 0.008@1.9M / -0.033 / 0.037@1.9M (seeds 1,3 stopped at 99%) |
| 4 | `formal/relocate_warmup500k_offline/20260915_233513_rerun/seed_{1,2,3}` | `n` | 1,2,3 | 2026-09-15 23:35 | `239edb7` | clean | `W=500000`, `offline_steps=1000000`, `online_steps=0` | -0.063 / 0.015 / -0.078 |
| 5 | `formal/relocate_warmup500k_offline_to_online/20260916_094908/seed_{1,2,3}` | `n` | 1,2,3 | 2026-09-16 09:49 | `239edb7` | clean | restores #4 at epoch 1,000,000, `W=0`, `offline_steps=0`, `online_steps=1000000` | 18.303 / 44.512 / 57.519 |
| 6 | `n_online_only_v1/seed00{1,2,3}` | `n` | 1,2,3 | 2026-09-16 20:00 | `239edb7` + `84e1cda` | dirty | `W=1000001` (control first active on the first online update) | 17.390 / 2.447 / 0.367 |
| 7 | `n_offline_gated_nowarmup_v1/seed00{1,2,3}` | `n_offline_gated` | 1,2,3 | 2026-09-17 04:07 | `dffbc6f` | clean | `W=0`, `control_eta_ramp_updates=500000`, `control_adjoint_clip=1.0`, `control_uncertainty_scale=0.25`, `control_uncertainty_end_update=1000000` | 0.057 / 0.028 / 0.035 |
| 8 | `n_path_local_warmup500k_v1/seed00{1,2,3}` | `n_path_local` | 1,2,3 | 2026-09-17 14:22 | `2047b66` | clean | `W=500000`, `transport_target_mode=path_local`, `control_eta=0.1`, ramp 0, clip 0, uncertainty 0, `end_update=-1` | in progress; 0.051@1.9M / 0.036@1.8M / 0.031@1.8M |
| 9 | `offline_gated_smoke/seed001` | `n_offline_gated` | 1 | 2026-09-17 02:38 | `0d09290` | clean | smoke: `W=2`, `offline_steps=10`, ramp 10 | -0.029 @10 |
| 10 | `n_path_local_smoke_v1/seed001` | `n_path_local` | 1 | 2026-09-17 14:18 | `2047b66` | clean | smoke: `W=2`, `offline_steps=10` | -0.205 @10 |
| 11 | `formal/humanoid_large_task1/{b0,n}_seed{1,2,3}_20260913_152300` | `b0`, `n` | 1,2,3 | 2026-09-13 15:23 | `239edb7` | clean | `humanoidmaze-large-navigate-singletask-task1-v0`, `alpha=6000`, `discount=0.995`, metric `evaluation/success` | b0 1.00 / 0.96 / 0.98, n 0.98 / 0.96 / 0.98 (saturated) |

Runs 1 and 2 are the source of the V2 Relocate result card. Runs 3-8 are the
ablations; run 11 is retained for completeness but does not discriminate
between arms because both saturate on `evaluation/success`.

## 3. Record inventory

`launch_manifest.txt` is written by `scripts/run_upstream_task.sh` at launch and
records `git_commit`, `git_status`, the arm, the task profile, the seed, the
budget, and the arm-specific flags.

| directory | `launch_manifest.txt` | W&B argv | note |
| --- | --- | --- | --- |
| `formal/relocate_cloned/20260914_113322/*` | yes | yes | |
| `formal/relocate_warmup0/20260915_124158/*` | yes | no | died before W&B initialised |
| `formal/relocate_warmup0/20260915_130233/*` | yes | yes | |
| `formal/relocate_warmup0/20260915_130940/*` | yes | yes | tree dirty at launch |
| `formal/relocate_warmup500k_offline/20260915_232137/*` | no | no | aborted, no result |
| `formal/relocate_warmup500k_offline/20260915_233228/*` | no | yes | aborted, no result |
| `formal/relocate_warmup500k_offline/20260915_233513_rerun/*` | no | yes | configurations reconstructed from the W&B argv |
| `formal/relocate_warmup500k_offline_to_online/20260916_094908/*` | no | yes | configurations reconstructed from the W&B argv |
| `n_online_only_v1/*` | yes | yes | tree dirty at launch |
| `n_offline_gated_nowarmup_v1/*` | yes | yes | |
| `n_path_local_warmup500k_v1/*`, `n_path_local_smoke_v1/*` | yes | yes | |
| `offline_gated_smoke/*` | yes | yes | |
| `formal/humanoid_large_task1/*` | yes | yes | |
| `pilots/`, `smoke/` | yes | yes | smoke only |

## 4. Known gaps

1. Four groups have no `launch_manifest.txt` (rows 4 and 5 above, plus the two
   aborted offline attempts). Their configuration is recoverable only from
   `wandb/wandb/offline-run-*/files/wandb-metadata.json`.
2. The D4RL dataset `~/.d4rl/datasets/relocate-cloned-v1.hdf5` was rewritten on
   2026-09-16 18:43, between run group 5 (finished 15:13) and run group 6
   (started 20:00). The file present today is 581,319,494 bytes with
   `sha256 8baf2b7f0a7e44e6263e89a4c544a5dd98475f2dada833ba60d887e08fc77772`.
   Earlier groups cannot be re-verified against that hash, and the training
   logs do not record a dataset digest.
3. `utils/evaluation.py` seeds evaluation action noise from NumPy and calls
   `env.reset()` without a per-episode seed, so evaluation trajectories are not
   paired across arms and the terminal evaluation point is a single noisy
   sample. Reporting should use an average over the last several evaluations.
4. Run group 3 has two seeds stopped at 99% of the budget with no error marker,
   so its per-seed values are not all at the same step.
5. Every run except the two path-local groups used the interpreter from a
   different worktree (`AM-MF-relocate-note/.venv`); the environment is not
   pinned per run.
