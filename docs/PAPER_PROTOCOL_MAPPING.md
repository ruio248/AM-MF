# Paper protocol mapping (MeanFlowQL, arXiv:2511.13035)

This file records how each `scripts/run_upstream_task.sh` task profile maps onto
the paper, so that a launcher profile and a `launch_manifest.txt` can be checked
against the published setting without re-reading the appendix.

Sources: Table 6 (general hyper-parameters), Table 7 (per-task `alpha`,
`num_candidates`, `time_steps`), Appendix D (training and evaluation protocol),
Table 1/2 (reported numbers).

## 1. Protocol implemented by the launcher

| Item | Value | Source |
| --- | --- | --- |
| Updates | 1,000,000 offline + 1,000,000 online | Table 2 caption / Appendix D |
| Reported points | 1M (offline end) and 2M gradient steps | Appendix D |
| Evaluation cadence | every 100,000 updates | Appendix D |
| Evaluation episodes | 50 rollouts | Appendix D |
| D4RL metric | value at the final training step | Appendix D |
| OGBench metric | mean of the last three evaluation epochs | Appendix D |
| Seeds | 8 (state-based) | Appendix D |
| discount | 0.99 (D4RL), 0.995 (OGBench) | Table 6 |
| batch size | 256 | Table 6 |
| sigma / noise | 1.0 / gaussian | Table 6 |
| actor lr / schedule | 1e-4 / cosine_with_warmup (lr_min_ratio 0.1) | Table 6 |
| critic lr / schedule | 1e-4 / constant, grad clip 1.0, kaiming_init | Table 6 |
| tau, tanh_squash | 0.005, False | Table 6 |
| adaptive_gamma / adaptive_c / bound_loss_weight | 0.8 / 1e-4 / 1.0 | Table 6 |
| dynamic alpha | enabled (x1.2 / x0.8 rule, update every 2000 steps) | Table 6, Eq. 21 |

## 2. Task profiles

`alpha`, `C` (num_candidates) and `time_steps` are the Table 7 values.
The last column is the reported offline-to-online number (Table 2, "Ours").

| launcher profile | env_name | alpha | C | time_steps | metric | paper offline -> online |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `relocate_cloned` | relocate-cloned-v1 | 10000 | 5 | 50 | normalized_return | 1+-1 -> 19+-8 |
| `pen_cloned_v1` | pen-cloned-v1 | 10000 | 5 | 100 | normalized_return | 79+-3 -> 151+-7 |
| `hammer_cloned_v1` | hammer-cloned-v1 | 11000 | 5 | 100 | normalized_return | 10+-5 -> 132+-11 |
| `door_cloned_v1` | door-cloned-v1 | 9000 | 5 | 50 | normalized_return | 3+-1 -> 96+-4 |
| `antmaze_umaze_v2` | antmaze-umaze-v2 | 100 | 5 | 10000 | normalized_return (= success %) | 96 -> (online reported) |
| `antmaze_umaze_diverse_v2` | antmaze-umaze-diverse-v2 | 130 | 5 | 100 | normalized_return (= success %) | see Table 2 |
| `antmaze_medium_play_v2` | antmaze-medium-play-v2 | 10 | 5 | 50 | normalized_return (= success %) | 86+-2 -> 99+-1 |
| `antmaze_medium_diverse_v2` | antmaze-medium-diverse-v2 | 50 | 5 | 50 | normalized_return (= success %) | 78+-2 -> 99+-2 |
| `antmaze_large_play_v2` | antmaze-large-play-v2 | 10 | 5 | 100 | normalized_return (= success %) | 80+-3 -> 94+-4 |
| `humanoidmaze_medium_task1` | humanoidmaze-medium-navigate-singletask-task1-v0 | 150 | 5 | 50 | success | 62+-1 -> 100+-1 (env row) |
| `cube_double_play_task2` | cube-double-play-singletask-task2-v0 | 120 | 5 | 50 | success | see Table 2 |
| `humanoid_large_task1` | humanoidmaze-large-navigate-singletask-task1-v0 | 6000 | 5 | 50 | success | upstream README value |
| `humanoid_large_task1_paper` | humanoidmaze-large-navigate-singletask-task1-v0 | 10000 | 5 | 50 | success | Table 7 value |

## 3. Known deviations and gotchas

1. **`humanoid_large_task1` alpha.** The upstream README example uses
   `alpha=6000`, Table 7 lists `10000` for the same task.  Both profiles exist;
   the historical runs (2026-09-13) used 6000, so a Table 7 reproduction must
   use `humanoid_large_task1_paper`.
2. **D4RL antmaze local file names.** `d4rl.offline_env.filepath_from_url` stores
   a dataset under the *basename of the download URL*, not the environment name.
   For antmaze the required file names are `Ant_maze_*_fixed.hdf5`; placing
   `antmaze-umaze-v2.hdf5` makes d4rl silently start a (very slow) download from
   the Berkeley host.  Adroit names do coincide with the environment name.
3. **D4RL antmaze metric.** `utils/evaluation.py` only emits
   `evaluation/success` for environments whose info dict contains it (OGBench).
   For D4RL antmaze, d4rl normalises the 0/1 return by x100, so
   `evaluation/episode.normalized_return` *is* the success percentage.
4. **`num_candidates` is per profile.** It was hard-coded to 5 before
   2026-09-18; Table 7 contains tasks with C=1, C=5 and C=10 (e.g.
   antmaze-giant-navigate uses C=1, antsoccer-arena task1 uses C=10).
5. **Evaluation is not seed-paired.** The upstream evaluator draws its action
   noise from NumPy and calls `env.reset()` without a per-episode seed, so the
   50-episode estimate is noisy (hammer-cloned readings span 0.5-9.5 at 100K
   granularity).  Report the curve, not only the final point.
6. **Missing dataset.** `antmaze-large-diverse-v2` needs
   `Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5`,
   which is absent from the mirror used below; it would require the slow
   Berkeley download.
7. **Visual tasks** (`visual-cube-single`, `visual-cube-double`,
   `visual-scene`, `visual-puzzle-3x3/4x4`) are excluded: the upstream runner
   asserts that online fine-tuning is not supported for visual environments and
   they need the pixel encoder.
8. **The Table 7 alpha values are mis-scaled for this harness on D4RL Adroit.**
   Measured B0 endpoints: door 1.9 at alpha=9000 but 90-91 at alpha=300-900;
   pen 118 at alpha=10000 but 150.1 at alpha=300; hammer 131 at alpha=11000 but
   141.3 at alpha=3000; relocate 0.24 at alpha=10000 but 16.5 at alpha=3000.
   With the retuned values all four tasks reproduce the published numbers.  See
   `DIAGNOSTICS_2026-09-19.md` sections 1-2 for the full sweep and for the
   audit matrix showing that the observation-normalisation zero-padding defect
   and the hard-coded `3e-4` critic learning rate are *not* the cause.

## 4. Dataset provenance

Mirror: `https://hf-mirror.com/datasets/imone/D4RL` (Hugging Face mirror
reachable from the A800 host; ~7.4 MB/s).  The mirror was validated by
downloading `relocate-cloned-v1.hdf5` from both the mirror and
`rail.eecs.berkeley.edu` and comparing sha256: both are
`8baf2b7f0a7e44e6263e89a4c544a5dd98475f2dada833ba60d887e08fc77772`.
Every file below was also checked against the official `Content-Length`.

D4RL datasets live in `D4RL_DATASET_DIR` (default set by
`scripts/experiment_env.sh` to `<worktree parent>/d4rl_datasets`).

| file | bytes | sha256 |
| --- | ---: | --- |
| relocate-cloned-v1.hdf5 | 581319494 | `8baf2b7f...fc77772` |
| pen-cloned-v1.hdf5 | 155074191 | `6d1a982c...c0be0c22` |
| hammer-cloned-v1.hdf5 | 446616879 | `e48d75b1...6cf159db3` |
| door-cloned-v1.hdf5 | 294356136 | `1f6a60ec...d4fe4d48` |
| Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse_fixed.hdf5 | 232532949 | `5ef15257...3065fbd130` |
| Ant_maze_u-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5 | 230854272 | `c771dd63...1d06e8c932` |
| Ant_maze_big-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5 | 231110764 | `c9fec1c1...8018827e5b` |
| Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5 | 231095166 | `ad9dcbac...1902dfae7` |
| Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5 | 230880653 | `6d353df2...fa944f471` |

OGBench datasets live in `<repo>/dataset` (the path `envs/env_utils.py` passes to
`ogbench.make_env_and_datasets`), downloaded from the per-environment mirrors
`Gwanwoo/ogbench-<env>-v0`.

| file | bytes | sha256 |
| --- | ---: | --- |
| humanoidmaze-medium-navigate-v0.npz | 1073650016 | `d5f7e9e7...3cfd241` |
| humanoidmaze-medium-navigate-v0-val.npz | 107386999 | `df603b1b...6a2777abe` |
| cube-double-play-v0.npz | 297435656 | `a73d1a33...cfbcf608c9` |
| cube-double-play-v0-val.npz | 29726524 | `b1fcdf4b...31966013e` |
