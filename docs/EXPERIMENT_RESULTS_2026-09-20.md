# Experiment results register: B0, N, alpha, and adjoint sweeps

Date: 2026-09-20.  Hosts: A800 (`new_server_rh_2`) and RTX 4090
(`new_server_4090`).  All final values below are the last
`evaluation/episode.normalized_return` recorded at 2,000,000 training updates
unless stated otherwise.

This document collects the completed results for the baseline reproduction,
baseline alpha sensitivity, the formal N method, the N alpha auxiliary sweep,
and the adjoint-control-strength (`control_eta`) sweep.  The headline
N-versus-B0 comparison is also recorded in
[`SEVEN_TASK_COMPARISON.md`](SEVEN_TASK_COMPARISON.md).

## Protocol

The formal N/B0 comparison uses the frozen upstream MeanFlowQL entry point:

| Setting | Value |
| --- | --- |
| Budget | 1,000,000 offline + 1,000,000 online updates |
| Evaluation | Every 100,000 updates; 50 rollout episodes |
| Metric | D4RL normalized return |
| N agent | `agents/am_meanflow_note.py` |
| B0 agent | Upstream `agents/meanflowql.py` |
| N defaults | `behavior_warmup_updates=500000`, `control_eta=0.1`, `teacher_steps=8`, `jacobian_coef=0.01`, `consistency_alpha=0` |

The alpha and `control_eta` sweeps are seed-1 runs unless noted otherwise.

## 1. Formal comparison: N vs B0 vs published MeanFlowQL/FQL

Values are `offline@1M -> online@2M`.  For Pen and Hammer, the offline values
show the two seed endpoints; the online value is the mean of the two final
points.

| Task | N (this repo) | B0 (this repo) | N - B0 online | Paper MeanFlowQL | Paper FQL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `relocate-cloned-v1` | -0.007 -> **35.7** | -0.03 -> 0.2 | **+35.5** | 1±1 -> 19±8 | 0±1 -> 62±8 |
| `pen-cloned-v1` | 45.8/61.7 -> **134.2** | 81.5/73.0 -> 118.3 | **+15.9** | 79±3 -> 151±7 | 53±14 -> 149±6 |
| `hammer-cloned-v1` | 0.2/9.0 -> 129.6 | 2.0/4.9 -> **131.0** | -1.4 | 10±5 -> 132±11 | 0±0 -> 127±17 |
| `door-cloned-v1` | -0.01 -> **36.3** | -0.07 -> 1.9 | **+34.4** | 3±1 -> 96±4 | 0±0 -> 102±5 |
| `antmaze-large-play-v2` | 46 -> 90 | 80 -> **96** | -6 | 80±3 -> 94±4 | 66±40 -> 84±30 |
| `antmaze-medium-diverse-v2` | 82 -> 92 | 78 -> **98** | -6 | 78±2 -> 99±2 | 55±19 -> 97±3 |
| `antmaze-umaze-v2` | 98 -> 98 | 94 -> 98 | 0 | 98±1 -> 99±1 | 97±2 -> 99±1 |

### Seed-level final online values

| Task | N final values | N mean | B0 final values | B0 mean |
| --- | --- | ---: | --- | ---: |
| `relocate-cloned-v1` | 21.06 / 47.14 / 39.05 | **35.75** | 0.72 / 0.03 / -0.03 | **0.24** |
| `pen-cloned-v1` | 133.86 / 134.46 | **134.16** | 117.92 / 118.62 | **118.27** |
| `hammer-cloned-v1` | 125.37 / 133.86 | **129.62** | 134.38 / 127.68 | **131.03** |
| `door-cloned-v1` | 36.32 | 36.32 | 1.92 | 1.92 |
| `antmaze-large-play-v2` | 90 | 90 | 96 | 96 |
| `antmaze-medium-diverse-v2` | 92 | 92 | 98 | 98 |
| `antmaze-umaze-v2` | 98 | 98 | 98 | 98 |

Headline outcome: N clearly beats the reproduced B0 on Relocate, Pen, and
Door; is effectively tied on Hammer and Umaze; and is below B0 on Large-Play
and Medium-Diverse AntMaze.  The published paper numbers are external
reference anchors, not a same-seed statistical test against this run.

## 2. Baseline alpha sensitivity: core sweep

Artifacts:

```text
artifacts/b0_alpha_sweep_v1/
```

All eight core runs completed 2M updates.

| Task | `alpha` | Arm | Seed | Final normalized return |
| --- | ---: | --- | ---: | ---: |
| `relocate-cloned-v1` | 300 | B0 | 1 | 0.03 |
| `relocate-cloned-v1` | 3000 | B0 | 1 | **16.53** |
| `pen-cloned-v1` | 300 | B0 | 1 | **150.13** |
| `pen-cloned-v1` | 3000 | B0 | 1 | 144.56 |
| `hammer-cloned-v1` | 3000 | B0 | 1 | **141.32** |
| `hammer-cloned-v1` | 33000 | B0 | 1 | 93.87 |
| `door-cloned-v1` | 300 | B0 | 1 | **90.03** |
| `door-cloned-v1` | 3000 | B0 | 1 | 61.79 |

## 3. N method: formal main experiment

Artifacts:

```text
artifacts/n_main_v1/
artifacts/formal/relocate_cloned/
```

The formal N main experiment contains 11 completed seed-runs: three Relocate
seeds, two Pen seeds, two Hammer seeds, and one seed for each of the three
AntMaze tasks plus Door.

| Task | Seeds | Offline@1M | Online@2M | Final status |
| --- | ---: | ---: | ---: | --- |
| `relocate-cloned-v1` | 1/2/3 | -0.007 ± 0.068 | **35.75 mean** | complete |
| `pen-cloned-v1` | 1/2 | 45.8 / 61.7 | **134.16 mean** | complete |
| `hammer-cloned-v1` | 1/2 | 0.2 / 9.0 | **129.62 mean** | complete |
| `door-cloned-v1` | 1 | -0.01 | **36.32** | complete |
| `antmaze-large-play-v2` | 1 | 46 | **90** | complete |
| `antmaze-medium-diverse-v2` | 1 | 82 | **92** | complete |
| `antmaze-umaze-v2` | 1 | 98 | **98** | complete |

## 4. N method: alpha auxiliary sweep

Artifacts:

```text
artifacts/n_alpha_sweep_v1/
artifacts/alpha_aligned_v1/
```

All eight listed runs completed 2M updates.  The final B0 row is the aligned
control run included with the N alpha experiments.

| Experiment group | Task | Arm | `alpha` | Seed | Final normalized return |
| --- | --- | --- | ---: | ---: | ---: |
| `n_alpha_sweep_v1` | `relocate-cloned-v1` | N | 300 | 1 | 2.30 |
| `n_alpha_sweep_v1` | `pen-cloned-v1` | N | 300 | 1 | **143.69** |
| `n_alpha_sweep_v1` | `hammer-cloned-v1` | N | 3000 | 1 | 129.14 |
| `n_alpha_sweep_v1` | `door-cloned-v1` | N | 300 | 1 | **33.86** |
| `alpha_aligned_v1` | `relocate-cloned-v1` | N | 1000 | 1 | **21.79** |
| `alpha_aligned_v1` | `pen-cloned-v1` | N | 1000 | 1 | 137.67 |
| `alpha_aligned_v1` | `door-cloned-v1` | N | 900 | 1 | 29.62 |
| `alpha_aligned_v1` | `relocate-cloned-v1` | B0 control | 1000 | 1 | 1.14 |

## 5. Adjoint control strength: `control_eta` sweep

The current implementation uses:

```text
controlled_velocity = prior - control_eta * adjoint
```

Artifacts:

```text
artifacts/eta_sweep_v1/
```

All eight runs completed 2M updates.

| Task | `control_eta` | Seed | Final normalized return | Rank within task |
| --- | ---: | ---: | ---: | ---: |
| `relocate-cloned-v1` | 0 | 1 | -0.05 | 5 |
| `relocate-cloned-v1` | 0.03 | 1 | 5.26 | 3 |
| `relocate-cloned-v1` | 0.1 | 1 | **52.24** | **1** |
| `relocate-cloned-v1` | 0.3 | 1 | 12.64 | 2 |
| `relocate-cloned-v1` | 1.0 | 1 | 0.16 | 4 |
| `door-cloned-v1` | 0 | 1 | -0.08 | 3 |
| `door-cloned-v1` | 0.3 | 1 | 1.34 | 2 |
| `door-cloned-v1` | 1.0 | 1 | **64.49** | **1** |

The best coefficient is task-dependent in this sweep: `0.1` for Relocate and
`1.0` for Door.

## 6. Formal B0 baseline reproduction

Artifacts:

```text
artifacts/repro8_v1/
artifacts/gate_pen_hammer_v1/
artifacts/formal/relocate_cloned/
```

The formal seven-task B0 comparison set is complete at 2M.

| Task | Seeds | Offline@1M | Online@2M | Final status |
| --- | ---: | ---: | ---: | --- |
| `relocate-cloned-v1` | 1/2/3 | approximately -0.03 | **0.24 mean** | complete |
| `pen-cloned-v1` | 1/2 | 81.5 / 73.0 | **118.27 mean** | complete |
| `hammer-cloned-v1` | 1/2 | 2.0 / 4.9 | **131.03 mean** | complete |
| `door-cloned-v1` | 1 | -0.07 | **1.92** | complete |
| `antmaze-large-play-v2` | 1 | 80 | **96** | complete |
| `antmaze-medium-diverse-v2` | 1 | 78 | **98** | complete |
| `antmaze-umaze-v2` | 1 | 94 | **98** | complete |

## Completion notes

- Formal B0 reproduction: complete.
- Formal N main experiment: complete.
- Baseline alpha core sweep: 8/8 complete.
- N alpha auxiliary sweep: 8/8 complete.
- Adjoint `control_eta` sweep: 8/8 complete.
- The separate 4090 `alpha_ablation_v1` experiment also contains two failed
  balanced variants (`door_balanced` and `pen_balanced`) caused by replay
  buffer/dataset shape mismatches.  Those are not part of the eight-run core
  B0 alpha sweep above and are recorded as follow-up failures rather than
  successful results.

