# Baseline reproduction gate: 2026-09-18

Runs executed on the A800 host with the frozen upstream MeanFlowQL runner.
Every run is B0 (`agents/meanflowql.py`), 1M offline + 1M online updates,
evaluation every 100K updates over 50 episodes, per-task hyperparameters from
Table 7 (see `PAPER_PROTOCOL_MAPPING.md`).  Values are the evaluation at
1,000,000 (offline end) and 2,000,000 gradient steps, matching Appendix D.

Artifacts:

- `artifacts/gate_pen_hammer_v1/` - pen and hammer, 2 seeds each, all four runs
  completed 2M updates in 2h59m - 3h13m.
- `artifacts/repro8_v1/` - eight tasks, 1 seed each, completed 2h56m - 3h39m.

## D4RL (metric: `evaluation/episode.normalized_return`)

| task | ours offline@1M | ours online@2M | paper offline -> online | verdict |
| --- | ---: | ---: | --- | --- |
| hammer-cloned-v1 (seed 1) | 2.03 | 134.38 | 10+-5 -> 132+-11 | online reproduced |
| hammer-cloned-v1 (seed 2) | 4.85 | 127.68 | " | online reproduced |
| pen-cloned-v1 (seed 1) | 81.47 | 117.92 | 79+-3 -> 151+-7 | offline ok, online ~4.6 sd low |
| pen-cloned-v1 (seed 2) | 72.95 | 118.62 | " | offline ok, online ~4.6 sd low |
| door-cloned-v1 | -0.07 | 1.92 | 3+-1 -> 96+-4 | offline marginal, online failed |
| antmaze-large-play-v2 | 80.00 | 96.00 | 80+-3 -> 94+-4 | reproduced |
| antmaze-medium-diverse-v2 | 78.00 | 98.00 | 78+-2 -> 99+-2 | reproduced |
| antmaze-medium-play-v2 | 24.00 | 96.00 | 86+-2 -> 99+-1 | online reproduced, offline low |
| antmaze-umaze-v2 | 94.00 | 98.00 | 98+-1 -> 99+-1 | online reproduced, offline 4 sd low |
| antmaze-umaze-diverse-v2 | 96.00 | 100.00 | 79+-2 -> 100+-1 | online reproduced, offline high |
| relocate-cloned-v1 (earlier, 3 seeds) | ~-0.02 | 21.06 / 47.14 / 39.05 | 1+-1 -> 19+-8 | offline ok, online task-specific |

## OGBench (metric: `evaluation/success`)

| task | ours offline@1M | ours online@2M | paper | verdict |
| --- | ---: | ---: | --- | --- |
| humanoidmaze-medium-navigate-task1 | 0.76 | 1.00 | Table 1 per-task 94+-3; Table 2 env avg 62+-1 -> 100+-1 | online reproduced, offline 6 sd low |
| cube-double-play-task2 | 0.00 | 0.20 | Table 1 per-task 2+-1; Table 2 env avg 3+-2 -> 95+-2 | offline ok, online failed |

## What this establishes

1. The harness itself is sound: seven of the twelve task/settings combinations
   reproduce the published online number (hammer, five antmaze settings,
   humanoidmaze-medium), including two exact offline+online matches
   (antmaze-large-play 80 -> 96 vs 80+-3 -> 94+-4, antmaze-medium-diverse
   78 -> 98 vs 78+-2 -> 99+-2).
2. The earlier relocate anomaly is not a global bug: the same online code path
   lifts hammer-cloned to 131 (paper 132+-11) and every D4RL antmaze task to
   within a few points of the paper.
3. The remaining failures are task-specific and reproducible in direction:
   pen (118 vs 151), door (1.9 vs 96), relocate (21/47/39 vs 19+-8 is the
   weakest link) and cube-double-play (0.2 vs 95) do not reach the published
   online level with the Table 7 settings.
4. Offline points are noisier than online points: antmaze-medium-play (24 vs
   86+-2) and humanoidmaze-medium-task1 (0.76 vs 94+-3) miss offline but
   recover online, which is consistent with the paper's own remark that some
   tasks start from suboptimal offline performance and improve during
   adaptation.  Single-seed offline comparisons should not be over-read.

## Recommended tasks for the N-vs-B0 headline comparison

Use tasks whose B0 online number reproduces, so a win cannot be attributed to a
broken baseline:

- antmaze-large-play-v2, antmaze-medium-diverse-v2, antmaze-umaze-v2,
  antmaze-umaze-diverse-v2 (all cheap, 2M steps in ~3h)
- hammer-cloned-v1
- humanoidmaze-medium-navigate-singletask-task1-v0 (OGBench, 5-task average
  requires the remaining four tasks for a paper-comparable row)

Treat pen / door / relocate / cube-double-play as a separate investigation:
their B0 online phase does not reproduce, so a comparison there measures the
baseline gap rather than the method.
