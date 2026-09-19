# Diagnostics: D4RL Adroit reproduction, alpha, and the online phase

Date: 2026-09-19.  Hosts: A800 (`new_server_rh_2`) and 4090 (`new_server_4090`).
All runs are B0 (`agents/meanflowql.py`) unless marked `n` (N / TI-AM), 1M
offline + 1M online updates, evaluation every 100K updates over 50 episodes,
per-task `time_steps`/`num_candidates` from the paper's Table 7.  Values are
`evaluation/episode.normalized_return` at the final training step unless a step
is shown.

## 1. Headline result: the per-task BC coefficient of Table 7 does not transfer

The paper's Table 7 alpha values are far too large for this implementation on
the D4RL Adroit tasks.  Retuning alpha per task recovers the published numbers:

| task | Table 7 alpha | B0 online at that alpha | best alpha found | B0 online at best alpha | paper |
| --- | ---: | ---: | ---: | ---: | ---: |
| door-cloned-v1 | 9000 | 1.9 | 300-900 | **90.0-91.0** | 96 +- 4 |
| pen-cloned-v1 | 10000 | 118.3 | 300 | **150.1** | 151 +- 7 |
| hammer-cloned-v1 | 11000 | 131.0 | 3000 | **141.3** | 132 +- 11 |
| relocate-cloned-v1 | 10000 | 0.24 | 3000 | **16.5** | 19 +- 8 |

Full sweep (B0, seed 1, final step; `offline@1M` shown because the trade-off
matters):

| task | alpha | offline@1M | online@2M |
| --- | ---: | ---: | ---: |
| door | 90 | -0.3 | 84.8 |
| door | 300 | -0.06 | 90.0 |
| door | 900 | -0.0 | 91.0 |
| door | 3000 | -0.02 | 61.8 |
| door | 9000 | -0.07 | 1.9 |
| pen | 300 | -0.38 | 150.1 |
| pen | 1000 | -0.2 | 142.3 |
| pen | 3000 | 32.97 | 144.6 |
| pen | 10000 | 77.2 | 118.3 |
| hammer | 3000 | 0.14 | 141.3 |
| hammer | 11000 | 3.4 | 131.0 |
| hammer | 33000 | 0.36 | 93.9 |
| relocate | 300 | -0.03 | 0.03 |
| relocate | 1000 | -0.0 | 1.1 |
| relocate | 3000 | -0.03 | 16.5 |
| relocate | 10000 | -0.007 | 0.24 |

Observations:

1. The direction of the correction is task-dependent in magnitude: door needs
   roughly 1/10-1/30 of the Table 7 value, pen 1/30, hammer 1/3, relocate 1/3.
2. Smaller alpha means a worse offline endpoint and a better online endpoint
   (pen: offline 77.2 -> -0.38 as alpha drops 10000 -> 300).  The published alpha
   values appear to be tuned for the offline endpoint; this harness's online
   phase prefers a much smaller BC weight.
3. All four Adroit tasks reproduce (within ~1.5 sd) once alpha is retuned, so
   the earlier "task-specific online failure" is an alpha-scale problem, not a
   missing mechanism.  Reproduction status by task (B0, best alpha):

| task | offline@1M (ours vs paper) | online@2M (ours vs paper) | status |
| --- | --- | --- | --- |
| door-cloned | -0.06 vs 3 +- 1 | 90.0 vs 96 +- 4 | reproduced (1.5 sd) |
| pen-cloned | -0.38 vs 79 +- 3 | 150.1 vs 151 +- 7 | reproduced (matches) |
| hammer-cloned | 0.14 vs 10 +- 5 | 141.3 vs 132 +- 11 | reproduced (above) |
| relocate-cloned | -0.03 vs 1 +- 1 | 16.5 vs 19 +- 8 | reproduced (0.3 sd) |
| antmaze-large-play | 80.0 vs 80 +- 3 | 96.0 vs 94 +- 4 | reproduced |
| antmaze-medium-diverse | 78.0 vs 78 +- 2 | 98.0 vs 99 +- 2 | reproduced |
| antmaze-medium-play | 24.0 vs 86 +- 2 | 96.0 vs 99 +- 1 | online reproduced |
| antmaze-umaze | 94.0 vs 98 +- 1 | 98.0 vs 99 +- 1 | online reproduced |
| antmaze-umaze-diverse | 96.0 vs 79 +- 2 | 100.0 vs 100 +- 1 | online reproduced |
| humanoidmaze-medium-task1 | 0.76 vs 94 +- 3 (Table 1) | 1.00 vs 100 +- 1 | online reproduced |
| cube-double-play-task2 | 0.00 vs 2 +- 1 (Table 1) | 0.20 vs 95 +- 2 (Table 2 env avg) | online failed |

## 2. The audit matrix: the normalisation bug and the critic LR are not the cause

`docs/AUDIT_PATCHES.md` documents two real upstream defects: the observation
statistics are computed over the zero-padded replay buffer (mean shifted by
50-75% depending on the dataset size), and the critic learning rate is
hard-coded to `3e-4` while the paper's Table 6 lists `1e-4`.  With both fixes
made opt-in and run as a 2x2 on door:

| arm | strict_norm_stats | critic_lr | online@2M |
| --- | --- | --- | ---: |
| A (control) | False | 3e-4 | 0.01 |
| B | True | 3e-4 | 0.04 |
| C | False | 1e-4 | 0.04 |
| D | True | 1e-4 | 0.00 |
| E (hammer, norm fix) | True | 3e-4 | 119.8 |

All four door arms stay near zero, so neither defect explains the door failure
(alpha does).  The hammer control suggests the normalisation fix is not
beneficial (119.8 vs 127.7-134.4 with the upstream path), so it should stay
off unless a separate reason appears.

## 3. Warmup ablation: how much does the offline AM phase contribute?

`n_online_only` sets `behavior_warmup_updates = offline_steps + 1`, so AM never
touches an offline update and only runs during the 1M online updates (verified:
no `transport` / `control/*` columns in the offline `train.csv`).

| task | AM only online | AM from 500k (full N) | B0 | paper |
| --- | ---: | ---: | ---: | ---: |
| pen seed1 / seed2 | 123.9 / 125.7 | 133.9 / 134.5 | 117.9 / 118.6 | 151 +- 7 |
| hammer seed1 | 101.2 | 125.4 | 134.4 | 132 +- 11 |
| door seed1 / seed2 | 1.8 / 34.5 | 36.3 | 1.9 | 96 +- 4 |
| antmaze-large-play | 92.0 | 90.0 | 96.0 | 94 +- 4 |
| antmaze-medium-diverse | 90.0 | 92.0 | 98.0 | 99 +- 2 |
| antmaze-umaze | 100.0 | 98.0 | 98.0 | 99 +- 1 |

Reading: the offline AM phase adds ~9 points on pen and ~24 on hammer relative
to online-only AM, and nothing on the saturated antmaze tasks.  Online-only AM
still beats B0 on pen (+6) but is clearly worse than B0 on hammer (-33).
Note these comparisons all use the Table 7 alpha, which is itself mis-scaled.

## 4. Fair comparison at matched alpha (in progress)

Once both arms use the same retuned alpha the picture changes substantially.
Values at the same step, N (single run, W=500k) vs B0:

| run | step | N | B0 (same alpha) |
| --- | ---: | ---: | ---: |
| pen, alpha=1000 | 1.4M | 113.2 | 135.2 |
| pen, alpha=300 | 1.5M | 132.5 | 144.6 |
| door, alpha=900 | 1.3M | 0.1 | 3.1 |
| door, alpha=300 | 1.4M | -0.1 | 71.5 |
| hammer, alpha=3000 | 1.2M | 107.0 | 82.7 |
| relocate, alpha=1000 | 1.3M | 1.1 | ~0.2 |
| relocate, alpha=300 | 1.5M | 0.1 | 0.03 (final) |

The N runs finish later on 2026-09-19; this section is preliminary and must not
be quoted as a result until the 2M endpoints are in.  So far N trails B0 at
matched alpha on pen and door, leads on hammer at 1.2M, and ties on relocate.

## 5. Why a clean method advantage has not appeared: candidate causes

1. **The control strength is an implicit-scale knob (eta).**  The guided field is
   `v_pre - eta * lambda`, and `lambda` is an unnormalised endpoint-Q pullback
   whose norm is task- and state-dependent (logged at 18-30 on door).  The
   effective step is therefore `eta * |lambda|`, not `eta`.  The gated ablation,
   which clipped the adjoint to norm 1.0 (an effective 20x reduction), collapsed
   to B0 level, so the method is very sensitive to the effective control
   magnitude.  No direct eta sweep has been run yet.
2. **The critic is weak.**  This repository uses a 2-member critic ensemble and
   `normalize_q_loss=False`.  The control direction is a critic gradient, so
   critic conditioning directly limits it.  We observed the mediated quantity
   (the teacher's endpoint Q) collapsing from ~2000 to 166-442 in the failed
   configurations.
3. **The method removes the `-Q + alpha * flow` structure in its guided phase.**
   Consequences measured here: (a) the offline endpoint degrades monotonically
   with how much AM acts offline (pen: B0 81.5 -> online-only 64.3 -> full AM
   45.8), and (b) part of N's apparent benefit is likely alpha avoidance rather
   than geometric transport, because alpha does not enter the guided actor loss
   at all.
4. **Evaluation noise.**  Single-episode-level evaluation swings are 10-20
   points on the Adroit tasks (hammer 0.3-9.5 across evaluations; door seeds land
   at 1.8 and 34.5).  Detecting a 5-10 point effect needs multiple seeds and a
   fixed evaluation protocol.

## 6. References that address the same pathology, and what to adopt

Q-learning with Adjoint Matching (arXiv 2601.14234) uses a state-independent
inverse temperature `tau` in the adjoint boundary condition
`g(1) = -tau * grad Q`, clips the parameter gradient element-wise, and uses a
10-member critic ensemble.  Their own sensitivity study states that among the
analysed components "the temperature parameter has the biggest impact on
performance and needs to be tuned" (they sweep 0.1x-10x around the best value).

Trust Region Q-Adjoint Matching (arXiv 2605.27079) exists because QAM "inherits
a fundamental fragility of critic-guided improvement: small critic errors are
amplified when critics are ill-conditioned, often leading to model collapse".
It replaces the fixed temperature with a trust-region parameter optimised by
projected dual descent, using a closed-form relation between that parameter and
the path-space KL to the pretrained flow policy.

Concrete adoptions for this codebase, in order of cost:

1. Element-wise clipping of the adjoint (instead of the current
   norm-based clip that is disabled by default) - matches QAM's numerical
   stabiliser without switching the control off.
2. Increase the critic ensemble from 2 to 10 and/or enable `normalize_q_loss`;
   both change only the actor/critic conditioning, not the trajectory design.
3. Replace the global `eta` with a trust-region parameter: either normalise
   `lambda` and use a radius with action-space units, or adopt the dual-descent
   formulation.  `eta` should then mean "how far from the behaviour policy the
   controlled path may move", which is comparable across tasks.
4. Only after 1-3: re-run the matched-alpha comparison and an eta/trust-region
   sweep.

## 7. Artifacts and provenance

| host | path | contents |
| --- | --- | --- |
| A800 | `artifacts/gate_pen_hammer_v1`, `artifacts/repro8_v1` | B0 gate on pen/hammer (2 seeds) and 8 tasks (1 seed) |
| A800 | `artifacts/n_main_v1` | full AM (W=500k) on 6 tasks |
| A800 | `artifacts/n_online_only_v1` | AM only during online (W=offline+1) on 6 tasks |
| A800 | `artifacts/b0_alpha_sweep_v1` | B0 alpha sweep (8 runs) |
| 4090 | `artifacts/alpha_ablation_v1` | B0 door/pen at alpha/10 and alpha/100 |
| 4090 | `artifacts/audit_matrix_v1` | A/B/C/D normalisation x critic-LR matrix, plus a hammer control |
| 4090 | `artifacts/alpha_aligned_v1` | N and B0 at the retuned alpha (in progress) |
| 4090 | `artifacts/n_alpha_sweep_v1` | N at alpha=300 (in progress) |

Every run directory contains `launch_manifest.txt` with the commit, the tree
state, and the full flag set.
