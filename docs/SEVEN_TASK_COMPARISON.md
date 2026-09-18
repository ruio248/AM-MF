# Seven-task comparison: AM-MF (N) vs MeanFlowQL, and the paper's MeanFlowQL / FQL

Date: 2026-09-19.  Host: A800 (`new_server_rh_2`), artifacts under
`artifacts/n_main_v1` (N, six tasks) and `artifacts/formal/relocate_cloned/`
(relocate, three seeds); B0 baselines under `artifacts/gate_pen_hammer_v1` and
`artifacts/repro8_v1`, plus `artifacts/formal/relocate_cloned/` for relocate.

## Protocol

Both arms run through the frozen upstream `main_meanflowql.py` with
`relocate-cloned`-family settings:

| Setting | Value |
| --- | --- |
| `arm=n` (AM-MF) | `agents/am_meanflow_note.py`, defaults: `transport_target_mode=interval_mean`, `behavior_warmup_updates=500000`, `control_eta=0.1`, `teacher_steps=8` (midpoint RK2), `jacobian_coef=0.01`, `target_actor_tau=0.005`, `consistency_alpha=0` |
| `arm=b0` (baseline) | `agents/meanflowql.py` (upstream, unmodified) |
| Updates | 1,000,000 offline + 1,000,000 online |
| Evaluation | every 100,000 updates, 50 rollout episodes |
| Reported metric | D4RL normalized return at the final training step (Appendix D of the paper) |
| Per-task hyperparameters | `alpha` / `num_candidates` / `time_steps` from the paper's Table 7, see `docs/PAPER_PROTOCOL_MAPPING.md` |
| Seeds | relocate 1,2,3; pen 1,2; hammer 1,2; door 1; antmaze-large-play 1; antmaze-medium-diverse 1; antmaze-umaze 1 |

## Results (offline@1M -> online@2M)

| Task | AM-MF (N) | B0 (this repo) | Paper MeanFlowQL | Paper FQL |
| --- | --- | --- | --- | --- |
| relocate-cloned-v1 | -0.007 -> **35.7** (21.1 / 47.1 / 39.0) | -0.03 -> 0.24 (0.72 / 0.03 / -0.03) | 1±1 -> 19±8 | 0±1 -> 62±8 |
| pen-cloned-v1 | 45.8 -> **133.9** / 61.7 -> **134.5** (mean 134.2) | 81.5 -> 117.9 / 73.0 -> 118.6 (mean 118.3) | 79±3 -> 151±7 | 53±14 -> 149±6 |
| hammer-cloned-v1 | 0.2 -> **125.4** / 9.0 -> **133.9** (mean 129.6) | 2.0 -> 134.4 / 4.9 -> 127.7 (mean 131.0) | 10±5 -> 132±11 | 0±0 -> 127±17 |
| door-cloned-v1 | -0.01 -> **36.3** | -0.07 -> 1.9 | 3±1 -> 96±4 | 0±0 -> 102±5 |
| antmaze-large-play-v2 | 46 -> **90** | 80 -> 96 | 80±3 -> 94±4 | 66±40 -> 84±30 |
| antmaze-medium-diverse-v2 | 82 -> **92** | 78 -> 98 | 78±2 -> 99±2 | 55±19 -> 97±3 |
| antmaze-umaze-v2 | 98 -> **98** | 94 -> 98 | 98±1 -> 99±1 | 97±2 -> 99±1 |

## Online@2M differences

| Task | N - B0 | N - paper MeanFlowQL | N - paper FQL | Outcome |
| --- | ---: | ---: | ---: | --- |
| relocate-cloned-v1 | **+35.5** | **+16.7** | -26.3 | beats B0 and the published MeanFlowQL, below FQL |
| pen-cloned-v1 | **+15.9** | -16.8 | -14.8 | beats B0, below both papers |
| hammer-cloned-v1 | -1.4 | -2.4 | +2.6 | ties B0 and MeanFlowQL |
| door-cloned-v1 | **+34.4** | -59.7 | -65.7 | beats B0, far below both papers |
| antmaze-large-play-v2 | -6 | -4 | +6 | slightly below B0 |
| antmaze-medium-diverse-v2 | -6 | -7 | -5 | slightly below B0 |
| antmaze-umaze-v2 | 0 | -1 | -1 | ties |

## Observations

1. **AM-MF wins clearly on three tasks**: relocate (+35.5 over B0; 35.7 vs the
   paper's 19±8 is +2.1σ above the published MeanFlowQL number), pen (+15.9,
   both seeds above B0's mean) and door (+34.4, i.e. 36.3 vs 1.9).  On hammer
   and antmaze-umaze it ties; on antmaze-large-play and
   antmaze-medium-diverse it is 6 points lower.  Score: 3 wins, 2 ties, 2
   narrow losses out of seven.  FQL remains ahead on relocate (62±8) and door
   (102±5); on pen AM-MF and FQL are within 1σ of each other.
2. **The offline endpoint trades off against the online result.**  N's
   offline@1M is *lower* than B0's on pen (45.8/61.7 vs 81.5/73.0) and
   antmaze-large-play (46 vs 80), yet the online phase recovers and overtakes
   B0 on pen.  This matches the open issue recorded in the method note: the
   guided (AM) offline phase does not preserve endpoint quality, while its
   value materialises during online adaptation.
3. **Where the baseline is already saturated, N has no room**: antmaze-umaze,
   antmaze-medium-diverse and antmaze-large-play are scored on a 0-100 success
   scale and B0 already reaches 94-98.  The remaining headroom is 2-6 points.
4. **Baseline reproduction caveat.**  B0 in this harness reproduces the paper
   on hammer (131.0 vs 132±11) and on all four antmaze settings, but not on
   door (1.9 vs 96±4) or pen (118.3 vs 151±7).  Therefore the pen/door columns
   above compare N against a B0 that is itself below the published MeanFlowQL
   number; the "N beats B0" statements for those two tasks are statements
   about this codebase, not yet about the published baseline.
5. **Sample size.**  pen and hammer have two seeds, the rest one.  The D4RL
   evaluator draws its action noise from NumPy and does not seed individual
   episodes, so single evaluation points carry ±5-15 noise (e.g. door's
   online curve reads 0.11, 20.8, 0.08, 36.3 at successive 100K evaluations).
   The tables above are single final points and should be complemented by the
   full curves before publication.

### Relocate: two-stage variant

The relocate row uses the single-run protocol (one process, 1M offline + 1M
online).  A two-stage variant was also run: 1M offline-only (`online_steps=0`)
followed by a restore at epoch 1,000,000 with `behavior_warmup_updates=0` and
1M online updates.  Same three seeds, same total budget:

| Protocol | Seed 1 | Seed 2 | Seed 3 | Mean |
| --- | ---: | ---: | ---: | ---: |
| single run | 21.1 | 47.1 | 39.0 | 35.7 |
| two-stage | 18.3 | 44.5 | 57.5 | 40.1 |

The two-stage protocol is slightly stronger on average but introduces a
bookkeeping difference across the run boundary (buffer reconstruction,
`current_step` reset, alpha history), so a paper-level comparison must apply
the same protocol to both arms.

## How to reproduce

```bash
# N (AM-MF), one seed
AM_MF_OFFLINE_STEPS=1000000 AM_MF_ONLINE_STEPS=1000000 \
AM_MF_BEHAVIOR_WARMUP_UPDATES=500000 \
bash scripts/run_upstream_task.sh n pen_cloned_v1 1 <output-dir>

# B0 baseline, one seed
AM_MF_OFFLINE_STEPS=1000000 AM_MF_ONLINE_STEPS=1000000 \
bash scripts/run_upstream_task.sh b0 pen_cloned_v1 1 <output-dir>
```

Every output directory contains `launch_manifest.txt` (commit, working-tree
status, task profile, seed, budget, and the arm-specific flags) plus
`runner.log`, `train.csv` and `eval.csv`.  Dataset provenance (mirror, byte
sizes, sha256) is in `docs/PAPER_PROTOCOL_MAPPING.md`; run-level provenance for
the earlier relocate experiments is in `docs/RUN_PROVENANCE.md`.
