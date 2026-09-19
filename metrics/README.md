# Metric and manifest archive

This directory is a compact, verifiable copy of the runs discussed in
`docs/DIAGNOSTICS_2026-09-19.md`.  Only `eval.csv`, `train.csv` and
`launch_manifest.txt` are included (a few MB in total); the full per-run
`runner.log` files (30-50 MB each) and the model checkpoints stay on the
machines.

Layout: `metrics/<host>/<run-group>/<run>/{eval.csv,train.csv,launch_manifest.txt}`.

| host | run group | contents |
| --- | --- | --- |
| a800 | `gate_pen_hammer_v1` | B0 pen/hammer, 2 seeds each, alpha from Table 7 |
| a800 | `repro8_v1` | B0 on 8 tasks, 1 seed each, alpha from Table 7 |
| a800 | `n_main_v1` | full AM (W=500k) on 6 tasks |
| a800 | `n_online_only_v1` | AM active only during online (W=offline+1) on 6 tasks |
| a800 | `b0_alpha_sweep_v1` | B0 alpha sweep: door/pen/relocate at 300 and 3000, hammer at 3000 and 33000 |
| gpu4090 | `alpha_ablation_v1` | B0 door alpha=900/90, pen alpha=1000 (the runs that reproduced door/pen) |
| gpu4090 | `audit_matrix_v1` | door: {strict_norm_stats x critic_lr} 2x2, plus hammer with the normalisation fix |
| gpu4090 | `alpha_aligned_v1` | N and B0 at the retuned alpha (N door 900, N pen 1000, B0/N relocate 1000) |
| gpu4090 | `n_alpha_sweep_v1` | N at alpha=300 (door, pen, relocate) and N hammer at alpha=3000 |

`eval.csv` is written by the frozen upstream evaluator: one row per evaluation
(every 100K updates), with `evaluation/episode.normalized_return` in the D4RL
normalized-return scale (for antmaze this equals the success percentage) and
`step` the gradient-step index.  `train.csv` carries the per-100K-phase
diagnostics, including `training/actor/q`, `training/critic/q_mean`,
`training/meanflow/flow_loss`, `training/alpha_weight`, and - in the AM arms -
`training/actor/transport`, `training/control/adjoint_norm`,
`training/control/eta` and `training/teacher/q`.
