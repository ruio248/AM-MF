# Global mean-field control: action-chunk experiments

Base: `global-mean-field-control` at `31e4b77defc97f16b6cc6229c906790b16cbfc44`.
Branch: `global-mean-field-control-chunk` (no `codex/` prefix).

This is a **new controlled protocol**, not pristine MeanFlowQL reproduction.
Rebuild H=1 for both B0 and N; historical scores must not be pooled with these runs.
The legacy entrypoint, agents, data loader and evaluator remain byte-for-byte
unchanged. Their integrity tests remain enabled. New adapters live in
`agents/chunked.py`, and the new entrypoint is `main_chunked.py`.

## Scientific question and fixed settings

Compare B0 against N separately at H = 1, 2, 5, 10 on `door-cloned-v1` and
`pen-cloned-v1`, seeds 1 and 2: 32 complete offline-to-online chains.

| Setting | Value |
| --- | --- |
| Offline budget | Exactly 1,000,000 learner updates |
| Online budget | Exactly 1,000,000 primitive environment steps |
| Online update ratio | One update after every primitive environment step |
| Batch / discount | 256 / 0.99 |
| Actor optimizer | Adam, LR peak 1e-4, offline 5% warmup + cosine, online 1e-5 |
| Critic optimizer | Adam, constant **3e-4 in both methods**, global norm clip 1 |
| Execution / target candidates | 5 / 5, explicitly static during JIT |
| Door initial alpha / time_steps | 9000 / 50 |
| Pen initial alpha / time_steps | 10000 / 100 |
| Alpha scheduling | Parent dynamic-alpha behavior and method-specific phase usage |
| Normalization | Valid original offline observations, frozen throughout training |
| Evaluation | 50 seeded episodes, every 100k updates offline / 100k env steps online |
| N warmup | Exactly the first 500k updates |
| N control | eta 0.1, midpoint RK2 with 8 teacher steps |
| N transport / Jacobian | 1.0 / 0.01; Jacobian every 100 updates, batch 4 |
| Extra pretrain / early stopping | None / disabled |

Task alpha, candidates and time discretization follow MeanFlowQL Appendix
Table 7, recorded previously in `docs/PAPER_PROTOCOL_MAPPING.md`. The learning
rate of 3e-4 deliberately follows the **implementation**, not Table 6's 1e-4.
The target candidate count of 5 preserves the legacy implementation's actual
JIT behavior, rather than silently changing it to the intended half-count of 2.
The new adapter fixes parameter handling and the explicit ensemble axis.
`temperature` remains a legacy interface argument; Gaussian noise scale is
controlled by `sigma`. No new alpha, noise or time-sampling scheme is introduced.

## Data and target contract

Store single-step transitions once. At sampling time, flatten H consecutive
actions into `[B, H*d]`, sum `gamma**k * reward[k]`, and use the observation
after the final transition. The target is:

```
y = sum(k=0..H-1, gamma**k * r[t+k])
    + gamma**H * product(masks[t:t+H]) * target_Q(s[t+H], next_action_chunk)
```

Only complete H-step windows are eligible. Episode boundaries may occur at the
last transition but never within the preceding H-1 transitions. Short tails are
not zero-padded or supervised as fictitious actions. A true terminal disables
bootstrap; a timeout ends the episode but retains the original transition mask.
The unchanged D4RL loader supplies these masks and detected episode boundaries.
The normalization mean/std are computed before replay allocation or windowing,
over the original valid observations, using float64 accumulation then float32
storage. They do not depend on H, capacity, seed, or subsequent online data.

The ring buffer keeps absolute transition IDs and episode IDs. Overwriting one
slot invalidates affected windows before adding any new complete window. A dense
valid-start pool permits uniform sampling over the combined offline/online
buffer without rejection sampling or a fixed online-data fraction. An episode
ID boundary separates the end of the offline data from the first online sample.

For an episode of length L, complete-window training provides `max(0,L-H+1)`
starts. This changes eligible state coverage with H; it is reported as a data
diagnostic, not hidden by padding. The number of supervised action coordinates
per batch also grows with H, although batch size and update counts are fixed.

## Actor, critic and N semantics

Both actors generate a whole H*d vector in one policy call, and both critics
score the whole vector. Environment action dimension and policy output dimension
are saved separately. Best-of-N ranks complete chunks with the target ensemble
mean. Critic initialization is identical across B0 and N for the same H and seed.

N retains its double-time residual map, frozen behavior prior, EMA actor, endpoint
adjoint, controlled RK2 trajectory integration, interval-average target and
Jacobian matching. Every action-space operation now acts on H*d coordinates.
The control strength and per-coordinate mean loss reductions are not rescaled
by H. Diagnostics report adjoint norm, coordinate RMS, each chunk position's
gradient norm, and eta-scaled control norm at a fixed generation time of 0.5.

Online execution and evaluation execute the entire sampled chunk open-loop,
unless the episode ends first. Learner updates occur between primitive actions,
but do not replace the remaining queued actions. A reset clears the queue.
Evaluation uses a separate environment and RNG stream, fixed episode seed lists,
and restores global NumPy/Python RNG states after evaluation. Evaluation steps
never increment the online training budget.

At each H, report `Delta_H = score(N,H) - score(B0,H)` with matched seeds, plus
`Delta_H - Delta_1`. This measures the advantage of the complete N method. Its
architecture/warmup/objective also differ from B0, so this alone does not isolate
adjoint control. `--control-eta 0` supports a later within-N control; it is not
included in the default matrix.

## Deployment and commands

The 4090 checkout is `/data/lrh/AM-MF-global-mean-field-control-chunk`.
Reuse `/home/lrh/AM-MF-4090/.venv/bin/python`; no environment installation is
needed. Data remain in `/home/lrh/d4rl_datasets`. `scripts/chunk_env.sh` wraps the
existing platform launcher and places caches on `/data`, outside the checkout.

Review the 32 formal commands without launching:

```bash
cd /data/lrh/AM-MF-global-mean-field-control-chunk
bash scripts/chunk_env.sh scripts/run_chunk_matrix.py \
  --output-root /data/lrh/am-mf-experiments/chunk-size/formal-v1 --dry-run
```

To launch only the 16 validation combinations, one seed each, on explicitly
selected GPUs (check availability before using these example IDs):

```bash
bash scripts/chunk_env.sh scripts/run_chunk_matrix.py \
  --output-root /data/lrh/am-mf-experiments/chunk-size/smoke-v1 \
  --gpus 0,1,2,3 --smoke
```

Smoke mode is explicitly recorded. It uses 40 offline updates, 20 online steps,
two evaluation episodes, actor width 32/depth 1, critic widths 32/32, batch 16,
time grid 8, N warmup 4, teacher steps 2/batch 4, Jacobian batch 2/every 4 updates.
It exercises actual adjoint updates but produces **no scientific performance
evidence**. It uses the full offline dataset and real task environments.

Each GPU lane is serial; supplied lanes run concurrently. Failed or partial run
directories are preserved and never overwritten. Use a new output root for a
retry. No formal jobs are launched by installation, tests, imports or dry runs.

Offline-only and same-H continuation use separate output directories:

```bash
bash scripts/chunk_env.sh main_chunked.py \
  --config configs/chunk/door-cloned-v1.yaml --method n --chunk-size 5 --seed 1 \
  --phase offline --output-dir /data/lrh/am-mf-experiments/chunk-size/door-n-H5-offline

bash scripts/chunk_env.sh main_chunked.py \
  --config configs/chunk/door-cloned-v1.yaml --method n --chunk-size 5 --seed 1 \
  --phase online \
  --restore /data/lrh/am-mf-experiments/chunk-size/door-n-H5-offline/checkpoints/offline_complete.pkl \
  --output-dir /data/lrh/am-mf-experiments/chunk-size/door-n-H5-online
```

The checkpoint includes all serialized agent state (optimizer, RNG, dynamic alpha
and history, N prior/EMA/counters), normalization and sampler RNG. Restore checks
task, seed, method, H/dimensions, effective agent settings, data hash and phase
boundary. It does not reset the actor, critic, optimizer, or frozen prior.
Changing H requires a fresh model. V1 intentionally rejects partial-offline and
online checkpoints for continuation: they do not contain full online replay and
simulator state. Such checkpoints can still be retained for analysis.

Summarize endpoint returns, AUC, paired gaps and PNG curves:

```bash
bash scripts/chunk_env.sh scripts/summarize_chunk_results.py \
  /data/lrh/am-mf-experiments/chunk-size/formal-v1 \
  --output-dir /data/lrh/am-mf-experiments/chunk-size/formal-v1-summary
```

Outputs are `per_seed.csv`, `summary.json`, `summary.md`, and per-task curve PNGs.
Raw AUC units are normalized-return times environment-steps. `AUC / online steps`
is the time-average score; it is not a different normalization of D4RL returns.
Standard deviation uses `ddof=1`; a single seed has no estimated seed std.
Endpoint rows must exactly match budgets. Incomplete runs are listed as excluded;
mixed commits, settings, normalization, or duplicate seeds are rejected. Smoke
runs require explicit `--include-smoke` and are prominently labeled.

Every run records source/base commits, dirty status, effective settings, dataset
path/hash/count, normalization fingerprint, evaluation seeds and device versions.
No results are reported as a process maximum or best-checkpoint score.

## Validation

```bash
JAX_PLATFORMS=cpu bash scripts/chunk_env.sh -m unittest discover -s tests -v
```

Tests cover legacy integrity and N behavior, lazy-window arithmetic and boundaries,
ring overwrite vs brute-force indexing, normalization invariance, B0/N critic
parity, static candidate counts, N snapshots/gradients, primitive-step budgets,
evaluation RNG isolation, state-complete offline restore/next-update equivalence,
matrix size, paired comparisons and AUC. The real-environment validation report
is recorded separately in [CHUNK_VALIDATION.md](CHUNK_VALIDATION.md): all 16 smoke
combinations passed, with a separate exact-state offline continuation check.

The production-dimension numerical check is opt-in and needs an available GPU:

```bash
CUDA_VISIBLE_DEVICES=0 AM_MF_FULL_MODEL_TEST=1 \
  bash scripts/chunk_env.sh -m unittest discover \
  -s tests -p 'test_chunk_production_shape.py' -v
```

It uses synthetic data with the real door dimensions and H=10, batch 256, the
production network/teacher sizes, and only shortens warmup to enter the N path.
It is not a learning-performance or long-run stability test.
