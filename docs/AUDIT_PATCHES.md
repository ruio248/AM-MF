# Audit patches (`audit-fixes` branch)

This branch is **audit-patched upstream**, not pristine upstream.  It exists to
run the normalisation / critic-learning-rate causal matrix described below.
Every flag introduced here defaults to the pristine upstream behaviour, so a run
started without the audit flags behaves like the original code path.

**Rule:** results produced from this branch must be labelled `audit-patched` and
must not be mixed into a table of pristine-upstream results.

## Patch 1 - observation normalisation statistics over valid transitions only

`ReplayBuffer.create_from_initial_dataset` allocates `buffer_size` rows and
leaves every row beyond the valid dataset zero-filled.  Upstream computes the
normalisation statistics over the whole array, so the zero rows enter the mean
and standard deviation:

```
mean_padded = p * mu                      with p = N / buffer_size
var_padded  = p * sigma^2 + p (1-p) mu^2
```

With the launcher default `buffer_size = 2_000_000` the contamination is large:

| dataset | valid N | p | mean shift | std shift |
| --- | ---: | ---: | ---: | ---: |
| pen-cloned-v1 | 500,000 | 0.25 | 75% | 59% |
| door-cloned-v1 | 1,000,000 | 0.50 | 50% | 21% |
| hammer-cloned-v1 | 1,000,000 | 0.50 | 50% | 48% |
| relocate-cloned-v1 | 1,000,000 | 0.50 | 50% | 21% |

The shift is largest on observation dimensions whose offset dominates their
spread (hand joint angles / object positions).  Example (pen, dimension 26):
`mu=+0.214, sigma=0.016` becomes `mu=+0.054, sigma=0.093`, so an observation
that should normalise to 0 normalises to +1.73.

Note this only affects the `balanced_sampling=0` path, because that is the path
that replaces `train_dataset` with the padded replay buffer.

```diff
--- a/utils/datasets.py
+++ b/utils/datasets.py
-    def compute_normalization_stats(self):
-        """Compute mean and standard deviation of observations for normalization."""
-        observations = self['observations']
+    def compute_normalization_stats(self, valid_only=False):
+        observations = self['observations']
+        if valid_only:
+            observations = observations[: self.size]
         self.obs_mean = np.mean(observations, axis=0, keepdims=True)
```

```diff
--- a/main_meanflowql.py
+++ b/main_meanflowql.py
+flags.DEFINE_boolean(
+    'strict_norm_stats', False,
+    'Compute observation normalisation statistics over valid transitions only.'
+)
@@
-            train_dataset.compute_normalization_stats()
+            train_dataset.compute_normalization_stats(
+                valid_only=FLAGS.strict_norm_stats
+            )
```

## Patch 2 - configurable critic learning rate

Upstream hard-codes `3e-4` for the critic, while the paper's Table 6 lists
`1e-4` (the same value the protocol-mapping document quotes).  The config key
`lr` only drives the actor.

```diff
--- a/agents/meanflowql.py
+++ b/agents/meanflowql.py
-        critic_lr_schedule = lambda _: 3e-4
+        critic_lr = config.get('critic_lr', 3e-4)
+        critic_lr_schedule = lambda _: critic_lr
@@ get_config()
+            critic_lr=3e-4,  # Upstream hard-codes 3e-4; paper Table 6 lists 1e-4
```

## Flags exposed by the launcher

| env var | flag | default (upstream) |
| --- | --- | --- |
| `AM_MF_STRICT_NORM_STATS` | `--strict_norm_stats` | `False` |
| `AM_MF_CRITIC_LR` | `--agent.critic_lr` | `3e-4` |

Both values are written into `launch_manifest.txt` for every run.

## File hashes

| file | pristine upstream | audit-patched |
| --- | --- | --- |
| `main_meanflowql.py` | `026a0ced1e5346d40b729827e0d1921cdc55d73049118a737ecfa0bae76a6fd7` | `ba6d9adc063041914b0b9966f0071672ee112285a6e9d270ee60cd46ca31b5ff` |
| `agents/meanflowql.py` | `50525c5ada2462ce47981390f89fefa5c71afad67891e61579a4e94ac0307b8c` | `ef8202d6041246fd458fad1c043b9e8df28cc0649c51de8b0380aff05afedcfd` |
| `utils/datasets.py` | `0a39d5c9da3b3ba3c37200abf451f3bbcb14a54d7b07872e25d8f9e7c28b40fa` | `f04914c4886252e48890ae53ac5526c98b6a291a8fee8cc9df6480ad047bac8e` |

The remaining five files in `tests/test_upstream_integrity.py` keep their
pristine hashes.  The test also asserts that the three audit patches are present
and opt-in.

## Known deviations that are documented but NOT patched here

1. **Critic initialiser.** The paper's Table 6 lists `kaiming_init` for the
   critic, while `utils/networks.py` uses
   `default_init() = variance_scaling(1, 'fan_avg', 'uniform')`.  Changing it
   also changes the actor (shared helper), so it is deferred to a second round
   to keep the first causal matrix to two factors.
2. **Actor learning rate during online.** The phase-aware schedule keeps the
   actor at `lr * lr_min_ratio = 1e-5` after the offline phase while the critic
   runs at `3e-4`.
3. **`balanced_sampling=1` is broken upstream.** With that flag `train_dataset`
   stays a `Dataset` (a `FrozenDict`) instead of a `ReplayBuffer`, so the online
   batch concatenation fails with a broadcast error.  The official FQL
   repository contains the same code path.
4. **Evaluation seeds.** Environment creation and `env.reset()` happen without a
   per-episode seed and the evaluator draws its action noise from the global
   NumPy RNG, so evaluation is not reproducible episode-by-episode.

## Causal matrix run from this branch

| arm | `strict_norm_stats` | `critic_lr` |
| --- | --- | --- |
| A (control) | False | 3e-4 |
| B | True | 3e-4 |
| C | False | 1e-4 |
| D | True | 1e-4 |

Four runs on `door-cloned-v1`, one seed, 1M offline + 1M online, everything else
at the Table 7 / upstream defaults.
