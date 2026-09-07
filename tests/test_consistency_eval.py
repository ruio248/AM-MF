import unittest

from absl import flags
import jax
import jax.numpy as jnp
import numpy as np

from agents.am_meanflow_target_changed import (
    AMMeanFlowTargetChangedAgent,
    get_config as get_changed_config,
)
from utils.consistency_eval import (
    ConsistencyThresholds,
    default_split_triplets,
    evaluate_agent_consistency,
    evaluate_meanflowql_consistency,
    evaluate_native_meanflow_consistency,
    judge_consistency,
    meanflowql_endpoint,
    meanflowql_reverse_rollout,
    native_endpoint,
    native_reverse_rollout,
)


FLAGS = flags.FLAGS
if "offline_steps" not in FLAGS:
    flags.DEFINE_integer("offline_steps", 100, "Test-only offline steps.")
if "online_steps" not in FLAGS:
    flags.DEFINE_integer("online_steps", 0, "Test-only online steps.")
if "pretrain_factor" not in FLAGS:
    flags.DEFINE_float("pretrain_factor", 0.1, "Test-only pretrain ratio.")
if not FLAGS.is_parsed():
    FLAGS(["test_consistency_eval"])


class ConsistencyEvaluationMathTest(unittest.TestCase):
    def setUp(self):
        self.observations = jnp.zeros((4, 3), dtype=jnp.float32)
        self.noises = jnp.asarray(
            [
                [1.0, -0.5],
                [0.4, 0.8],
                [-0.3, 0.6],
                [0.7, -0.2],
            ],
            dtype=jnp.float32,
        )

    def test_native_constant_field_passes_all_structural_checks(self):
        velocity = jnp.asarray([0.25, -0.4])

        def velocity_fn(observations, state, target_time, current_time):
            del observations, target_time, current_time
            return jnp.broadcast_to(velocity, state.shape)

        metrics = evaluate_native_meanflow_consistency(
            velocity_fn,
            self.observations,
            self.noises,
            nfe_values=(1, 2, 4),
            trajectory_steps=4,
            jacobian_probe_pairs=4,
            split_triplets=((0, 2, 4), (1, 2, 3)),
            rng=jax.random.PRNGKey(1),
        )

        self.assertEqual(metrics["family"], "native_meanflow")
        self.assertAlmostEqual(metrics["k1_k2_mse"], 0.0, places=7)
        self.assertAlmostEqual(metrics["k1_k4_mse"], 0.0, places=7)
        self.assertAlmostEqual(metrics["endpoint_map_mse"], 0.0, places=7)
        self.assertAlmostEqual(metrics["endpoint_jvp_mse"], 0.0, places=7)
        self.assertAlmostEqual(
            metrics["split_consistency_mse"], 0.0, places=7
        )
        self.assertAlmostEqual(
            metrics["trajectory_path_length_ratio"], 1.0, places=6
        )

    def test_meanflowql_exact_endpoint_map_passes_common_checks(self):
        expected_endpoint = jnp.asarray([0.2, -0.3])

        def direct_map_fn(observations, state, current_time):
            del observations
            endpoint = jnp.broadcast_to(expected_endpoint, state.shape)
            return (
                endpoint - (1.0 - current_time) * state
            ) / current_time

        metrics = evaluate_meanflowql_consistency(
            direct_map_fn,
            self.observations,
            self.noises,
            nfe_values=(1, 3, 5),
            trajectory_steps=5,
            jacobian_probe_pairs=4,
            rng=jax.random.PRNGKey(2),
        )

        self.assertEqual(metrics["family"], "meanflowql")
        self.assertNotIn("split_consistency_mse", metrics)
        self.assertAlmostEqual(metrics["k1_k3_mse"], 0.0, places=7)
        self.assertAlmostEqual(metrics["k1_k5_mse"], 0.0, places=7)
        self.assertAlmostEqual(metrics["endpoint_map_mse"], 0.0, places=7)
        self.assertAlmostEqual(metrics["endpoint_jvp_mse"], 0.0, places=7)

    def test_inconsistent_direct_map_produces_nonzero_errors(self):
        def direct_map_fn(observations, state, current_time):
            del observations
            return 0.5 * state + 0.2 * current_time

        metrics = evaluate_meanflowql_consistency(
            direct_map_fn,
            self.observations,
            self.noises,
            nfe_values=(1, 4),
            trajectory_steps=4,
            jacobian_probe_pairs=4,
            rng=jax.random.PRNGKey(3),
        )
        batched_metrics = evaluate_meanflowql_consistency(
            direct_map_fn,
            self.observations,
            self.noises,
            nfe_values=(1, 4),
            trajectory_steps=4,
            jacobian_probe_pairs=4,
            inference_batch_size=2,
            rng=jax.random.PRNGKey(3),
        )

        self.assertGreater(metrics["k1_k4_mse"], 0.0)
        self.assertGreater(metrics["endpoint_map_mse"], 0.0)
        self.assertGreater(metrics["endpoint_jvp_mse"], 0.0)
        for key in (
            "k1_k4_mse",
            "endpoint_map_mse",
            "endpoint_jvp_mse",
        ):
            self.assertAlmostEqual(metrics[key], batched_metrics[key], places=7)

    def test_endpoint_and_rollout_helpers_respect_time_orientation(self):
        def velocity_fn(observations, state, target_time, current_time):
            del observations, target_time, current_time
            return jnp.ones_like(state) * 0.5

        time = jnp.ones((4, 1))
        expected = self.noises - 0.5
        np.testing.assert_allclose(
            native_endpoint(
                velocity_fn, self.observations, self.noises, time
            ),
            expected,
        )
        final, states = native_reverse_rollout(
            velocity_fn, self.observations, self.noises, 4
        )
        np.testing.assert_allclose(final, expected)
        np.testing.assert_allclose(states[0], expected)
        np.testing.assert_allclose(states[-1], self.noises)

        def direct_map_fn(observations, state, current_time):
            del observations
            return -(1.0 - current_time) * state / current_time

        np.testing.assert_allclose(
            meanflowql_endpoint(
                direct_map_fn, self.observations, self.noises, time
            ),
            0.0,
        )
        direct_final, direct_states = meanflowql_reverse_rollout(
            direct_map_fn, self.observations, self.noises, 4
        )
        np.testing.assert_allclose(direct_final, 0.0, atol=1e-7)
        np.testing.assert_allclose(direct_states[0], direct_final)
        np.testing.assert_allclose(direct_states[-1], self.noises)

    def test_invalid_shapes_and_triplets_are_rejected(self):
        def velocity_fn(observations, state, target_time, current_time):
            del observations, target_time, current_time
            return jnp.zeros_like(state)

        with self.assertRaisesRegex(ValueError, "leading dimension"):
            evaluate_native_meanflow_consistency(
                velocity_fn,
                self.observations[:2],
                self.noises,
                trajectory_steps=2,
                split_triplets=((0, 1, 2),),
            )
        with self.assertRaisesRegex(ValueError, "0 <= r < s < t"):
            evaluate_native_meanflow_consistency(
                velocity_fn,
                self.observations,
                self.noises,
                trajectory_steps=2,
                split_triplets=((0, 2, 1),),
            )

    def test_default_split_triplets_scale_to_requested_grid(self):
        self.assertEqual(default_split_triplets(2), ((0, 1, 2),))
        self.assertIn((0, 5, 10), default_split_triplets(10))
        with self.assertRaisesRegex(ValueError, ">= 2"):
            default_split_triplets(1)


class ConsistencyJudgementTest(unittest.TestCase):
    def test_explicit_thresholds_control_binary_judgement(self):
        metrics = {
            "endpoint_map_nmse": 0.02,
            "endpoint_jvp_nmse": 0.03,
            "split_consistency_nmse": 0.04,
            "k1_k2_mse": 0.05,
            "k1_k10_mse": 0.06,
        }
        passing = judge_consistency(
            metrics,
            ConsistencyThresholds(
                endpoint_map_nmse=0.02,
                endpoint_jvp_nmse=0.04,
                split_consistency_nmse=0.05,
                k1_k_reference_mse=0.07,
            ),
        )
        failing = judge_consistency(
            metrics,
            ConsistencyThresholds(endpoint_map_nmse=0.01),
        )

        self.assertEqual(passing["status"], "passed")
        self.assertTrue(passing["passed"])
        self.assertIn("k1_k10_mse", passing["checks"])
        self.assertEqual(failing["status"], "failed")
        self.assertFalse(failing["passed"])

    def test_no_thresholds_means_not_requested(self):
        result = judge_consistency({}, ConsistencyThresholds())
        self.assertEqual(result["status"], "not_requested")
        self.assertIsNone(result["passed"])

    def test_unavailable_family_metric_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unavailable"):
            judge_consistency(
                {"endpoint_map_nmse": 0.0},
                ConsistencyThresholds(split_consistency_nmse=0.1),
            )


class ConsistencyAgentAdapterTest(unittest.TestCase):
    def test_native_and_changed_target_agents_dispatch_separately(self):
        class FakeNativeAgent:
            config = {"agent_name": "am_meanflow"}

            def _velocity(
                self,
                observations,
                state,
                target_time,
                current_time,
                target=False,
            ):
                del observations, target_time, current_time, target
                return jnp.ones_like(state) * 0.1

        class FakeNetwork:
            def select(self, name):
                self.name = name

                def direct_map(observations, state, current_time):
                    del observations, current_time
                    return jnp.zeros_like(state)

                return direct_map

        class FakeChangedTargetAgent:
            config = {
                "agent_name": "am_meanflow_target_changed",
                "encoder": None,
            }
            network = FakeNetwork()

        native_metrics = evaluate_agent_consistency(
            FakeNativeAgent(),
            jnp.zeros((2, 3)),
            jnp.ones((2, 2)),
            nfe_values=(1, 2),
            trajectory_steps=2,
            jacobian_probe_pairs=1,
            split_triplets=((0, 1, 2),),
        )
        changed_metrics = evaluate_agent_consistency(
            FakeChangedTargetAgent(),
            jnp.zeros((2, 3)),
            jnp.ones((2, 2)),
            nfe_values=(1, 2),
            trajectory_steps=2,
            jacobian_probe_pairs=1,
        )

        self.assertEqual(native_metrics["family"], "native_meanflow")
        self.assertIn("split_consistency_mse", native_metrics)
        self.assertEqual(changed_metrics["family"], "meanflowql")
        self.assertNotIn("split_consistency_mse", changed_metrics)
        with self.assertRaisesRegex(ValueError, "no EMA target actor"):
            evaluate_agent_consistency(
                FakeChangedTargetAgent(),
                jnp.zeros((2, 3)),
                jnp.ones((2, 2)),
                nfe_values=(1, 2),
                trajectory_steps=2,
                jacobian_probe_pairs=1,
                use_target_actor=True,
            )

    def test_real_changed_target_agent_produces_finite_diagnostics(self):
        config = get_changed_config()
        config.encoder = None
        config.actor_hidden_dims = 16
        config.actor_depth = 1
        config.actor_num_heads = 2
        config.value_hidden_dims = (16, 16)
        config.batch_size = 2
        config.num_candidates = 1
        config.action_mode = "normal"
        config.time_steps = 4
        config.use_dynamic_alpha = False
        observations = jnp.zeros((2, 3), dtype=jnp.float32)
        actions = jnp.zeros((2, 2), dtype=jnp.float32)
        agent = AMMeanFlowTargetChangedAgent.create(
            9, observations[:1], actions[:1], config
        )

        metrics = evaluate_agent_consistency(
            agent,
            observations,
            jnp.ones_like(actions),
            nfe_values=(1, 2),
            trajectory_steps=2,
            jacobian_probe_pairs=1,
            rng=jax.random.PRNGKey(10),
        )
        target_metrics = evaluate_agent_consistency(
            agent,
            observations,
            jnp.ones_like(actions),
            nfe_values=(1, 2),
            trajectory_steps=2,
            jacobian_probe_pairs=1,
            use_target_actor=True,
            rng=jax.random.PRNGKey(10),
        )

        self.assertEqual(metrics["agent_name"], "am_meanflow_target_changed")
        self.assertEqual(metrics["family"], "meanflowql")
        self.assertEqual(metrics["use_target_actor"], 0)
        self.assertEqual(target_metrics["use_target_actor"], 1)
        self.assertNotIn("split_consistency_mse", metrics)
        for key in (
            "k1_k2_mse",
            "endpoint_map_mse",
            "endpoint_jvp_mse",
        ):
            self.assertTrue(np.isfinite(float(metrics[key])))
            self.assertTrue(np.isfinite(float(target_metrics[key])))


if __name__ == "__main__":
    unittest.main()
