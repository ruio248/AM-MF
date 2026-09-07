import unittest

from absl import flags
import flax
import jax
import jax.numpy as jnp
import numpy as np

from agents import agents
from agents.am_meanflow_target_changed import (
    AMMeanFlowTargetChangedAgent,
    get_config as get_changed_config,
)
from agents.meanflowql import MeanFlowQL_Agent, get_config as get_meanflowql_config
from utils.am_meanflow import (
    meanflowql_adjoint_guided_velocity,
    meanflowql_endpoint_map,
    meanflowql_reformulated_target,
)


FLAGS = flags.FLAGS
if "offline_steps" not in FLAGS:
    flags.DEFINE_integer("offline_steps", 100, "Test-only offline steps.")
if "online_steps" not in FLAGS:
    flags.DEFINE_integer("online_steps", 0, "Test-only online steps.")
if "pretrain_factor" not in FLAGS:
    flags.DEFINE_float("pretrain_factor", 0.1, "Test-only pretrain ratio.")
if not FLAGS.is_parsed():
    FLAGS(["test_am_meanflow_target_changed"])


def _tree_allclose(left, right, rtol=1e-6, atol=1e-6):
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    return len(left_leaves) == len(right_leaves) and all(
        np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)
        for a, b in zip(left_leaves, right_leaves)
    )


def _small_config(config):
    config.encoder = None
    config.actor_hidden_dims = 32
    config.actor_depth = 1
    config.actor_num_heads = 2
    config.value_hidden_dims = (32, 32)
    config.batch_size = 4
    config.num_candidates = 1
    config.action_mode = "normal"
    config.time_steps = 16
    config.alpha = 1.0
    config.use_dynamic_alpha = False
    config.consistency_alpha = 0.0
    config.bound_loss_weight = 0.0
    return config


class AMMeanFlowTargetChangedMathTest(unittest.TestCase):
    def test_meanflowql_endpoint_map_has_correct_boundaries(self):
        state = jnp.asarray([[2.0, -1.0]])
        direct_map = jnp.asarray([[0.5, 0.25]])

        np.testing.assert_allclose(
            meanflowql_endpoint_map(state, jnp.asarray([[0.0]]), direct_map),
            state,
        )
        np.testing.assert_allclose(
            meanflowql_endpoint_map(state, jnp.asarray([[1.0]]), direct_map),
            direct_map,
        )
        np.testing.assert_allclose(
            meanflowql_endpoint_map(state, jnp.asarray([[0.4]]), direct_map),
            [[1.4, -0.5]],
        )

    def test_interval_scaled_guidance_uses_action_to_noise_sign(self):
        velocity = jnp.asarray([[1.0, -2.0]])
        adjoint = jnp.asarray([[2.0, 3.0]])
        guided = meanflowql_adjoint_guided_velocity(
            velocity,
            adjoint,
            eta=0.5,
            remaining_time=jnp.asarray([[0.4]]),
        )
        np.testing.assert_allclose(guided, [[0.6, -2.6]])

    def test_reverse_path_shift_is_q_ascent(self):
        velocity = jnp.asarray([[1.0, -2.0]])
        adjoint = jnp.asarray([[2.0, 3.0]])
        eta = 0.25
        remaining_time = jnp.asarray([[0.4]])
        guided = meanflowql_adjoint_guided_velocity(
            velocity, adjoint, eta, remaining_time
        )
        base_reverse_step = -remaining_time * velocity
        guided_reverse_step = -remaining_time * guided
        q_first_order_change = jnp.sum(
            adjoint * (guided_reverse_step - base_reverse_step)
        )
        self.assertGreater(float(q_first_order_change), 0.0)

    def test_reformulated_guided_target_matches_closed_form(self):
        target = meanflowql_reformulated_target(
            state=jnp.asarray([[3.0, 4.0]]),
            target_time=jnp.asarray([[0.0]]),
            current_time=jnp.asarray([[0.5]]),
            guided_velocity=jnp.asarray([[2.0, -1.0]]),
            total_derivative=jnp.asarray([[4.0, 2.0]]),
        )
        # x + (t-1)v - t*D_t g
        np.testing.assert_allclose(target, [[0.0, 3.5]])

    def test_zero_eta_recovers_official_meanflowql_target(self):
        state = jnp.asarray([[3.0, 4.0]])
        velocity = jnp.asarray([[2.0, -1.0]])
        adjoint = jnp.asarray([[5.0, 6.0]])
        time = jnp.asarray([[0.5]])
        derivative = jnp.asarray([[4.0, 2.0]])
        guided = meanflowql_adjoint_guided_velocity(
            velocity, adjoint, eta=0.0, remaining_time=time
        )
        target = meanflowql_reformulated_target(
            state,
            jnp.zeros_like(time),
            time,
            guided,
            derivative,
        )
        expected = state + (time - 1.0) * velocity - time * derivative
        np.testing.assert_allclose(guided, velocity)
        np.testing.assert_allclose(target, expected)

    def test_negative_eta_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "eta"):
            meanflowql_adjoint_guided_velocity(
                jnp.zeros((1, 1)),
                jnp.zeros((1, 1)),
                eta=-0.1,
                remaining_time=jnp.ones((1, 1)),
            )


class AMMeanFlowTargetChangedAgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = _small_config(get_changed_config())
        generator = np.random.default_rng(17)
        cls.observations = jnp.asarray(
            generator.normal(size=(4, 6)), dtype=jnp.float32
        )
        cls.actions = jnp.asarray(
            generator.normal(size=(4, 3)), dtype=jnp.float32
        )
        cls.batch = {
            "observations": cls.observations,
            "actions": cls.actions,
            "rewards": jnp.asarray(
                generator.normal(size=(4,)), dtype=jnp.float32
            ),
            "masks": jnp.ones((4,), dtype=jnp.float32),
            "next_observations": jnp.asarray(
                generator.normal(size=(4, 6)), dtype=jnp.float32
            ),
        }

    def test_agent_is_a_meanflowql_variant_with_fixed_contract(self):
        self.assertIs(
            agents["am_meanflow_target_changed"],
            AMMeanFlowTargetChangedAgent,
        )
        self.assertTrue(
            issubclass(AMMeanFlowTargetChangedAgent, MeanFlowQL_Agent)
        )
        agent = AMMeanFlowTargetChangedAgent.create(
            0, self.observations[:1], self.actions[:1], self.config
        )
        self.assertEqual(
            agent.config["am_target_variant"],
            "meanflowql_reformulated_adjoint",
        )
        self.assertNotIn(
            "modules_target_actor_bc_flow", agent.network.params
        )

        wrong = self.config.copy_and_resolve_references()
        wrong.am_target_variant = "native_jvp_adjoint"
        with self.assertRaisesRegex(
            ValueError, "meanflowql_reformulated_adjoint"
        ):
            AMMeanFlowTargetChangedAgent.create(
                0, self.observations[:1], self.actions[:1], wrong
            )

    def test_same_seed_matches_meanflowql_network_exactly(self):
        baseline_config = _small_config(get_meanflowql_config())
        baseline = MeanFlowQL_Agent.create(
            1,
            self.observations[:1],
            self.actions[:1],
            baseline_config,
        )
        changed = AMMeanFlowTargetChangedAgent.create(
            1,
            self.observations[:1],
            self.actions[:1],
            self.config,
        )
        self.assertTrue(
            _tree_allclose(baseline.network.params, changed.network.params)
        )

    def test_eta_zero_matches_meanflowql_loss_on_same_sample(self):
        baseline_config = _small_config(get_meanflowql_config())
        baseline = MeanFlowQL_Agent.create(
            2,
            self.observations[:1],
            self.actions[:1],
            baseline_config,
        )
        changed_config = self.config.copy_and_resolve_references()
        changed_config.adjoint_eta = 0.0
        changed = AMMeanFlowTargetChangedAgent.create(
            2,
            self.observations[:1],
            self.actions[:1],
            changed_config,
        )
        rng = jax.random.PRNGKey(22)
        baseline_loss, baseline_info = baseline.meanflow_loss(
            self.batch, baseline.network.params, rng
        )
        changed_loss, changed_info = changed.meanflow_loss(
            self.batch, changed.network.params, rng
        )
        np.testing.assert_allclose(changed_loss, baseline_loss, rtol=1e-6)
        np.testing.assert_allclose(
            changed_info["mean_flow_loss"],
            baseline_info["mean_flow_loss"],
            rtol=1e-6,
        )
        self.assertAlmostEqual(
            float(changed_info["correction_norm"]), 0.0, places=7
        )
        self.assertAlmostEqual(
            float(changed_info["target_shift_norm"]), 0.0, places=7
        )

    def test_pretrain_uses_original_target_and_does_not_update_critic(self):
        agent = AMMeanFlowTargetChangedAgent.create(
            3, self.observations[:1], self.actions[:1], self.config
        )
        before = agent.network.params
        warmed, _ = agent.pretrain(self.batch, current_step=1)
        updated, info = warmed.pretrain(self.batch, current_step=2)
        after = updated.network.params

        self.assertAlmostEqual(float(info["am_enabled"]), 0.0, places=7)
        self.assertTrue(
            _tree_allclose(before["modules_critic"], after["modules_critic"])
        )
        self.assertFalse(
            _tree_allclose(
                before["modules_actor_bc_flow"],
                after["modules_actor_bc_flow"],
            )
        )

    def test_update_uses_am_reformulated_target_without_direct_q(self):
        agent = AMMeanFlowTargetChangedAgent.create(
            4, self.observations[:1], self.actions[:1], self.config
        )
        before_actor = agent.network.params["modules_actor_bc_flow"]
        warmed, _ = agent.update(self.batch, current_step=1)
        updated, info = warmed.update(self.batch, current_step=2)

        self.assertFalse(
            _tree_allclose(
                before_actor,
                updated.network.params["modules_actor_bc_flow"],
            )
        )
        self.assertAlmostEqual(
            float(info["meanflow/am_enabled"]), 1.0, places=7
        )
        self.assertAlmostEqual(float(info["actor/q_loss"]), 0.0, places=7)
        self.assertTrue(
            np.isfinite(float(info["meanflow/adjoint_norm"]))
        )
        self.assertTrue(
            np.isfinite(float(info["meanflow/target_shift_norm"]))
        )
        self.assertTrue(np.isfinite(float(info["total_loss"])))

    def test_sampling_is_the_meanflowql_direct_map(self):
        agent = AMMeanFlowTargetChangedAgent.create(
            5, self.observations[:1], self.actions[:1], self.config
        )
        sampled = agent.sample_actions(
            self.observations, seed=jax.random.PRNGKey(7)
        )
        self.assertEqual(sampled.shape, self.actions.shape)
        self.assertTrue(np.all(np.isfinite(np.asarray(sampled))))

    def test_meanflowql_conversion_and_checkpoint_round_trip(self):
        baseline_config = _small_config(get_meanflowql_config())
        baseline = MeanFlowQL_Agent.create(
            6,
            self.observations[:1],
            self.actions[:1],
            baseline_config,
        )
        changed = AMMeanFlowTargetChangedAgent.from_meanflowql_agent(
            baseline, self.config
        )
        self.assertTrue(
            _tree_allclose(changed.network.params, baseline.network.params)
        )
        self.assertEqual(
            changed.config["am_target_variant"],
            "meanflowql_reformulated_adjoint",
        )

        state = flax.serialization.to_state_dict(changed)
        fresh = AMMeanFlowTargetChangedAgent.create(
            6, self.observations[:1], self.actions[:1], self.config
        )
        restored = flax.serialization.from_state_dict(fresh, state)
        self.assertTrue(
            _tree_allclose(restored.network.params, changed.network.params)
        )


if __name__ == "__main__":
    unittest.main()
