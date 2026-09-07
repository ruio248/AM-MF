import unittest

import flax
import jax
import jax.numpy as jnp
import numpy as np

from agents import agents
from agents.am_meanflow import AMMeanFlowAgent, get_config
from agents.native_meanflow import (
    NativeMeanFlowAgent,
    get_config as get_native_config,
)


def _tree_allclose(left, right, rtol=1e-6, atol=1e-6):
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    return len(left_leaves) == len(right_leaves) and all(
        np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)
        for a, b in zip(left_leaves, right_leaves)
    )


class AMMeanFlowAgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = get_config()
        config.encoder = None
        config.actor_hidden_dims = 32
        config.actor_depth = 1
        config.actor_num_heads = 2
        config.value_hidden_dims = (32, 32)
        config.batch_size = 4
        config.num_candidates = 1
        config.alpha_mode = "fixed"
        config.alpha_value = 0.5
        config.critic_coef = 1.0
        config.native_regularizer_coef = 0.0
        cls.config = config
        cls.observations = jnp.zeros((4, 6), dtype=jnp.float32)
        cls.actions = jnp.zeros((4, 3), dtype=jnp.float32)
        cls.batch = {
            "observations": cls.observations,
            "actions": cls.actions,
            "rewards": jnp.zeros((4,), dtype=jnp.float32),
            "masks": jnp.ones((4,), dtype=jnp.float32),
            "next_observations": cls.observations,
        }

    def test_agent_is_registered_and_roles_start_synchronized(self):
        self.assertIs(agents["am_meanflow"], AMMeanFlowAgent)
        agent = AMMeanFlowAgent.create(
            0, self.observations[:1], self.actions[:1], self.config
        )
        params = agent.network.params
        self.assertTrue(
            _tree_allclose(
                agent.pre_actor_params,
                params["modules_actor_bc_flow"],
            )
        )
        self.assertTrue(
            _tree_allclose(
                params["modules_target_actor_bc_flow"],
                params["modules_actor_bc_flow"],
            )
        )
        self.assertEqual(agent.config["am_target_variant"], "note_average")

        wrong_config = self.config.copy_and_resolve_references()
        wrong_config.am_target_variant = "local_velocity"
        with self.assertRaisesRegex(ValueError, "note_average"):
            AMMeanFlowAgent.create(
                0,
                self.observations[:1],
                self.actions[:1],
                wrong_config,
            )

    def test_update_freezes_pre_actor_and_ema_updates_targets(self):
        agent = AMMeanFlowAgent.create(
            1, self.observations[:1], self.actions[:1], self.config
        )
        before = agent.network.params
        frozen_pre = agent.pre_actor_params
        old_actor_target = before["modules_target_actor_bc_flow"]
        old_critic_target = before["modules_target_critic"]

        updated, info = agent.update(self.batch, current_step=100)
        after = updated.network.params

        self.assertTrue(_tree_allclose(frozen_pre, updated.pre_actor_params))
        self.assertFalse(
            _tree_allclose(
                before["modules_actor_bc_flow"],
                after["modules_actor_bc_flow"],
            )
        )
        expected_actor_target = jax.tree_util.tree_map(
            lambda online, target: (
                self.config.tau * online
                + (1.0 - self.config.tau) * target
            ),
            after["modules_actor_bc_flow"],
            old_actor_target,
        )
        expected_critic_target = jax.tree_util.tree_map(
            lambda online, target: (
                self.config.tau * online
                + (1.0 - self.config.tau) * target
            ),
            after["modules_critic"],
            old_critic_target,
        )
        self.assertTrue(
            _tree_allclose(
                after["modules_target_actor_bc_flow"],
                expected_actor_target,
            )
        )
        self.assertTrue(
            _tree_allclose(
                after["modules_target_critic"], expected_critic_target
            )
        )
        self.assertEqual(int(updated.am_updates), 1)
        self.assertTrue(np.isfinite(float(info["total_loss"])))
        self.assertTrue(np.isfinite(float(info["am/adjoint_norm"])))

    def test_pretrain_hard_syncs_online_pre_and_target_actor(self):
        agent = AMMeanFlowAgent.create(
            2, self.observations[:1], self.actions[:1], self.config
        )
        before_critic = agent.network.params["modules_critic"]
        updated, info = agent.pretrain(self.batch, current_step=1)
        params = updated.network.params

        self.assertTrue(
            _tree_allclose(
                before_critic, params["modules_critic"]
            )
        )
        self.assertTrue(
            _tree_allclose(
                params["modules_actor_bc_flow"],
                params["modules_target_actor_bc_flow"],
            )
        )
        self.assertTrue(
            _tree_allclose(
                params["modules_actor_bc_flow"], updated.pre_actor_params
            )
        )
        self.assertEqual(int(updated.am_updates), 0)
        self.assertTrue(np.isfinite(float(info["mean_flow_loss"])))

    def test_from_native_agent_resets_all_target_roles(self):
        native_config = get_native_config()
        native_config.encoder = None
        native_config.actor_hidden_dims = 32
        native_config.actor_depth = 1
        native_config.actor_num_heads = 2
        native_config.value_hidden_dims = (32, 32)
        native = NativeMeanFlowAgent.create(
            3,
            self.observations[:1],
            self.actions[:1],
            native_config,
        )
        native_params = dict(native.network.params)
        native_params["modules_target_actor_bc_flow"] = (
            jax.tree_util.tree_map(
                lambda value: value + 1.0,
                native_params["modules_target_actor_bc_flow"],
            )
        )
        native_params["modules_target_critic"] = jax.tree_util.tree_map(
            lambda value: value - 1.0,
            native_params["modules_target_critic"],
        )
        native = native.replace(
            network=native.network.replace(params=native_params)
        )

        am = AMMeanFlowAgent.from_native_agent(
            native,
            {
                "alpha_mode": "fixed",
                "alpha_value": 0.4,
            },
        )
        params = am.network.params
        self.assertTrue(
            _tree_allclose(
                params["modules_actor_bc_flow"], am.pre_actor_params
            )
        )
        self.assertTrue(
            _tree_allclose(
                params["modules_actor_bc_flow"],
                params["modules_target_actor_bc_flow"],
            )
        )
        self.assertTrue(
            _tree_allclose(
                params["modules_critic"], params["modules_target_critic"]
            )
        )
        self.assertAlmostEqual(float(am.current_alpha), 0.4, places=6)
        self.assertEqual(int(am.am_updates), 0)

    def test_zero_alpha_uses_jvp_and_note_correction_vanishes(self):
        config = self.config.copy_and_resolve_references()
        config.alpha_value = 0.0
        agent = AMMeanFlowAgent.create(
            4, self.observations[:1], self.actions[:1], config
        )
        updated, info = agent.update(self.batch, current_step=1)

        self.assertAlmostEqual(float(info["am/jvp_branch"]), 1.0, places=7)
        self.assertAlmostEqual(
            float(info["am/correction_norm"]), 0.0, places=7
        )
        self.assertTrue(np.isfinite(float(info["am/jvp_loss"])))
        self.assertTrue(np.isfinite(float(info["total_loss"])))
        self.assertTrue(_tree_allclose(agent.pre_actor_params, updated.pre_actor_params))

    def test_am_checkpoint_state_round_trip_preserves_stage_roles(self):
        template = AMMeanFlowAgent.create(
            5, self.observations[:1], self.actions[:1], self.config
        )
        trained, _ = template.update(self.batch, current_step=1)
        state = flax.serialization.to_state_dict(trained)

        fresh = AMMeanFlowAgent.create(
            5, self.observations[:1], self.actions[:1], self.config
        )
        restored = flax.serialization.from_state_dict(fresh, state)

        self.assertTrue(
            _tree_allclose(restored.network.params, trained.network.params)
        )
        self.assertTrue(
            _tree_allclose(restored.pre_actor_params, trained.pre_actor_params)
        )
        self.assertEqual(int(restored.am_updates), int(trained.am_updates))
        self.assertAlmostEqual(
            float(restored.current_alpha),
            float(trained.current_alpha),
            places=7,
        )

    def test_sampling_interfaces_are_inherited(self):
        agent = AMMeanFlowAgent.create(
            6, self.observations[:1], self.actions[:1], self.config
        )
        k1 = agent.sample_actions(
            self.observations, seed=jax.random.PRNGKey(7)
        )
        k3 = agent.sample_actions_nfe(
            self.observations,
            seed=jax.random.PRNGKey(7),
            num_steps=3,
        )
        self.assertEqual(k1.shape, self.actions.shape)
        self.assertEqual(k3.shape, self.actions.shape)
        self.assertTrue(np.all(np.isfinite(np.asarray(k1))))
        self.assertTrue(np.all(np.isfinite(np.asarray(k3))))


if __name__ == "__main__":
    unittest.main()
