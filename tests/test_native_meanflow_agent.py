import unittest

import jax
import jax.numpy as jnp
import numpy as np

from agents import agents
from agents.native_meanflow import NativeMeanFlowAgent, get_config


def _tree_allclose(left, right, rtol=1e-6, atol=1e-6):
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    return all(
        np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)
        for a, b in zip(left_leaves, right_leaves)
    )


class NativeMeanFlowAgentTest(unittest.TestCase):
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

    def test_agent_is_registered(self):
        self.assertIs(agents["native_meanflow"], NativeMeanFlowAgent)

    def test_create_update_and_multistep_sampling(self):
        agent = NativeMeanFlowAgent.create(
            0, self.observations[:1], self.actions[:1], self.config
        )
        updated, info = agent.update(self.batch, current_step=1)
        k1 = updated.sample_actions(
            self.observations, seed=jax.random.PRNGKey(2)
        )
        k4 = updated.sample_actions_nfe(
            self.observations,
            seed=jax.random.PRNGKey(2),
            num_steps=4,
        )

        self.assertEqual(k1.shape, self.actions.shape)
        self.assertEqual(k4.shape, self.actions.shape)
        self.assertTrue(np.all(np.isfinite(np.asarray(k1))))
        self.assertTrue(np.all(np.isfinite(np.asarray(k4))))
        self.assertTrue(np.isfinite(float(info["total_loss"])))

    def test_pretrain_changes_actor_but_not_critic(self):
        agent = NativeMeanFlowAgent.create(
            1, self.observations[:1], self.actions[:1], self.config
        )
        before = agent.network.params
        updated, _ = agent.pretrain(self.batch, current_step=1)
        after = updated.network.params

        self.assertTrue(
            _tree_allclose(
                before["modules_critic"], after["modules_critic"]
            )
        )
        self.assertTrue(
            _tree_allclose(
                before["modules_target_critic"],
                after["modules_target_critic"],
            )
        )
        self.assertFalse(
            _tree_allclose(
                before["modules_actor_bc_flow"],
                after["modules_actor_bc_flow"],
            )
        )

    def test_actor_target_ema_is_exact(self):
        agent = NativeMeanFlowAgent.create(
            2, self.observations[:1], self.actions[:1], self.config
        )
        old_target = agent.network.params["modules_target_actor_bc_flow"]
        updated, _ = agent.pretrain(self.batch, current_step=1)
        online = updated.network.params["modules_actor_bc_flow"]
        target = updated.network.params["modules_target_actor_bc_flow"]
        expected = jax.tree_util.tree_map(
            lambda new, old: (
                self.config.tau * new
                + (1.0 - self.config.tau) * old
            ),
            online,
            old_target,
        )
        self.assertTrue(_tree_allclose(target, expected))

    def test_critic_target_ema_is_exact(self):
        agent = NativeMeanFlowAgent.create(
            3, self.observations[:1], self.actions[:1], self.config
        )
        old_target = agent.network.params["modules_target_critic"]
        updated, _ = agent.update(self.batch, current_step=1)
        online = updated.network.params["modules_critic"]
        target = updated.network.params["modules_target_critic"]
        expected = jax.tree_util.tree_map(
            lambda new, old: (
                self.config.tau * new
                + (1.0 - self.config.tau) * old
            ),
            online,
            old_target,
        )
        self.assertTrue(_tree_allclose(target, expected))

    def test_best_of_n_sampling_has_expected_shape(self):
        agent = NativeMeanFlowAgent.create(
            4, self.observations[:1], self.actions[:1], self.config
        )
        sampled = agent.sample_actions(
            self.observations,
            seed=jax.random.PRNGKey(5),
            num_candidates=3,
        )
        self.assertEqual(sampled.shape, self.actions.shape)
        self.assertTrue(np.all(np.isfinite(np.asarray(sampled))))

    def test_invalid_sampling_counts_are_rejected(self):
        agent = NativeMeanFlowAgent.create(
            5, self.observations[:1], self.actions[:1], self.config
        )
        with self.assertRaisesRegex(ValueError, "num_candidates"):
            agent.sample_actions(
                self.observations,
                seed=jax.random.PRNGKey(6),
                num_candidates=0,
            )
        with self.assertRaisesRegex(ValueError, "num_steps"):
            agent.sample_actions_nfe(
                self.observations,
                seed=jax.random.PRNGKey(7),
                num_steps=0,
            )


if __name__ == "__main__":
    unittest.main()
