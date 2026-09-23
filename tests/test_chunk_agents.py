import unittest

import jax
import jax.numpy as jnp
import numpy as np

from agents.chunked import create_agent, make_config


def tiny_agent(method, horizon=2, seed=1, critic_lr=3e-4):
    overrides = dict(
        actor_hidden_dims=16, actor_depth=1, actor_num_heads=2,
        value_hidden_dims=[16, 16], batch_size=4, alpha=1.0, time_steps=8,
        critic_lr=critic_lr, num_candidates=5, target_num_candidates=2,
        use_dynamic_alpha=False,
    )
    if method == "n":
        overrides.update(behavior_warmup_updates=2, teacher_steps=2,
                         teacher_batch_size=2, jacobian_batch_size=1, jacobian_interval=2)
    config = make_config(method, overrides, horizon, 2)
    observations = jnp.array([[0.1, 0.2, 0.3], [0.2, -0.5, 0.1],
                              [0.3, 0.1, -0.2], [-0.5, 0.2, 0.4]])
    actions = jnp.linspace(-0.5, 0.5, 4 * horizon * 2).reshape(4, -1)
    agent = create_agent(method, seed, observations[:1], actions[:1], config, 4, 4)
    return agent, dict(observations=observations, actions=actions,
                       rewards=jnp.arange(4.0), next_observations=observations + 0.01,
                       masks=jnp.array([1.0, 1.0, 1.0, 0.0]))


def assert_tree_equal(case, left, right, atol=0):
    case.assertEqual(jax.tree_util.tree_structure(left), jax.tree_util.tree_structure(right))
    for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
        np.testing.assert_allclose(a, b, rtol=0, atol=atol)


class ChunkAgentTest(unittest.TestCase):
    def test_shared_critic_initialization_lr_and_chunk_shapes(self):
        for horizon in (1, 2, 5, 10):
            b0, batch = tiny_agent("b0", horizon, critic_lr=7e-5)
            note, _ = tiny_agent("n", horizon, critic_lr=7e-5)
            for name in ("modules_critic", "modules_target_critic"):
                assert_tree_equal(self, b0.network.params[name], note.network.params[name])
            for agent in (b0, note):
                self.assertAlmostEqual(float(agent.config["critic_lr_schedule"](100)), 7e-5)
                action = agent.sample_actions(batch["observations"], seed=jax.random.PRNGKey(7))
                self.assertEqual(action.shape, (4, horizon * 2))
                _, info = agent.critic_loss(batch, agent.network.params, jax.random.PRNGKey(7))
                self.assertAlmostEqual(float(info["bootstrap_discount"]), 0.99 ** horizon, places=6)
                _, draw = jax.random.split(jax.random.PRNGKey(7))
                next_action = agent.sample_actions(batch["next_observations"], seed=draw, num_candidates=2)
                next_q = agent.network.select("target_critic")(
                    batch["next_observations"], actions=next_action).mean(axis=0)
                target = batch["rewards"] + (0.99 ** horizon) * batch["masks"] * next_q
                q = agent.network.select("critic")(batch["observations"], actions=batch["actions"])
                np.testing.assert_allclose(info["critic_loss"], jnp.mean((q - target) ** 2), rtol=1e-6)
                self.assertEqual(float(target[-1]), float(batch["rewards"][-1]))

    def test_static_candidates_match_explicit_reference(self):
        agent, batch = tiny_agent("b0")
        # Initialize nonzero actor outputs with training before comparing draws.
        for step in range(2):
            agent, _ = agent.update(batch, step)
        observations = batch["observations"][:1]
        seed = jax.random.PRNGKey(43)
        for count in (1, 2, 5):
            @jax.jit
            def reference(model, observations, seed):
                # Compile the whole independent reference, matching production's
                # GPU fusion/rounding, not a sequence of eager per-op kernels.
                action_seed, _ = jax.random.split(seed)
                noises = jax.vmap(lambda key: model.sample_noise(key, (1, 4)))(
                    jax.random.split(action_seed, count)
                ).reshape(count, 4)
                obs = jnp.repeat(observations, count, axis=0)
                actions = jnp.clip(model.network.select("actor_bc_flow")(
                    obs, noises, jnp.ones((count, 1))), -1, 1)
                scores = model.network.select("target_critic")(obs, actions=actions).mean(axis=0)
                index = jnp.argmax(scores)
                return actions[index][None], index, scores, actions

            expected, index, scores, candidates = reference(agent, observations, seed)
            self.assertEqual(candidates.shape, (count, 4))
            self.assertEqual(int(index), int(jnp.argmax(scores)))
            actual = agent.sample_actions(observations, seed=seed, num_candidates=count)
            np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_note_guided_path_freeze_ema_and_gradients(self):
        agent, batch = tiny_agent("n")
        for step in range(2):
            agent, _ = agent.update(batch, step)
        frozen = agent.pre_actor_params
        previous_ema = agent.target_actor_params
        updated, info = agent.update(batch, 2)
        assert_tree_equal(self, frozen, updated.pre_actor_params)
        expected = jax.tree_util.tree_map(
            lambda new, old: 0.005 * new + 0.995 * old,
            updated.network.params["modules_actor_bc_flow"], previous_ema,
        )
        assert_tree_equal(self, expected, updated.target_actor_params, atol=1e-6)
        self.assertEqual(int(updated.completed_updates), 3)
        diagnostics = updated.adjoint_metrics(batch["observations"])
        self.assertIn("adjoint/position_1_norm", diagnostics)
        self.assertTrue(all(np.all(np.isfinite(x)) for x in jax.tree_util.tree_leaves((info, diagnostics))))
        gradient = jax.grad(lambda a: updated.target_reward(batch["observations"], a).sum())(batch["actions"])
        self.assertEqual(gradient.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(gradient)))
        self.assertGreater(float(jnp.linalg.norm(gradient)), 0)


if __name__ == "__main__":
    unittest.main()
