import unittest

from absl import flags
from absl.testing import flagsaver
import flax
import jax
import jax.numpy as jnp
import numpy as np

from agents.am_meanflow_note import AMMeanFlowNoteAgent, get_config


FLAGS = flags.FLAGS
for name, value in (("offline_steps", 20), ("online_steps", 10)):
    if name not in FLAGS:
        flags.DEFINE_integer(name, value, "Test training budget")
if "pretrain_factor" not in FLAGS:
    flags.DEFINE_float("pretrain_factor", 0.0, "Test behavior-only budget")
if not FLAGS.is_parsed():
    FLAGS(["test_note"])


def close_tree(a, b, atol=1e-6):
    leaves_a, leaves_b = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(leaves_a) == len(leaves_b) and all(
        np.allclose(x, y, atol=atol, rtol=1e-6)
        for x, y in zip(leaves_a, leaves_b)
    )


def make_agent(variant="note_adjoint", transport_target_mode="interval_mean"):
    config = get_config()
    config.encoder = None
    config.actor_hidden_dims = 16
    config.actor_depth = 1
    config.actor_num_heads = 2
    config.value_hidden_dims = (16, 16)
    config.batch_size = 4
    config.teacher_batch_size = 2
    config.teacher_steps = 2
    config.jacobian_batch_size = 1
    config.jacobian_interval = 2
    config.behavior_warmup_updates = 3
    config.alpha = 1.0
    config.variant = variant
    config.transport_target_mode = transport_target_mode
    config.use_dynamic_alpha = False
    config.time_steps = 8
    observations = jnp.array(
        [
            [0.1, 0.2, 0.3],
            [0.3, -0.2, 0.5],
            [-0.1, -0.5, 0.2],
            [0.3, 0.4, 0.9],
        ]
    )
    actions = jnp.array([[0.2, -0.1], [0.4, 0.3], [-0.2, 0.5], [0.1, 0.2]])
    with flagsaver.flagsaver(
        offline_steps=20, online_steps=10, pretrain_factor=0.0
    ):
        agent = AMMeanFlowNoteAgent.create(
            1, observations[:1], actions[:1], config
        )
    batch = {
        "observations": observations,
        "actions": actions,
        "next_observations": observations + 0.02,
        "rewards": jnp.arange(4.0) * 0.1,
        "masks": jnp.ones(4),
    }
    return agent, batch


class NoteAgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.initial, cls.batch = make_agent()
        cls.warm = cls.initial
        for step in range(3):
            cls.warm, _ = cls.warm.update(cls.batch, step)

    def test_warmup_and_frozen_prior_lifecycle(self):
        self.assertFalse(
            close_tree(
                self.initial.network.params["modules_actor_bc_flow"],
                self.warm.network.params["modules_actor_bc_flow"],
                atol=1e-9,
            )
        )
        self.assertFalse(
            close_tree(
                self.initial.network.params["modules_critic"],
                self.warm.network.params["modules_critic"],
                atol=1e-9,
            )
        )
        self.assertTrue(
            close_tree(
                self.warm.pre_actor_params,
                self.warm.network.params["modules_actor_bc_flow"],
            )
        )
        following, info = self.warm.update(self.batch, 3)
        self.assertTrue(
            close_tree(following.pre_actor_params, self.warm.pre_actor_params)
        )
        self.assertEqual(int(following.completed_updates), 4)
        expected = jax.tree_util.tree_map(
            lambda new, old: 0.005 * new + 0.995 * old,
            following.network.params["modules_actor_bc_flow"],
            self.warm.target_actor_params,
        )
        self.assertTrue(close_tree(expected, following.target_actor_params))
        self.assertTrue(
            all(
                np.all(np.isfinite(x))
                for x in jax.tree_util.tree_leaves(info)
            )
        )

    def test_actor_target_does_not_update_critic_or_snapshots(self):
        loss = lambda params: self.warm.guided_actor_loss(
            self.batch, params, jax.random.PRNGKey(11), True
        )[0]
        gradient = jax.jit(jax.grad(loss))(self.warm.network.params)
        for name in ("modules_critic", "modules_target_critic"):
            self.assertTrue(
                all(
                    np.all(np.asarray(value) == 0)
                    for value in jax.tree_util.tree_leaves(gradient[name])
                )
            )
        self.assertGreater(
            sum(
                float(jnp.sum(value**2))
                for value in jax.tree_util.tree_leaves(
                    gradient["modules_actor_bc_flow"]
                )
            ),
            0.0,
        )

        for mode in ("interval_mean", "path_local"):
            config = flax.core.unfreeze(self.warm.config)
            config["transport_target_mode"] = mode
            agent = self.warm.replace(config=config)
            target = lambda pre, ema: agent.replace(
                pre_actor_params=pre, target_actor_params=ema
            ).transport_batch(
                self.batch["observations"], jax.random.PRNGKey(4)
            )[0][-1].sum()
            pre_gradient, ema_gradient = jax.grad(target, argnums=(0, 1))(
                agent.pre_actor_params, agent.target_actor_params
            )
            self.assertTrue(
                all(
                    np.all(np.asarray(value) == 0)
                    for value in jax.tree_util.tree_leaves(
                        (pre_gradient, ema_gradient)
                    )
                ),
                mode,
            )

    def test_transport_batch_shape_and_endpoint_target(self):
        data, _, path = jax.jit(self.warm.transport_batch)(
            self.batch["observations"], jax.random.PRNGKey(7)
        )
        _, x, r, t, velocity = data
        self.assertEqual(x.shape, (16, 2))
        full = np.arange(0, 16, 8)
        np.testing.assert_allclose(x[full] - velocity[full], path[-1], atol=1e-6)
        self.assertEqual(int(jnp.sum(r == t)), 4)
        self.assertTrue(np.all(np.isfinite(velocity)))

    def test_path_local_uses_same_path_and_only_instantaneous_targets(self):
        config = flax.core.unfreeze(self.warm.config)
        config["transport_target_mode"] = "path_local"
        local = self.warm.replace(config=config)
        rng = jax.random.PRNGKey(7)
        interval_data, interval_noise, interval_path = self.warm.transport_batch(
            self.batch["observations"], rng
        )
        local_data, local_noise, local_path = local.transport_batch(
            self.batch["observations"], rng
        )
        interval_obs, interval_x, _, interval_t, _ = interval_data
        local_obs, local_x, local_r, local_t, local_velocity = local_data
        np.testing.assert_allclose(local_noise, interval_noise, atol=1e-6)
        np.testing.assert_allclose(local_path, interval_path, atol=1e-6)
        np.testing.assert_allclose(local_x, interval_x, atol=1e-6)
        np.testing.assert_allclose(local_obs, interval_obs, atol=1e-6)
        np.testing.assert_allclose(local_t, interval_t, atol=1e-6)
        np.testing.assert_allclose(local_r, local_t, atol=0.0)
        expected = local.teacher_field(local_obs, local_x, local_t)
        np.testing.assert_allclose(local_velocity, expected, atol=1e-6)
        self.assertEqual(local_x.shape, (16, 2))
        self.assertTrue(np.all(np.isfinite(local_velocity)))

    def test_invalid_transport_target_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "transport target mode"):
            make_agent(transport_target_mode="unknown")

    def test_gated_control_schedule_and_terms_are_bounded(self):
        config = flax.core.unfreeze(self.warm.config)
        config["control_eta_ramp_updates"] = 10
        config["control_adjoint_clip"] = 0.05
        config["control_uncertainty_scale"] = 0.25
        config["control_uncertainty_end_update"] = 8
        controlled = self.warm.replace(config=config)
        self.assertAlmostEqual(float(controlled.scheduled_control_eta(3)), 0.0)
        self.assertAlmostEqual(float(controlled.scheduled_control_eta(8)), 0.05)
        self.assertAlmostEqual(float(controlled.scheduled_control_eta(13)), 0.1)
        self.assertAlmostEqual(
            float(controlled.scheduled_uncertainty_scale(7)), 0.25
        )
        self.assertAlmostEqual(
            float(controlled.scheduled_uncertainty_scale(8)), 0.0
        )
        noise = jnp.zeros((2, 2))
        time = jnp.ones((2, 1))
        _, gate, info = controlled.control_terms(
            self.batch["observations"][:2], noise, time
        )
        self.assertTrue(np.all(np.isfinite(gate)))
        self.assertTrue(np.all((np.asarray(gate) > 0) & (np.asarray(gate) <= 1)))
        self.assertLessEqual(float(jnp.max(info["clipped_norm"])), 0.050001)
        updated, update_info = controlled.update(self.batch, 8)
        self.assertEqual(int(updated.completed_updates), 4)
        self.assertAlmostEqual(float(update_info["control/eta"]), 0.05)
        self.assertTrue(np.isfinite(float(update_info["control/gate"])))


if __name__ == "__main__":
    unittest.main()
