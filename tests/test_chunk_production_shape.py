"""Opt-in stress check of actual production model/batch/teacher dimensions.

This is synthetic numerical validation, not a task-performance experiment.
AM_MF_FULL_MODEL_TEST=1 enables it; the ordinary CPU suite skips the costly check.
"""

import os
from pathlib import Path
import unittest

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import yaml

from agents.chunked import create_agent, make_config


@unittest.skipUnless(os.environ.get("AM_MF_FULL_MODEL_TEST") == "1", "opt-in production-dimension GPU check")
class ProductionShapeTest(unittest.TestCase):
    def test_h10_production_b0_and_guided_n(self):
        root = Path(os.environ.get("AM_MF_REPO", Path(__file__).resolve().parents[1]))
        with (root / "configs/chunk/door-cloned-v1.yaml").open() as file:
            profile = yaml.safe_load(file)
        data_path = Path(os.environ.get("D4RL_DATASET_DIR", "/home/lrh/d4rl_datasets")) / "door-cloned-v1.hdf5"
        with h5py.File(data_path, "r") as file:
            observation_dim = file["observations"].shape[-1]
            action_dim = file["actions"].shape[-1]
        rng = np.random.default_rng(981)
        batch = dict(
            observations=jnp.asarray(rng.normal(size=(256, observation_dim)), dtype=jnp.float32),
            next_observations=jnp.asarray(rng.normal(size=(256, observation_dim)), dtype=jnp.float32),
            actions=jnp.asarray(rng.uniform(-1, 1, size=(256, 10 * action_dim)), dtype=jnp.float32),
            rewards=jnp.linspace(-5, 5, 256), masks=jnp.ones(256),
        )
        for method in ("b0", "n"):
            settings = dict(profile["agent"])
            if method == "n":
                settings.update(profile["n_agent"])
                settings["behavior_warmup_updates"] = 2
            config = make_config(method, settings, 10, action_dim)
            agent = create_agent(method, 1, batch["observations"][:1], batch["actions"][:1], config, 1000000, 1000000)
            for step in range(3):
                agent, info = agent.update(batch, step)
                self.assertTrue(all(np.all(np.isfinite(value)) for value in jax.tree_util.tree_leaves(info)))
            actions = agent.sample_actions(batch["observations"], seed=jax.random.PRNGKey(11))
            self.assertEqual(actions.shape, (256, 10 * action_dim))
            if method == "n":
                self.assertIn("actor/transport", info)
                self.assertTrue(all(np.all(np.isfinite(value)) for value in jax.tree_util.tree_leaves(agent.adjoint_metrics(batch["observations"]))))
            print(f"Production-shape PASS: {method}, H=10, B=256, action_dim={action_dim}, actor=256x3, critic=512x4", flush=True)
            del agent
            jax.clear_caches()


if __name__ == "__main__":
    unittest.main()
