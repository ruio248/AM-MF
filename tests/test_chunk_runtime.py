from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from utils.chunk_data import ObservationNormalizer
from utils.chunk_runtime import (
    ChunkCollector, checkpoint_contract, evaluate_chunked, evaluation_seeds,
    restore_offline_checkpoint, save_checkpoint,
)
from test_chunk_agents import assert_tree_equal, tiny_agent


class ToyEnv:
    action_space = SimpleNamespace(shape=(2,))

    def __init__(self, episode_length=3, timeout=False):
        self.episode_length = episode_length
        self.timeout = timeout
        self.unwrapped = self
        self.steps = 0
        self.total_steps = 0
        self.resets = 0

    def reset(self, seed=None):
        self.steps = 0
        self.resets += 1
        return np.array([0.0, 0.0]), {}

    def step(self, action):
        self.steps += 1
        self.total_steps += 1
        done = self.steps == self.episode_length
        return np.array([self.steps, 0.0]), float(action[0]), done and not self.timeout, done and self.timeout, {}

    def get_normalized_score(self, value):
        return value / 10


class ToyAgent:
    def __init__(self, horizon):
        self.horizon = horizon
        self.calls = 0

    def sample_actions(self, observations, seed):
        self.calls += 1
        return np.arange(self.horizon * 2, dtype=np.float32)[None] / 20


class ChunkRuntimeTest(unittest.TestCase):
    def test_queue_reset_budget_and_timeout_masks(self):
        norm = ObservationNormalizer.fit(np.array([[0.0, 0.0], [1.0, 1.0]]))
        for horizon in (1, 2, 5, 10):
            for timeout in (False, True):
                env = ToyEnv(timeout=timeout)
                agent = ToyAgent(horizon)
                collector = ChunkCollector(env, norm, horizon, 1)
                updates = 0
                for _ in range(12):
                    transition, _ = collector.step(agent)
                    updates += 1
                    if transition["terminals"]:
                        self.assertEqual(transition["masks"], float(timeout))
                        self.assertFalse(collector.queue)
                self.assertEqual(env.total_steps, 12)
                self.assertEqual(collector.env_steps, updates)
                self.assertEqual(agent.calls, 4 * int(np.ceil(3 / horizon)))

    def test_evaluation_preserves_training_global_rng_and_budget(self):
        norm = ObservationNormalizer.fit(np.array([[0.0, 0.0], [1.0, 1.0]]))
        train_env = ToyEnv()
        collector = ChunkCollector(train_env, norm, 5, 1)
        collector.step(ToyAgent(5))
        key_before = np.array(collector.key)
        np.random.seed(81)
        expected = np.random.random(3)
        np.random.seed(81)
        result = evaluate_chunked(ToyAgent(5), ToyEnv(), norm, 5, evaluation_seeds(1, 2))
        np.testing.assert_array_equal(np.random.random(3), expected)
        np.testing.assert_array_equal(collector.key, key_before)
        self.assertEqual(collector.env_steps, 1)
        self.assertEqual(len(result["episodes"]), 2)

    def test_offline_resume_full_state_and_next_update(self):
        agent, batch = tiny_agent("n")
        for step in range(4):
            agent, _ = agent.update(batch, step)
        # Exercise serialization of nontrivial dynamic-alpha/history state too.
        agent = agent.replace(current_alpha=jnp.asarray(2.5), valid_count=jnp.asarray(7),
                              loss_history=jnp.arange(len(agent.loss_history), dtype=jnp.float32))
        norm = ObservationNormalizer.fit(np.asarray(batch["observations"]))
        contract = checkpoint_contract("door-cloned-v1", "n", 1, agent.config, norm, {"sha256": "test"})
        rng = np.random.default_rng(11)
        rng.random(7)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "offline.pkl"
            save_checkpoint(path, agent, norm, rng, contract, 4, 0, "offline_complete")
            template, _ = tiny_agent("n")
            restored, saved_norm, saved_rng = restore_offline_checkpoint(path, template, contract, 4)
            # Compare serialized state: static function identities intentionally differ.
            import flax
            assert_tree_equal(self, flax.serialization.to_state_dict(agent), flax.serialization.to_state_dict(restored))
            self.assertEqual(saved_norm.fingerprint(), norm.fingerprint())
            np.testing.assert_array_equal(rng.integers(1000, size=20), saved_rng.integers(1000, size=20))
            next_a, _ = agent.update(batch, 4)
            next_b, _ = restored.update(batch, 4)
            assert_tree_equal(self, flax.serialization.to_state_dict(next_a), flax.serialization.to_state_dict(next_b), atol=1e-6)
            bad = dict(contract, env_name="pen-cloned-v1")
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                restore_offline_checkpoint(path, template, bad, 4)
            different_h, _ = tiny_agent("n", 5)
            bad = checkpoint_contract("door-cloned-v1", "n", 1, different_h.config, norm, {"sha256": "test"})
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                restore_offline_checkpoint(path, different_h, bad, 4)


if __name__ == "__main__":
    unittest.main()
