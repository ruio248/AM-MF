import unittest

import numpy as np

from utils.datasets import Dataset


class ChunkDatasetTest(unittest.TestCase):
    def setUp(self):
        count = 6
        observations = np.arange(count * 2, dtype=np.float32).reshape(count, 2)
        next_observations = observations + 0.5
        actions = np.arange(count, dtype=np.float32).reshape(count, 1)
        rewards = np.arange(1, count + 1, dtype=np.float32)
        terminals = np.array([0, 0, 1, 0, 0, 1], dtype=np.float32)
        masks = 1.0 - terminals
        self.dataset = Dataset.create(
            observations=observations,
            actions=actions,
            rewards=rewards,
            masks=masks,
            terminals=terminals,
            next_observations=next_observations,
        )

    def test_h1_preserves_transition_semantics(self):
        chunked = self.dataset.to_chunked(1, discount=0.9)
        np.testing.assert_array_equal(chunked["observations"], self.dataset["observations"])
        np.testing.assert_array_equal(chunked["actions"], self.dataset["actions"])
        np.testing.assert_array_equal(chunked["rewards"], self.dataset["rewards"])
        np.testing.assert_array_equal(chunked["masks"], self.dataset["masks"])
        np.testing.assert_array_equal(
            chunked["next_observations"], self.dataset["next_observations"]
        )
        np.testing.assert_array_equal(
            chunked["executed_steps"], np.ones((len(self.dataset), 1))
        )
        np.testing.assert_array_equal(
            chunked["valid_action_mask"], np.ones((len(self.dataset), 1))
        )

    def test_h2_aggregates_rewards_and_does_not_cross_terminal(self):
        chunked = self.dataset.to_chunked(2, discount=0.9)
        # Valid starts are 0, 1, 3, 4. Start 2 would cross the terminal at 2.
        np.testing.assert_array_equal(
            chunked["actions"].reshape(-1, 2),
            np.array([[0, 1], [1, 2], [3, 4], [4, 5]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            chunked["rewards"],
            np.array([1.0 + 0.9 * 2.0, 2.0 + 0.9 * 3.0,
                      4.0 + 0.9 * 5.0, 5.0 + 0.9 * 6.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            chunked["next_observations"], self.dataset["next_observations"][[1, 2, 4, 5]]
        )
        np.testing.assert_array_equal(
            chunked["masks"], np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        )


if __name__ == "__main__":
    unittest.main()

