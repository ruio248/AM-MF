import unittest

import numpy as np

from utils.chunk_data import ChunkReplayBuffer, ObservationNormalizer


def dataset(length=24, episode_length=12):
    observations = np.arange(length * 2, dtype=np.float32).reshape(length, 2)
    terminals = ((np.arange(length) + 1) % episode_length == 0).astype(np.float32)
    terminals[-1] = 1
    return dict(observations=observations, next_observations=observations + 0.5,
                actions=np.arange(length, dtype=np.float32)[:, None],
                rewards=np.arange(1, length + 1, dtype=np.float32),
                terminals=terminals, masks=1.0 - terminals)


class ChunkDataTest(unittest.TestCase):
    def test_h1_and_full_h_rewards_boundaries(self):
        data = dataset()
        for horizon in (1, 2, 5, 10):
            replay = ChunkReplayBuffer(data, 40, horizon, 0.9)
            expected = [i for i in range(24 - horizon + 1)
                        if not np.any(data["terminals"][i:i + horizon - 1])]
            self.assertEqual(sorted(replay.pool[:replay.pool_size]), expected)
            for start in expected:
                batch = replay.batch_from_starts([start])
                end = start + horizon
                np.testing.assert_array_equal(batch["actions"][0], data["actions"][start:end, 0])
                np.testing.assert_allclose(batch["rewards"][0],
                    np.dot(data["rewards"][start:end], 0.9 ** np.arange(horizon)), rtol=1e-6)
                np.testing.assert_allclose(batch["next_observations"],
                    replay.normalizer(data["next_observations"][[end - 1]]))
                self.assertEqual(batch["masks"][0], np.prod(data["masks"][start:end]))

    def test_timeout_bootstraps_but_true_terminal_does_not(self):
        data = dataset(6, 3)
        data["masks"][2] = 1  # timeout still ends the episode
        replay = ChunkReplayBuffer(data, 12, 2, 0.99)
        np.testing.assert_array_equal(replay.batch_from_starts([1, 4])["masks"], [1, 0])
        with self.assertRaises(ValueError):
            replay.batch_from_starts([2])

    def test_normalization_independent_of_capacity_horizon_and_online_data(self):
        data = dataset()
        expected = ObservationNormalizer.fit(data["observations"])
        for capacity in (24, 100):
            for horizon in (1, 10):
                replay = ChunkReplayBuffer(data, capacity, horizon, 0.99)
                self.assertEqual(replay.normalizer.fingerprint(), expected.fingerprint())
                transition = {key: value[0].copy() for key, value in data.items()}
                transition["observations"] = np.array([1e6, 1e6], dtype=np.float32)
                replay.add_transition(transition)
                self.assertEqual(replay.normalizer.fingerprint(), expected.fingerprint())

    def test_ring_wrap_and_overwrite_match_brute_force(self):
        for horizon in (1, 2, 5, 10):
            replay = ChunkReplayBuffer(dataset(12, 12), 14, horizon, 0.99)
            rng = np.random.default_rng(31)
            for step in range(80):
                done = step % 13 == 12
                replay.add_transition(dict(
                    observations=np.array([step, step + 1], dtype=np.float32),
                    next_observations=np.array([step + 1, step + 2], dtype=np.float32),
                    actions=np.array([step], dtype=np.float32), rewards=np.float32(step),
                    terminals=np.float32(done), masks=np.float32(not done),
                ))
                expected = [start for start in range(14) if replay._valid_start(start)]
                actual = replay.pool[:replay.pool_size]
                self.assertEqual(sorted(actual), expected)
                self.assertEqual(len(set(actual)), replay.pool_size)
                self.assertEqual(replay.online_windows, int(replay.is_online[actual].sum()))
                if replay.pool_size:
                    self.assertEqual(replay.sample(8, rng)["actions"].shape, (8, horizon))

    def test_short_episodes_have_no_padded_targets(self):
        replay = ChunkReplayBuffer(dataset(6, 2), 10, 5, 0.99)
        self.assertEqual(replay.pool_size, 0)
        with self.assertRaisesRegex(ValueError, "No complete"):
            replay.sample(2, np.random.default_rng(0))


if __name__ == "__main__":
    unittest.main()
