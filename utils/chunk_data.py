"""Lazy full-H windows over a single-step ring buffer.

No padded actions are supervised. A terminal may occur at the last transition
of a window, but never before it. Episode IDs also separate offline from online.
"""

from dataclasses import dataclass
import hashlib

import numpy as np


FIELDS = ("observations", "actions", "rewards", "next_observations", "terminals", "masks")


@dataclass(frozen=True)
class ObservationNormalizer:
    mean: np.ndarray
    std: np.ndarray
    count: int

    @classmethod
    def fit(cls, observations):
        observations = np.asarray(observations)
        if observations.ndim != 2 or len(observations) == 0:
            raise ValueError("Normalization needs nonempty valid vector observations")
        if not np.all(np.isfinite(observations)):
            raise ValueError("Nonfinite offline observations")
        mean = observations.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
        std = observations.std(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
        std = np.where(std < 1e-6, np.float32(1.0), std)
        return cls(mean, std, len(observations))

    def __call__(self, observations):
        return ((np.asarray(observations) - self.mean) / self.std).astype(np.float32)

    def fingerprint(self):
        digest = hashlib.sha256()
        for array in (self.mean, self.std):
            digest.update(array.tobytes())
        digest.update(str(self.count).encode())
        return digest.hexdigest()

    def state_dict(self):
        return dict(mean=self.mean.copy(), std=self.std.copy(), count=self.count)

    @classmethod
    def from_state_dict(cls, state):
        return cls(np.asarray(state["mean"]), np.asarray(state["std"]), int(state["count"]))


class ChunkReplayBuffer:
    """Uniform O(batch*H) sampling; O(H) insertion/index maintenance.

    The dense valid-start pool and inverse index allow O(1) removal of each
    invalidated start. Storage remains O(capacity*d), not O(capacity*H*d).
    """

    def __init__(self, initial_data, capacity, chunk_size, discount, normalizer=None):
        self.horizon = int(chunk_size)
        self.capacity = int(capacity)
        self.discount = float(discount)
        if self.horizon < 1 or self.capacity < self.horizon:
            raise ValueError("Require capacity >= chunk_size >= 1")
        if not 0 < self.discount <= 1:
            raise ValueError("discount must be in (0,1]")
        missing = set(FIELDS) - set(initial_data)
        if missing:
            raise ValueError(f"Missing transition fields: {sorted(missing)}")
        data = {key: np.asarray(initial_data[key], dtype=np.float32) for key in FIELDS}
        length = len(data["observations"])
        if length < 1 or length > self.capacity:
            raise ValueError("Initial dataset must be nonempty and fit buffer capacity")
        if any(len(value) != length for value in data.values()):
            raise ValueError("Transition arrays must have equal lengths")
        if data["observations"].ndim != 2 or data["actions"].ndim != 2:
            raise ValueError("Only vector observations/actions are supported")
        if data["next_observations"].shape != data["observations"].shape:
            raise ValueError("Observation and next-observation shapes differ")
        for key in ("rewards", "terminals", "masks"):
            if data[key].shape != (length,):
                raise ValueError(f"{key} must have shape [N]")
        if not data["terminals"][-1]:
            raise ValueError("The offline dataset must end at an explicit episode boundary")
        if np.any((data["masks"] == 0) & (data["terminals"] == 0)):
            raise ValueError("A true terminal must also mark an episode boundary")
        self.normalizer = normalizer or ObservationNormalizer.fit(data["observations"])
        self.data = {}
        for key, values in data.items():
            array = np.zeros((self.capacity, *values.shape[1:]), dtype=values.dtype)
            array[:length] = values
            self.data[key] = array
        self.size = length
        self.next_id = length
        self.pointer = length % self.capacity
        self.transition_ids = np.full(self.capacity, -1, dtype=np.int64)
        self.transition_ids[:length] = np.arange(length)
        self.episode_ids = np.full(self.capacity, -1, dtype=np.int64)
        self.episode_ids[:length] = np.concatenate((
            np.zeros(1, dtype=np.int64), np.cumsum(data["terminals"][:-1] > 0)
        ))
        self.online_episode = int(self.episode_ids[length - 1]) + 1
        self.is_online = np.zeros(self.capacity, dtype=bool)
        self.pool = np.empty(self.capacity, dtype=np.int64)
        self.pool_positions = np.full(self.capacity, -1, dtype=np.int64)
        self.pool_size = 0
        self.online_windows = 0
        self.offsets = np.arange(self.horizon, dtype=np.int64)
        self.reward_weights = np.power(np.float32(self.discount), self.offsets).astype(np.float32)
        starts = np.arange(max(0, length - self.horizon + 1), dtype=np.int64)
        prefix = np.concatenate(([0], np.cumsum(data["terminals"] > 0, dtype=np.int64)))
        starts = starts[prefix[starts + self.horizon - 1] == prefix[starts]]
        self.pool[:len(starts)] = starts
        self.pool_positions[starts] = np.arange(len(starts))
        self.pool_size = len(starts)

    def _remove_start(self, start):
        position = self.pool_positions[start]
        if position < 0:
            return
        self.online_windows -= int(self.is_online[start])
        self.pool_size -= 1
        last = self.pool[self.pool_size]
        self.pool[position] = last
        self.pool_positions[last] = position
        self.pool_positions[start] = -1

    def _valid_start(self, start):
        indices = (start + self.offsets) % self.capacity
        first_id = self.transition_ids[start]
        return bool(
            first_id >= 0
            and np.all(self.transition_ids[indices] == first_id + self.offsets)
            and np.all(self.episode_ids[indices] == self.episode_ids[start])
            and not np.any(self.data["terminals"][indices[:-1]] > 0)
        )

    def add_transition(self, transition):
        if set(transition) != set(FIELDS):
            raise ValueError("Online transition fields must match the offline schema")
        if transition["masks"] == 0 and not transition["terminals"]:
            raise ValueError("A true terminal must mark an episode boundary")
        slot = self.pointer
        # Validate all shapes before changing buffer state.
        values = {key: np.asarray(transition[key], dtype=np.float32) for key in FIELDS}
        for key, value in values.items():
            if value.shape != self.data[key].shape[1:]:
                raise ValueError(f"Online {key} shape mismatch")
        for start in (slot - self.offsets) % self.capacity:
            self._remove_start(int(start))
        for key, value in values.items():
            self.data[key][slot] = value
        self.transition_ids[slot] = self.next_id
        self.episode_ids[slot] = self.online_episode
        self.is_online[slot] = True
        self.online_episode += int(bool(transition["terminals"]))
        self.next_id += 1
        self.pointer = (slot + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        start = (slot - self.horizon + 1) % self.capacity
        if self._valid_start(start):
            self.pool[self.pool_size] = start
            self.pool_positions[start] = self.pool_size
            self.pool_size += 1
            self.online_windows += int(self.is_online[start])

    def sample(self, batch_size, rng):
        if self.pool_size == 0:
            raise ValueError(f"No complete H={self.horizon} windows in replay")
        starts = self.pool[rng.integers(self.pool_size, size=batch_size)]
        return self.batch_from_starts(starts)

    def batch_from_starts(self, starts):
        starts = np.asarray(starts, dtype=np.int64)
        if starts.ndim != 1 or len(starts) == 0:
            raise ValueError("starts must be a nonempty vector")
        if np.any(starts < 0) or np.any(starts >= self.capacity) or np.any(self.pool_positions[starts] < 0):
            raise ValueError("Invalid or overwritten chunk start")
        indices = (starts[:, None] + self.offsets[None]) % self.capacity
        end = indices[:, -1]
        return dict(
            observations=self.normalizer(self.data["observations"][starts]),
            actions=self.data["actions"][indices].reshape(len(starts), -1),
            rewards=np.sum(self.data["rewards"][indices] * self.reward_weights, axis=1),
            masks=np.prod(self.data["masks"][indices], axis=1),
            next_observations=self.normalizer(self.data["next_observations"][end]),
        )

    def metrics(self):
        potential = max(0, self.size - self.horizon + 1)
        return {
            "replay/transitions": self.size,
            "replay/valid_windows": self.pool_size,
            "replay/boundary_excluded_fraction": 1.0 - self.pool_size / max(potential, 1),
            "replay/ineligible_start_fraction": 1.0 - self.pool_size / max(self.size, 1),
            "replay/online_window_fraction": self.online_windows / max(self.pool_size, 1),
        }
