"""Primitive-step collection, independent seeded evaluation and phase checkpoints."""

from collections import deque
import json
import os
from pathlib import Path
import pickle
import random
import tempfile

import flax
import jax
import numpy as np

from utils.chunk_data import ObservationNormalizer


def jsonable(value):
    if hasattr(value, "items"):
        return {key: jsonable(item) for key, item in value.items() if not callable(item)}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
        json.dump(jsonable(value), file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
        temporary = file.name
    os.replace(temporary, path)


def reshape_action(action, horizon, action_dim):
    flat = np.asarray(action).reshape(-1)
    if flat.size != horizon * action_dim or not np.all(np.isfinite(flat)):
        raise ValueError(f"Policy must return {horizon * action_dim} finite action values")
    return np.clip(flat.reshape(horizon, action_dim), -1, 1)


class ChunkCollector:
    """A queue survives learner updates/evaluation; it is cleared on reset."""

    def __init__(self, env, normalizer, horizon, seed):
        self.env = env
        self.normalizer = normalizer
        self.horizon = horizon
        self.action_dim = int(np.prod(env.action_space.shape))
        self.key = jax.random.PRNGKey(seed)
        self.reset_rng = np.random.default_rng(np.random.SeedSequence([seed, 491]))
        self.queue = deque()
        self.observation = None
        self.env_steps = 0
        self.policy_calls = 0
        self.episodes = 0

    def step(self, agent):
        if self.observation is None:
            self.observation, _ = self.env.reset(seed=int(self.reset_rng.integers(0, 2**31)))
            self.observation = np.asarray(self.observation).copy()
            self.queue.clear()
        if not self.queue:
            self.key, key = jax.random.split(self.key)
            action = agent.sample_actions(self.normalizer(self.observation[None]), seed=key)
            self.queue.extend(reshape_action(action, self.horizon, self.action_dim))
            self.policy_calls += 1
        action = self.queue.popleft().copy()
        following, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        final_observation = info.get("final_observation", following) if done else following
        transition = dict(
            observations=self.observation.copy(), actions=action,
            rewards=np.float32(reward), next_observations=np.asarray(final_observation).copy(),
            terminals=np.float32(done), masks=np.float32(not terminated),
        )
        self.env_steps += 1
        if done:
            self.queue.clear()
            self.observation = None
            self.episodes += 1
        else:
            self.observation = np.asarray(following).copy()
        return transition, info


def evaluation_seeds(seed, episodes):
    return [int(value) for value in np.random.SeedSequence([seed, 92371]).generate_state(episodes)]


def evaluate_chunked(agent, env, normalizer, horizon, seeds):
    """No learner/replay RNG is consumed; common episode and action-noise seeds."""
    global_numpy_state = np.random.get_state()
    global_python_state = random.getstate()
    action_dim = int(np.prod(env.action_space.shape))
    rows = []
    try:
        for seed in seeds:
            np.random.seed(seed)
            random.seed(seed)
            observation, _ = env.reset(seed=seed)
            key = jax.random.PRNGKey(seed)
            total_return, length, done = 0.0, 0, False
            while not done:
                key, draw = jax.random.split(key)
                action = agent.sample_actions(normalizer(np.asarray(observation)[None]), seed=draw)
                for primitive in reshape_action(action, horizon, action_dim):
                    observation, reward, terminated, truncated, info = env.step(primitive.copy())
                    total_return += float(reward)
                    length += 1
                    done = bool(terminated or truncated)
                    if done:
                        break
            normalized = float(env.unwrapped.get_normalized_score(total_return) * 100.0)
            if not np.isfinite(normalized) or not np.isfinite(total_return):
                raise ValueError("Nonfinite evaluation return")
            rows.append(dict(seed=seed, raw_return=total_return,
                             normalized_return=normalized, length=length))
    finally:
        np.random.set_state(global_numpy_state)
        random.setstate(global_python_state)
    if not rows:
        raise ValueError("At least one evaluation episode is required")
    return {
        "evaluation/episode.normalized_return": float(np.mean([r["normalized_return"] for r in rows])),
        "evaluation/episode.return": float(np.mean([r["raw_return"] for r in rows])),
        "evaluation/episode.length": float(np.mean([r["length"] for r in rows])),
        "episodes": rows,
    }


def checkpoint_contract(env_name, method, seed, config, normalizer, dataset_identity):
    return jsonable(dict(
        env_name=env_name, method=method, seed=seed, agent_config=config,
        normalization_fingerprint=normalizer.fingerprint(), dataset_identity=dataset_identity,
    ))


def save_checkpoint(path, agent, normalizer, sample_rng, contract, offline_updates, online_env_steps, phase):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        format_version=1, phase=phase, contract=contract,
        offline_updates=int(offline_updates), online_env_steps=int(online_env_steps),
        agent=jax.device_get(flax.serialization.to_state_dict(agent)),
        normalizer=normalizer.state_dict(), sample_rng=sample_rng.bit_generator.state,
        resumable_offline_boundary=(phase == "offline_complete" and online_env_steps == 0),
    )
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
        temporary = file.name
    os.replace(temporary, path)


def restore_offline_checkpoint(path, agent, expected_contract, expected_offline_updates):
    # Only load checkpoints from this experiment; pickle is not an untrusted format.
    with Path(path).open("rb") as file:
        payload = pickle.load(file)
    if payload.get("format_version") != 1 or not payload.get("resumable_offline_boundary"):
        raise ValueError("Only completed offline-boundary checkpoints can start online in v1")
    if payload["contract"] != expected_contract:
        keys = sorted(key for key in set(payload["contract"]) | set(expected_contract)
                      if payload["contract"].get(key) != expected_contract.get(key))
        raise ValueError(f"Checkpoint contract mismatch (including H/task/dimensions): {keys}")
    if payload["offline_updates"] != expected_offline_updates or payload["online_env_steps"] != 0:
        raise ValueError("Checkpoint is not at the requested offline endpoint")
    normalizer = ObservationNormalizer.from_state_dict(payload["normalizer"])
    if normalizer.fingerprint() != expected_contract["normalization_fingerprint"]:
        raise ValueError("Checkpoint normalization does not match its contract")
    restored = flax.serialization.from_state_dict(agent, payload["agent"])
    sample_rng = np.random.default_rng()
    sample_rng.bit_generator.state = payload["sample_rng"]
    return restored, normalizer, sample_rng
