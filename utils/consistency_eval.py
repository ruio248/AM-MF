"""Consistency diagnostics shared by Native MeanFlow and MeanFlowQL.

The two policy families use different parameterizations:

* Native MeanFlow predicts an interval-average velocity ``u(o, x_t, r, t)``.
* MeanFlowQL predicts a direct map ``g(o, x_t, t)`` whose implied endpoint is
  ``(1 - t) * x_t + t * g(o, x_t, t)``.

This module keeps those contracts separate while reporting a common set of
endpoint, multi-NFE, trajectory-geometry, and endpoint-Jacobian diagnostics.
Native MeanFlow additionally receives the interval split-identity test.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


Array = jax.Array
NativeVelocityFn = Callable[[Array, Array, Array, Array], Array]
DirectMapFn = Callable[[Array, Array, Array], Array]
EndpointFn = Callable[[Array, Array, Array], Array]
RolloutFn = Callable[[Array, Array, int], tuple[Array, Array]]
RemainingRolloutFn = Callable[[Array, Array, int, int], Array]

DEFAULT_NFES = (1, 2, 4, 10)
DEFAULT_SPLIT_TRIPLETS = (
    (0, 5, 10),
    (0, 2, 5),
    (2, 5, 10),
    (0, 1, 2),
    (4, 7, 10),
)


@dataclass(frozen=True)
class ConsistencyThresholds:
    """Optional task-calibrated upper bounds for binary judgement.

    No universal threshold is assumed.  A field set to ``None`` is omitted
    from the judgement, which lets experiments calibrate thresholds from a
    behavior-only reference checkpoint before comparing RL variants.
    """

    endpoint_map_nmse: float | None = None
    endpoint_jvp_nmse: float | None = None
    split_consistency_nmse: float | None = None
    k1_k_reference_mse: float | None = None


def _as_float(value: Any) -> float:
    return float(np.asarray(jax.device_get(value)))


def _validate_inputs(
    observations: Array,
    noises: Array,
    nfe_values: Sequence[int],
    trajectory_steps: int,
    jacobian_probe_pairs: int,
) -> tuple[int, ...]:
    if observations.shape[0] != noises.shape[0]:
        raise ValueError(
            "observations and noises must have the same leading dimension."
        )
    if observations.shape[0] == 0:
        raise ValueError("consistency evaluation requires at least one pair.")
    if noises.ndim != 2:
        raise ValueError(
            f"noises must have shape [batch, action_dim], got {noises.shape}."
        )
    if trajectory_steps < 1:
        raise ValueError("trajectory_steps must be positive.")
    if jacobian_probe_pairs < 0:
        raise ValueError("jacobian_probe_pairs must be non-negative.")

    normalized_nfes = tuple(dict.fromkeys(int(value) for value in nfe_values))
    if not normalized_nfes or any(value < 1 for value in normalized_nfes):
        raise ValueError("nfe_values must contain positive integers.")
    if 1 not in normalized_nfes:
        normalized_nfes = (1, *normalized_nfes)
    return normalized_nfes


def _resolve_batch_size(total_pairs: int, inference_batch_size: int | None) -> int:
    if inference_batch_size is None:
        return total_pairs
    if inference_batch_size < 1:
        raise ValueError("inference_batch_size must be positive.")
    return min(inference_batch_size, total_pairs)


def _batched_rollout(
    rollout_fn: RolloutFn,
    observations: Array,
    noises: Array,
    num_steps: int,
    batch_size: int,
) -> tuple[Array, Array]:
    final_chunks = []
    state_chunks = []
    for start in range(0, observations.shape[0], batch_size):
        stop = min(start + batch_size, observations.shape[0])
        final, states = rollout_fn(
            observations[start:stop], noises[start:stop], num_steps
        )
        final_chunks.append(final)
        state_chunks.append(states)
    return (
        jnp.concatenate(final_chunks, axis=0),
        jnp.concatenate(state_chunks, axis=1),
    )


def default_split_triplets(
    trajectory_steps: int,
) -> tuple[tuple[int, int, int], ...]:
    """Scale the historical K10 split probes to another trajectory grid."""

    if trajectory_steps < 2:
        raise ValueError(
            "Native split consistency requires trajectory_steps >= 2."
        )
    fractions = (
        (0.0, 0.5, 1.0),
        (0.0, 0.2, 0.5),
        (0.2, 0.5, 1.0),
        (0.0, 0.1, 0.2),
        (0.4, 0.7, 1.0),
    )
    triplets = []
    for values in fractions:
        triplet = tuple(round(value * trajectory_steps) for value in values)
        if triplet[0] < triplet[1] < triplet[2]:
            triplets.append(triplet)
    if not triplets:
        triplets.append((0, 1, trajectory_steps))
    return tuple(dict.fromkeys(triplets))


def native_endpoint(
    velocity_fn: NativeVelocityFn,
    observations: Array,
    state: Array,
    current_time: Array,
) -> Array:
    """Predict the time-zero endpoint from a native MeanFlow state."""

    target_time = jnp.zeros_like(current_time)
    average_velocity = velocity_fn(
        observations, state, target_time, current_time
    )
    return state - current_time * average_velocity


def meanflowql_endpoint(
    direct_map_fn: DirectMapFn,
    observations: Array,
    state: Array,
    current_time: Array,
) -> Array:
    """Return MeanFlowQL's endpoint implied by its direct-map output."""

    direct_map = direct_map_fn(observations, state, current_time)
    return (1.0 - current_time) * state + current_time * direct_map


def native_reverse_rollout(
    velocity_fn: NativeVelocityFn,
    observations: Array,
    noises: Array,
    num_steps: int,
) -> tuple[Array, Array]:
    """Roll native interval velocities from noise at t=1 to t=0.

    Returned states are ordered by increasing time, so ``states[0]`` is the
    generated endpoint and ``states[-1]`` is the supplied noise.
    """

    if num_steps < 1:
        raise ValueError("num_steps must be positive.")
    state = noises
    descending_states = [state]
    batch_size = noises.shape[0]
    for index in range(num_steps, 0, -1):
        current_time = jnp.full(
            (batch_size, 1), index / num_steps, dtype=noises.dtype
        )
        target_time = jnp.full(
            (batch_size, 1), (index - 1) / num_steps, dtype=noises.dtype
        )
        velocity = velocity_fn(
            observations, state, target_time, current_time
        )
        state = state - (current_time - target_time) * velocity
        descending_states.append(state)
    states = jnp.stack(tuple(reversed(descending_states)), axis=0)
    return state, states


def meanflowql_reverse_rollout(
    direct_map_fn: DirectMapFn,
    observations: Array,
    noises: Array,
    num_steps: int,
) -> tuple[Array, Array]:
    """Segment MeanFlowQL's implied endpoint map into ``num_steps`` steps."""

    if num_steps < 1:
        raise ValueError("num_steps must be positive.")
    state = noises
    descending_states = [state]
    batch_size = noises.shape[0]
    for index in range(num_steps, 0, -1):
        current_time = jnp.full(
            (batch_size, 1), index / num_steps, dtype=noises.dtype
        )
        target_time = jnp.full(
            (batch_size, 1), (index - 1) / num_steps, dtype=noises.dtype
        )
        endpoint = meanflowql_endpoint(
            direct_map_fn, observations, state, current_time
        )
        fraction = (current_time - target_time) / current_time
        state = state + fraction * (endpoint - state)
        descending_states.append(state)
    states = jnp.stack(tuple(reversed(descending_states)), axis=0)
    return state, states


def _native_remaining_rollout(
    velocity_fn: NativeVelocityFn,
    observations: Array,
    state: Array,
    start_index: int,
    total_steps: int,
) -> Array:
    for index in range(start_index, 0, -1):
        current_time = jnp.full(
            (state.shape[0], 1),
            index / total_steps,
            dtype=state.dtype,
        )
        target_time = jnp.full(
            (state.shape[0], 1),
            (index - 1) / total_steps,
            dtype=state.dtype,
        )
        velocity = velocity_fn(
            observations, state, target_time, current_time
        )
        state = state - (current_time - target_time) * velocity
    return state


def _meanflowql_remaining_rollout(
    direct_map_fn: DirectMapFn,
    observations: Array,
    state: Array,
    start_index: int,
    total_steps: int,
) -> Array:
    for index in range(start_index, 0, -1):
        current_time = jnp.full(
            (state.shape[0], 1),
            index / total_steps,
            dtype=state.dtype,
        )
        target_time = jnp.full(
            (state.shape[0], 1),
            (index - 1) / total_steps,
            dtype=state.dtype,
        )
        endpoint = meanflowql_endpoint(
            direct_map_fn, observations, state, current_time
        )
        fraction = (current_time - target_time) / current_time
        state = state + fraction * (endpoint - state)
    return state


def _trajectory_geometry(states: Array, epsilon: float) -> dict[str, float]:
    displacements = states[1:] - states[:-1]
    path_length = jnp.sum(jnp.linalg.norm(displacements, axis=-1), axis=0)
    endpoint_distance = jnp.linalg.norm(states[-1] - states[0], axis=-1)

    if states.shape[0] < 3:
        turning_residual = jnp.asarray(0.0)
        active_fraction = jnp.asarray(0.0)
    else:
        left = displacements[:-1]
        right = displacements[1:]
        left_norm = jnp.linalg.norm(left, axis=-1)
        right_norm = jnp.linalg.norm(right, axis=-1)
        active = jnp.logical_and(left_norm > epsilon, right_norm > epsilon)
        cosine = jnp.sum(left * right, axis=-1) / jnp.maximum(
            left_norm * right_norm, epsilon
        )
        residual = 1.0 - jnp.clip(cosine, -1.0, 1.0)
        turning_residual = jnp.sum(residual * active) / jnp.maximum(
            jnp.sum(active), 1
        )
        active_fraction = jnp.mean(active)

    metrics = {
        "trajectory_turning_residual": _as_float(turning_residual),
        "trajectory_turning_active_fraction": _as_float(active_fraction),
        "trajectory_path_length_ratio": _as_float(
            jnp.mean(path_length / jnp.maximum(endpoint_distance, epsilon))
        ),
    }
    trajectory_steps = states.shape[0] - 1
    metrics[f"k{trajectory_steps}_turning_residual"] = metrics[
        "trajectory_turning_residual"
    ]
    metrics[f"k{trajectory_steps}_turning_active_fraction"] = metrics[
        "trajectory_turning_active_fraction"
    ]
    metrics[f"k{trajectory_steps}_path_length_ratio"] = metrics[
        "trajectory_path_length_ratio"
    ]
    return metrics


def _endpoint_map_metrics(
    endpoint_fn: EndpointFn,
    observations: Array,
    states: Array,
    trajectory_steps: int,
    inference_batch_size: int,
    epsilon: float,
) -> dict[str, float]:
    endpoint = states[0]
    errors = []
    reference_powers = []
    metrics = {}
    for index in range(1, trajectory_steps + 1):
        current_time = jnp.full(
            (observations.shape[0], 1),
            index / trajectory_steps,
            dtype=states.dtype,
        )
        predicted_chunks = []
        for start in range(0, observations.shape[0], inference_batch_size):
            stop = min(
                start + inference_batch_size, observations.shape[0]
            )
            predicted_chunks.append(
                endpoint_fn(
                    observations[start:stop],
                    states[index, start:stop],
                    current_time[start:stop],
                )
            )
        predicted_endpoint = jnp.concatenate(predicted_chunks, axis=0)
        error = jnp.mean(jnp.square(predicted_endpoint - endpoint))
        reference_power = jnp.mean(jnp.square(states[-1] - endpoint))
        errors.append(error)
        reference_powers.append(reference_power)
        metrics[f"endpoint_map_mse_t{index}"] = _as_float(error)

    mean_error = jnp.mean(jnp.stack(errors))
    mean_reference = jnp.mean(jnp.stack(reference_powers))
    metrics["endpoint_map_mse"] = _as_float(mean_error)
    metrics["endpoint_map_nmse"] = _as_float(
        mean_error / jnp.maximum(mean_reference, epsilon)
    )
    # Preserve the historical AM-MF evaluator names used by result tables.
    metrics["actual_endpoint_map_mse"] = metrics["endpoint_map_mse"]
    metrics["actual_endpoint_map_nmse"] = metrics["endpoint_map_nmse"]
    return metrics


def _endpoint_jvp_metrics(
    endpoint_fn: EndpointFn,
    remaining_rollout_fn: RemainingRolloutFn,
    observations: Array,
    states: Array,
    trajectory_steps: int,
    jacobian_probe_pairs: int,
    rng: Array,
    epsilon: float,
) -> dict[str, float]:
    probe_count = min(jacobian_probe_pairs, observations.shape[0])
    if probe_count == 0:
        return {"jacobian_probe_pairs": 0}

    probe_observations = observations[:probe_count]
    directions = jax.random.normal(
        rng, (probe_count, states.shape[-1]), dtype=states.dtype
    )
    directions = directions / jnp.maximum(
        jnp.linalg.norm(directions, axis=-1, keepdims=True), epsilon
    )
    errors = []
    reference_powers = []
    metrics: dict[str, float] = {}

    for index in range(1, trajectory_steps + 1):
        probe_states = states[index, :probe_count]
        current_time = jnp.full(
            (probe_count, 1),
            index / trajectory_steps,
            dtype=states.dtype,
        )

        def endpoint_batch(current_state):
            return endpoint_fn(
                probe_observations, current_state, current_time
            )

        def rollout_batch(current_state):
            return remaining_rollout_fn(
                probe_observations,
                current_state,
                index,
                trajectory_steps,
            )

        _, endpoint_jvp = jax.jvp(
            endpoint_batch, (probe_states,), (directions,)
        )
        _, rollout_jvp = jax.jvp(
            rollout_batch, (probe_states,), (directions,)
        )
        error = jnp.mean(jnp.square(endpoint_jvp - rollout_jvp))
        reference_power = jnp.mean(jnp.square(rollout_jvp))
        errors.append(error)
        reference_powers.append(reference_power)
        metrics[f"endpoint_jvp_mse_t{index}"] = _as_float(error)
        metrics[f"k{trajectory_steps}_endpoint_jvp_mse_t{index}"] = (
            metrics[f"endpoint_jvp_mse_t{index}"]
        )

    mean_error = jnp.mean(jnp.stack(errors))
    mean_reference = jnp.mean(jnp.stack(reference_powers))
    metrics["endpoint_jvp_mse"] = _as_float(mean_error)
    metrics["endpoint_jvp_nmse"] = _as_float(
        mean_error / jnp.maximum(mean_reference, epsilon)
    )
    metrics[f"k{trajectory_steps}_endpoint_jvp_mse"] = metrics[
        "endpoint_jvp_mse"
    ]
    metrics[f"k{trajectory_steps}_endpoint_jvp_nmse"] = metrics[
        "endpoint_jvp_nmse"
    ]
    metrics["jacobian_probe_pairs"] = int(probe_count)
    return metrics


def _native_split_metrics(
    velocity_fn: NativeVelocityFn,
    observations: Array,
    states: Array,
    trajectory_steps: int,
    split_triplets: Iterable[tuple[int, int, int]],
    inference_batch_size: int,
    epsilon: float,
) -> dict[str, float]:
    errors = []
    reference_powers = []
    metrics = {}
    for r_index, s_index, t_index in split_triplets:
        if not 0 <= r_index < s_index < t_index <= trajectory_steps:
            raise ValueError(
                "split triplets must satisfy "
                "0 <= r < s < t <= trajectory_steps."
            )
        r = jnp.full(
            (observations.shape[0], 1),
            r_index / trajectory_steps,
            dtype=states.dtype,
        )
        s = jnp.full(
            (observations.shape[0], 1),
            s_index / trajectory_steps,
            dtype=states.dtype,
        )
        t = jnp.full(
            (observations.shape[0], 1),
            t_index / trajectory_steps,
            dtype=states.dtype,
        )
        full_chunks = []
        split_chunks = []
        for start in range(0, observations.shape[0], inference_batch_size):
            stop = min(
                start + inference_batch_size, observations.shape[0]
            )
            obs_chunk = observations[start:stop]
            state_t = states[t_index, start:stop]
            state_s = states[s_index, start:stop]
            r_chunk = r[start:stop]
            s_chunk = s[start:stop]
            t_chunk = t[start:stop]
            full_chunks.append(
                (t_chunk - r_chunk)
                * velocity_fn(
                    obs_chunk, state_t, r_chunk, t_chunk
                )
            )
            split_chunks.append(
                (t_chunk - s_chunk)
                * velocity_fn(
                    obs_chunk, state_t, s_chunk, t_chunk
                )
                + (s_chunk - r_chunk)
                * velocity_fn(
                    obs_chunk, state_s, r_chunk, s_chunk
                )
            )
        full_displacement = jnp.concatenate(full_chunks, axis=0)
        split_displacement = jnp.concatenate(split_chunks, axis=0)
        error = jnp.mean(
            jnp.square(full_displacement - split_displacement)
        )
        reference_power = jnp.mean(jnp.square(split_displacement))
        errors.append(error)
        reference_powers.append(reference_power)
        metrics[
            f"split_consistency_mse_r{r_index}_s{s_index}_t{t_index}"
        ] = _as_float(error)

    if not errors:
        raise ValueError("split_triplets must contain at least one triplet.")
    mean_error = jnp.mean(jnp.stack(errors))
    mean_reference = jnp.mean(jnp.stack(reference_powers))
    metrics["split_consistency_mse"] = _as_float(mean_error)
    metrics["split_consistency_nmse"] = _as_float(
        mean_error / jnp.maximum(mean_reference, epsilon)
    )
    return metrics


def _evaluate(
    *,
    family: str,
    observations: Array,
    noises: Array,
    endpoint_fn: EndpointFn,
    rollout_fn: RolloutFn,
    remaining_rollout_fn: RemainingRolloutFn,
    native_velocity_fn: NativeVelocityFn | None,
    nfe_values: Sequence[int],
    trajectory_steps: int,
    jacobian_probe_pairs: int,
    split_triplets: Iterable[tuple[int, int, int]],
    inference_batch_size: int | None,
    rng: Array,
    epsilon: float,
) -> dict[str, float | int | str]:
    normalized_nfes = _validate_inputs(
        observations,
        noises,
        nfe_values,
        trajectory_steps,
        jacobian_probe_pairs,
    )
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    batch_size = _resolve_batch_size(
        observations.shape[0], inference_batch_size
    )

    finals = {}
    for nfe in normalized_nfes:
        finals[nfe] = _batched_rollout(
            rollout_fn, observations, noises, nfe, batch_size
        )[0]
    _, states = _batched_rollout(
        rollout_fn,
        observations,
        noises,
        trajectory_steps,
        batch_size,
    )

    metrics: dict[str, float | int | str] = {
        "family": family,
        "num_pairs": int(observations.shape[0]),
        "trajectory_steps": int(trajectory_steps),
    }
    reference_final = finals[1]
    for nfe, final in finals.items():
        metrics[f"k{nfe}_action_std"] = _as_float(jnp.std(final))
        metrics[f"k{nfe}_clip_fraction"] = _as_float(
            jnp.mean(jnp.abs(final) > 1.0)
        )
        if nfe != 1:
            metrics[f"k1_k{nfe}_mse"] = _as_float(
                jnp.mean(jnp.square(reference_final - final))
            )

    metrics.update(_trajectory_geometry(states, epsilon))
    metrics.update(
        _endpoint_map_metrics(
            endpoint_fn,
            observations,
            states,
            trajectory_steps,
            batch_size,
            epsilon,
        )
    )
    metrics.update(
        _endpoint_jvp_metrics(
            endpoint_fn,
            remaining_rollout_fn,
            observations,
            states,
            trajectory_steps,
            jacobian_probe_pairs,
            rng,
            epsilon,
        )
    )
    if native_velocity_fn is not None:
        metrics.update(
            _native_split_metrics(
                native_velocity_fn,
                observations,
                states,
                trajectory_steps,
                split_triplets,
                batch_size,
                epsilon,
            )
        )
    return metrics


def evaluate_native_meanflow_consistency(
    velocity_fn: NativeVelocityFn,
    observations: Array,
    noises: Array,
    *,
    nfe_values: Sequence[int] = DEFAULT_NFES,
    trajectory_steps: int = 10,
    jacobian_probe_pairs: int = 128,
    split_triplets: Iterable[tuple[int, int, int]] | None = None,
    inference_batch_size: int | None = None,
    rng: Array | None = None,
    epsilon: float = 1e-8,
) -> dict[str, float | int | str]:
    """Evaluate the full Native MeanFlow consistency contract."""

    if rng is None:
        rng = jax.random.PRNGKey(0)
    if split_triplets is None:
        split_triplets = default_split_triplets(trajectory_steps)
    def endpoint_fn(obs, state, time):
        return native_endpoint(velocity_fn, obs, state, time)

    def rollout_fn(obs, noise, steps):
        return native_reverse_rollout(velocity_fn, obs, noise, steps)

    def remaining_fn(obs, state, index, steps):
        return _native_remaining_rollout(
            velocity_fn, obs, state, index, steps
        )
    return _evaluate(
        family="native_meanflow",
        observations=observations,
        noises=noises,
        endpoint_fn=endpoint_fn,
        rollout_fn=rollout_fn,
        remaining_rollout_fn=remaining_fn,
        native_velocity_fn=velocity_fn,
        nfe_values=nfe_values,
        trajectory_steps=trajectory_steps,
        jacobian_probe_pairs=jacobian_probe_pairs,
        split_triplets=split_triplets,
        inference_batch_size=inference_batch_size,
        rng=rng,
        epsilon=epsilon,
    )


def evaluate_meanflowql_consistency(
    direct_map_fn: DirectMapFn,
    observations: Array,
    noises: Array,
    *,
    nfe_values: Sequence[int] = DEFAULT_NFES,
    trajectory_steps: int = 10,
    jacobian_probe_pairs: int = 128,
    inference_batch_size: int | None = None,
    rng: Array | None = None,
    epsilon: float = 1e-8,
) -> dict[str, float | int | str]:
    """Evaluate consistency of MeanFlowQL's implied endpoint map."""

    if rng is None:
        rng = jax.random.PRNGKey(0)
    def endpoint_fn(obs, state, time):
        return meanflowql_endpoint(direct_map_fn, obs, state, time)

    def rollout_fn(obs, noise, steps):
        return meanflowql_reverse_rollout(direct_map_fn, obs, noise, steps)

    def remaining_fn(obs, state, index, steps):
        return _meanflowql_remaining_rollout(
            direct_map_fn, obs, state, index, steps
        )
    return _evaluate(
        family="meanflowql",
        observations=observations,
        noises=noises,
        endpoint_fn=endpoint_fn,
        rollout_fn=rollout_fn,
        remaining_rollout_fn=remaining_fn,
        native_velocity_fn=None,
        nfe_values=nfe_values,
        trajectory_steps=trajectory_steps,
        jacobian_probe_pairs=jacobian_probe_pairs,
        split_triplets=(),
        inference_batch_size=inference_batch_size,
        rng=rng,
        epsilon=epsilon,
    )


def evaluate_agent_consistency(
    agent: Any,
    observations: Array,
    noises: Array,
    *,
    nfe_values: Sequence[int] = DEFAULT_NFES,
    trajectory_steps: int = 10,
    jacobian_probe_pairs: int = 128,
    split_triplets: Iterable[tuple[int, int, int]] | None = None,
    inference_batch_size: int | None = None,
    use_target_actor: bool = False,
    rng: Array | None = None,
) -> dict[str, float | int | str]:
    """Dispatch consistency evaluation from an AM-MF/MeanFlowQL agent."""

    agent_name = str(agent.config["agent_name"])
    if agent_name in {"native_meanflow", "am_meanflow"}:
        def velocity_fn(obs, state, r, t):
            return agent._velocity(
                obs, state, r, t, target=use_target_actor
            )

        metrics = evaluate_native_meanflow_consistency(
            velocity_fn,
            observations,
            noises,
            nfe_values=nfe_values,
            trajectory_steps=trajectory_steps,
            jacobian_probe_pairs=jacobian_probe_pairs,
            split_triplets=split_triplets,
            inference_batch_size=inference_batch_size,
            rng=rng,
        )
    elif agent_name in {
        "meanflowql",
        "meanflowql_beta",
        "am_meanflow_target_changed",
    }:
        if use_target_actor:
            raise ValueError(
                "use_target_actor is only available for Native MeanFlow "
                "agents; MeanFlowQL has no EMA target actor."
            )
        actor = agent.network.select("actor_bc_flow")
        encoder = (
            agent.network.select("actor_bc_flow_encoder")
            if agent.config.get("encoder") is not None
            else None
        )

        def direct_map_fn(obs, state, time):
            if encoder is not None:
                encoded_obs = encoder(obs)
                return actor(
                    encoded_obs, state, time, is_encoded=True
                )
            return actor(obs, state, time)

        metrics = evaluate_meanflowql_consistency(
            direct_map_fn,
            observations,
            noises,
            nfe_values=nfe_values,
            trajectory_steps=trajectory_steps,
            jacobian_probe_pairs=jacobian_probe_pairs,
            inference_batch_size=inference_batch_size,
            rng=rng,
        )
    else:
        raise ValueError(
            f"Agent {agent_name!r} is not a supported MeanFlow family."
        )
    metrics["agent_name"] = agent_name
    metrics["use_target_actor"] = int(use_target_actor)
    return metrics


def judge_consistency(
    metrics: Mapping[str, float | int | str],
    thresholds: ConsistencyThresholds,
) -> dict[str, Any]:
    """Apply explicit thresholds without inventing a universal cutoff."""

    specifications = {
        "endpoint_map_nmse": thresholds.endpoint_map_nmse,
        "endpoint_jvp_nmse": thresholds.endpoint_jvp_nmse,
        "split_consistency_nmse": thresholds.split_consistency_nmse,
    }
    k_reference = max(
        (
            int(key.removeprefix("k1_k").removesuffix("_mse"))
            for key in metrics
            if key.startswith("k1_k") and key.endswith("_mse")
        ),
        default=None,
    )
    if thresholds.k1_k_reference_mse is not None:
        if k_reference is None:
            raise ValueError("No K1-vs-K metric is available for judgement.")
        specifications[f"k1_k{k_reference}_mse"] = (
            thresholds.k1_k_reference_mse
        )

    checks = {}
    for metric_name, maximum in specifications.items():
        if maximum is None:
            continue
        if maximum < 0.0:
            raise ValueError("Consistency thresholds must be non-negative.")
        if metric_name not in metrics:
            raise ValueError(
                f"Metric {metric_name!r} is unavailable for this agent family."
            )
        value = float(metrics[metric_name])
        checks[metric_name] = {
            "value": value,
            "maximum": float(maximum),
            "passed": bool(value <= maximum),
        }

    if not checks:
        return {"status": "not_requested", "passed": None, "checks": {}}
    return {
        "status": "passed" if all(v["passed"] for v in checks.values()) else "failed",
        "passed": all(value["passed"] for value in checks.values()),
        "checks": checks,
    }
