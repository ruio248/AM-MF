"""Pure AM-MF target construction utilities.

The native MeanFlow implementation in :mod:`utils.native_meanflow` uses the
action-to-noise time convention.  AM-MF is easier to state in the reverse,
noise-to-action convention.  This module makes that conversion explicit and
keeps the AlphaFlow/adjoint target independent from the agent implementation.
"""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp


Array = jax.Array


def alpha_flow_schedule(
    step: Array,
    start_step: int,
    warmup_steps: int,
    transition_steps: int,
    floor: float,
    gamma: float,
) -> Array:
    """Anneal AlphaFlow from the reward target to EMA consistency.

    ``alpha=1`` selects the frozen-reference/adjoint target and ``alpha=0``
    selects the EMA bootstrap target.  A normalized sigmoid gives exact
    endpoint values while retaining a smooth transition.
    """

    if start_step < 0:
        raise ValueError(f"start_step must be non-negative, got {start_step}.")
    if warmup_steps < 0:
        raise ValueError(
            f"warmup_steps must be non-negative, got {warmup_steps}."
        )
    if transition_steps <= 0:
        raise ValueError(
            "transition_steps must be positive, got "
            f"{transition_steps}."
        )
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"floor must be in [0, 1], got {floor}.")
    if gamma <= 0.0:
        raise ValueError(f"gamma must be positive, got {gamma}.")

    step = jnp.asarray(step, dtype=jnp.float32)
    after_start = jnp.maximum(step - float(start_step), 0.0)
    progress = jnp.clip(
        (after_start - float(warmup_steps)) / float(transition_steps),
        0.0,
        1.0,
    )
    high = jax.nn.sigmoid(0.5 * gamma)
    low = jax.nn.sigmoid(-0.5 * gamma)
    sigmoid_value = jax.nn.sigmoid(gamma * (0.5 - progress))
    normalized = (sigmoid_value - low) / jnp.maximum(high - low, 1e-8)
    annealed = floor + (1.0 - floor) * normalized
    return jnp.where(after_start < float(warmup_steps), 1.0, annealed)


def native_interval_from_generative(
    start_time: Array, end_time: Array
) -> Tuple[Array, Array]:
    """Map a noise-to-action interval to native action-to-noise time.

    For a generative interval ``start -> end``, the corresponding native
    interval is ``(1-end) <- (1-start)``.  The return order matches the native
    actor signature ``u(x_t, r, t)``: target time first, current time second.
    """

    return 1.0 - end_time, 1.0 - start_time


def generative_velocity_from_native(native_velocity: Array) -> Array:
    """Reverse the sign when changing the direction of MeanFlow time."""

    return -native_velocity


def endpoint_map(
    state: Array, start_time: Array, average_velocity: Array
) -> Array:
    """Map a state at generative time ``start_time`` to the action endpoint."""

    return state + (1.0 - start_time) * average_velocity


def endpoint_reward_adjoint(
    endpoint_fn: Callable[[Array], Array],
    reward_fn: Callable[[Array], Array],
    state: Array,
) -> Tuple[Array, Array]:
    """Return ``F(state)`` and ``J_F(state)^T grad reward(F(state))``.

    ``reward_fn`` may return either a scalar or independent per-example
    rewards.  Summing the latter produces the same per-example gradient while
    keeping the helper convenient for batched critics.
    """

    endpoint, pullback = jax.vjp(endpoint_fn, state)
    reward_grad = jax.grad(lambda action: jnp.sum(reward_fn(action)))(endpoint)
    return endpoint, pullback(reward_grad)[0]


def adjoint_corrected_velocity(
    base_velocity: Array,
    adjoint: Array,
    eta: float,
    interval: Array,
) -> Array:
    """Apply the AM-MF interval-scaled reward correction."""

    if eta < 0.0:
        raise ValueError(f"eta must be non-negative, got {eta}.")
    return base_velocity + eta * interval * adjoint


def meanflowql_adjoint_guided_velocity(
    conditional_velocity: Array,
    adjoint: Array,
    eta: float,
    remaining_time: Array,
) -> Array:
    """Inject AM endpoint guidance into MeanFlowQL's path velocity.

    MeanFlowQL uses the action-to-noise path ``v = noise - action``.  The AM
    adjoint points in the action-improving reverse direction, hence the minus
    sign.  ``remaining_time`` preserves the original AM interval scaling; for
    the official MeanFlowQL target with ``b=0`` it is simply ``t``.
    """

    if eta < 0.0:
        raise ValueError(f"eta must be non-negative, got {eta}.")
    return conditional_velocity - eta * remaining_time * adjoint


def meanflowql_endpoint_map(
    state: Array,
    current_time: Array,
    direct_map: Array,
) -> Array:
    """Recover MeanFlowQL's implied action endpoint from ``g(x_t, t)``.

    MeanFlowQL parameterizes ``g = x_t - u``.  The native one-step endpoint
    ``x_t - t*u`` is therefore ``(1-t)*x_t + t*g``.  At ``t=1`` this reduces
    to the direct policy output used by the repository's sampler.
    """

    return (1.0 - current_time) * state + current_time * direct_map


def meanflowql_reformulated_target(
    state: Array,
    target_time: Array,
    current_time: Array,
    guided_velocity: Array,
    total_derivative: Array,
) -> Array:
    """Apply MeanFlowQL's reformulated identity to an AM-guided path.

    ``total_derivative`` must be the JVP of
    ``g_theta(o, x_t, t)`` along ``(guided_velocity, 1)`` while
    ``target_time`` (the paper's ``b``) is fixed.  The repository's official
    MeanFlowQL implementation uses ``target_time=0``.
    """

    interval = current_time - target_time
    return (
        state
        + (interval - 1.0) * guided_velocity
        - interval * total_derivative
    )


def meanflowql_alphaflow_state(
    state: Array,
    current_time: Array,
    alpha: Array,
    bootstrap_direct_map: Array,
) -> Tuple[Array, Array, Array]:
    """Split a MeanFlowQL reverse path with the AlphaFlow coefficient.

    MeanFlowQL uses action-to-noise time, the reverse of the convention used
    by the original AlphaFlow note.  Therefore the note's intermediate time
    becomes ``s_alpha = alpha * t`` when the target time is zero.  The EMA
    direct map supplies the average velocity for the bootstrap segment from
    ``t`` to ``s_alpha``.

    Returns ``(s_alpha, x_s, u_boot)`` where ``u_boot = x_t - g_bar`` and
    ``x_s = x_t - (t - s_alpha) * u_boot``.
    """

    alpha = jnp.asarray(alpha, dtype=state.dtype)
    intermediate_time = alpha * current_time
    bootstrap_velocity = state - bootstrap_direct_map
    intermediate_state = state - (
        current_time - intermediate_time
    ) * bootstrap_velocity
    return intermediate_time, intermediate_state, bootstrap_velocity


def meanflowql_alphaflow_target(
    state: Array,
    intermediate_state: Array,
    reward_direct_target: Array,
    bootstrap_direct_map: Array,
    alpha: Array,
) -> Tuple[Array, Array, Array]:
    """Mix changed-target AM and EMA consistency in velocity space.

    ``g = x - u`` converts a MeanFlowQL direct-map target to the average
    velocity used by the original AlphaFlow note.  This helper applies the
    note's ``alpha * u_reward + (1-alpha) * u_boot`` mixture and maps the
    result back to a direct-map target at ``state``.
    """

    alpha = jnp.asarray(alpha, dtype=state.dtype)
    reward_velocity = intermediate_state - reward_direct_target
    bootstrap_velocity = state - bootstrap_direct_map
    mixed_velocity = compose_am_target(
        reward_velocity, bootstrap_velocity, alpha
    )
    return state - mixed_velocity, reward_velocity, bootstrap_velocity


def meanflow_endpoint_jvp_target(
    local_velocity: Array,
    start_time: Array,
    total_derivative: Array,
) -> Array:
    """Return the explicit ``alpha=0`` endpoint-MeanFlow JVP target.

    In the generative noise-to-action convention, endpoint consistency of
    ``F(x_t,t)=x_t+(1-t)u(x_t,t,1)`` gives

    ``u_target = v + (1-t) * d u_bar / dt``.
    """

    return local_velocity + (1.0 - start_time) * total_derivative


def compose_am_target(
    reward_velocity: Array,
    bootstrap_velocity: Array,
    alpha: Array,
) -> Array:
    """Construct the AlphaFlow mixture used as the AM-MF actor target."""

    alpha = jnp.asarray(alpha, dtype=reward_velocity.dtype)
    return alpha * reward_velocity + (1.0 - alpha) * bootstrap_velocity


def positive_alpha_regression_loss(
    prediction: Array,
    reward_velocity: Array,
    bootstrap_velocity: Array,
    alpha: Array,
    alpha_eps: float = 1e-4,
    normalize_by_alpha: bool = True,
) -> Tuple[Array, Array]:
    """Regress onto the stopped AM target for strictly positive ``alpha``.

    The Note-MF equation contains a ``1 / alpha`` factor, while some
    pseudocode versions omit it.  ``normalize_by_alpha`` exposes that choice
    explicitly.  The agent handles exact ``alpha=0`` in a separate JVP branch;
    this helper must not be used as a numerical approximation to that limit.
    """

    if alpha_eps <= 0.0:
        raise ValueError(f"alpha_eps must be positive, got {alpha_eps}.")

    alpha = jnp.asarray(alpha, dtype=prediction.dtype)
    target = compose_am_target(
        reward_velocity, bootstrap_velocity, alpha
    )
    target = jax.lax.stop_gradient(target)
    target_error = jnp.mean(jnp.square(prediction - target), axis=-1)

    if normalize_by_alpha:
        per_sample = target_error / jnp.maximum(alpha, alpha_eps)
    else:
        per_sample = target_error
    return jnp.mean(per_sample), target


def critical_alpha(
    reward_error: Array,
    bootstrap_error: Array,
    fallback: float = 1.0,
) -> Tuple[Array, Array, Array, Array]:
    """Return the bounded quadratic-mixture diagnostic from Note-MF."""

    a = jnp.mean(jnp.sum(jnp.square(reward_error), axis=-1))
    b = jnp.mean(jnp.sum(reward_error * bootstrap_error, axis=-1))
    c = jnp.mean(jnp.sum(jnp.square(bootstrap_error), axis=-1))
    denominator = jnp.maximum(a - 2.0 * b + c, 0.0)
    value = jnp.clip(
        (c - b) / jnp.maximum(denominator, 1e-8), 0.0, 1.0
    )
    alpha = jnp.where(denominator > 1e-8, value, fallback)
    return jax.lax.stop_gradient(alpha), a, b, c
