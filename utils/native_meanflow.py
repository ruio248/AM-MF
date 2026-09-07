"""Core mathematics for the native MeanFlow parameterization.

The repository's original MeanFlowQL agent predicts a reformulated endpoint.
This module keeps the average velocity explicit as u(x_t, r, t) so the
MeanFlow identity and reverse-time transport can be tested independently of
the reinforcement-learning agent.
"""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp


Array = jax.Array


def _sample_times(
    rng: Array,
    shape: Tuple[int, ...],
    distribution: str,
    logit_mean: float,
    logit_std: float,
) -> Array:
    if distribution == "uniform":
        return jax.random.uniform(rng, shape, minval=0.0, maxval=1.0)
    if distribution == "logit_normal":
        normal = jax.random.normal(rng, shape)
        return jax.nn.sigmoid(normal * logit_std + logit_mean)
    raise ValueError(
        f"Unsupported time distribution {distribution!r}; "
        "expected 'uniform' or 'logit_normal'."
    )


def sample_time_pairs(
    rng: Array,
    batch_size: int,
    flow_ratio: float = 0.5,
    distribution: str = "logit_normal",
    logit_mean: float = -0.4,
    logit_std: float = 1.0,
) -> Tuple[Array, Array, Array]:
    """Sample 0 <= r <= t <= 1 and optionally impose r == t.

    flow_ratio is the probability of drawing an instantaneous-flow sample.
    Those samples enforce the boundary condition u(x_t, t, t) = v_t.
    """

    if not 0.0 <= flow_ratio <= 1.0:
        raise ValueError(f"flow_ratio must be in [0, 1], got {flow_ratio}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if logit_std <= 0.0:
        raise ValueError(f"logit_std must be positive, got {logit_std}.")

    first_rng, second_rng, mask_rng = jax.random.split(rng, 3)
    shape = (batch_size, 1)
    first = _sample_times(first_rng, shape, distribution, logit_mean, logit_std)
    second = _sample_times(second_rng, shape, distribution, logit_mean, logit_std)
    t = jnp.maximum(first, second)
    r = jnp.minimum(first, second)

    instantaneous_mask = jax.random.bernoulli(
        mask_rng, p=flow_ratio, shape=shape
    )
    r = jnp.where(instantaneous_mask, t, r)
    return t, r, instantaneous_mask


def meanflow_jvp_target(
    field_fn: Callable[[Array, Array, Array], Array],
    x_t: Array,
    r: Array,
    t: Array,
    conditional_velocity: Array,
) -> Tuple[Array, Array, Array]:
    """Evaluate the native MeanFlow identity with a JVP.

    The total derivative follows dx_t / dt = conditional_velocity while
    holding the interval's target time r fixed:

        d u / dt = J_x u @ conditional_velocity + partial_t u.
    """

    u_pred, du_dt = jax.jvp(
        field_fn,
        (x_t, r, t),
        (
            conditional_velocity,
            jnp.zeros_like(r),
            jnp.ones_like(t),
        ),
    )
    u_target = conditional_velocity - (t - r) * du_dt
    return u_pred, jax.lax.stop_gradient(u_target), du_dt


def endpoint_from_average_velocity(noise: Array, average_velocity: Array) -> Array:
    """Map noise at t=1 to the action endpoint at r=0."""

    return noise - average_velocity


def reverse_transport_step(
    x_t: Array,
    average_velocity: Array,
    r: Array,
    t: Array,
) -> Array:
    """Transport x_t backward from time t to r in one step."""

    return x_t - (t - r) * average_velocity
