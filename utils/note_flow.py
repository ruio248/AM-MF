"""Reverse-time interval maps and a fixed-grid, differentiable RK2 reference.

Time 1 is noise; time 0 is the action. A velocity is action-to-noise oriented.
These helpers are used only by the Note agent; the upstream MeanFlowQL runner,
dataset pipeline, evaluator, and RNG behavior remain unchanged.
"""

import jax
import jax.numpy as jnp


def interval_map(x, r, t, g):
    """Return the residual-parameterized map F^{r<-t}(x)."""
    return x - (t - r) * (x - g)


def endpoint_adjoint(endpoint_fn, reward_fn, x):
    """Pull a terminal reward gradient back through an endpoint map."""
    endpoint, pullback = jax.vjp(endpoint_fn, x)
    reward_gradient = jax.grad(lambda a: jnp.sum(reward_fn(a)))(endpoint)
    return endpoint, pullback(reward_gradient)[0]


def controlled_velocity(prior_velocity, adjoint, eta):
    """Apply the time-weighted local proximal control derived in the Note."""
    return prior_velocity - eta * adjoint


def midpoint_step(field, x, t, dt):
    midpoint = x - 0.5 * dt * field(x, t)
    return x - dt * field(midpoint, t - 0.5 * dt)


def rk2_path(field, x, steps=8, start_index=0, end_index=None):
    """Compose fixed global-grid RK2 maps and return every visited state."""
    if steps < 1:
        raise ValueError("steps must be positive")
    end_index = steps if end_index is None else end_index
    if not 0 <= start_index <= end_index <= steps:
        raise ValueError("invalid interval indices")
    dt = jnp.asarray(1.0 / steps, dtype=x.dtype)

    def advance(state, index):
        t = jnp.full((state.shape[0], 1), 1.0 - index / steps, dtype=state.dtype)
        following = midpoint_step(field, state, t, dt)
        return following, following

    _, states = jax.lax.scan(advance, x, jnp.arange(start_index, end_index))
    return jnp.concatenate([x[None], states], axis=0)


def directional_difference(function, x, direction, epsilon=1e-3):
    """Centered finite-difference directional derivative."""
    return (function(x + epsilon * direction) - function(x - epsilon * direction)) / (2 * epsilon)


def rk2_endpoint_from_grid(field, x, start_indices, steps=8):
    """Evaluate batched suffix maps without changing the global RK2 grid."""
    dt = jnp.asarray(1.0 / steps, dtype=x.dtype)

    def advance(state, index):
        t = jnp.full((len(x), 1), 1.0 - index / steps, dtype=x.dtype)
        following = midpoint_step(field, state, t, dt)
        return jnp.where((index >= start_indices)[:, None], following, state), None

    endpoint, _ = jax.lax.scan(advance, x, jnp.arange(steps))
    return endpoint


def clipping_metrics(raw):
    excess = jnp.abs(raw - jnp.clip(raw, -1, 1))
    return {
        "any_oob": jnp.mean(jnp.any(excess > 0, axis=-1)),
        "coordinate_oob": jnp.mean(excess > 0),
        "mean_clip": jnp.mean(excess),
        "max_clip": jnp.max(excess),
    }


def sensitivity_errors(map_a, map_b, x, reward_fn):
    """Separate endpoint, Jacobian, and adjoint errors for diagnostics."""
    fa, pa = endpoint_adjoint(map_a, reward_fn, x)
    fb, pb = endpoint_adjoint(map_b, reward_fn, x)
    ja = jax.vmap(jax.jacrev(lambda z: map_a(z[None])[0]))(x)
    jb = jax.vmap(jax.jacrev(lambda z: map_b(z[None])[0]))(x)
    na = jnp.linalg.norm(pa, axis=-1)
    nb = jnp.linalg.norm(pb, axis=-1)
    valid = (na > 1e-8) & (nb > 1e-8)
    cosine = jnp.sum(pa * pb, axis=-1) / jnp.maximum(na * nb, 1e-8)
    return {
        "endpoint_rms": jnp.sqrt(jnp.mean((fa - fb) ** 2)),
        "jacobian_rms": jnp.sqrt(jnp.mean((ja - jb) ** 2)),
        "adjoint_relative": jnp.linalg.norm(pa - pb) / jnp.maximum(jnp.linalg.norm(pb), 1e-8),
        "adjoint_cosine": jnp.sum(jnp.where(valid, cosine, 0)) / jnp.maximum(jnp.sum(valid), 1),
        "adjoint_valid_fraction": jnp.mean(valid),
    }
