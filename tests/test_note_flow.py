import unittest

import jax
import jax.numpy as jnp
import numpy as np

from utils.note_flow import (
    clipping_metrics,
    controlled_velocity,
    directional_difference,
    endpoint_adjoint,
    interval_map,
    rk2_endpoint_from_grid,
    rk2_path,
    sensitivity_errors,
)


class NoteFlowMathTest(unittest.TestCase):
    def test_coordinate_boundaries(self):
        x, g = jnp.array([[2.0, -1.0]]), jnp.array([[0.4, 0.2]])
        np.testing.assert_allclose(interval_map(x, 0.0, 1.0, g), g, atol=1e-7)
        np.testing.assert_allclose(interval_map(x, 0.3, 0.3, g), x)
        np.testing.assert_allclose(
            interval_map(x, 0.2, 0.6, g), x - 0.4 * (x - g)
        )

    def test_original_scaling_counterexample_and_repair(self):
        eta, gradient = 0.1, jnp.array([[1.0, -2.0]])
        original_full = eta * gradient
        original_halves = 2 * eta * 0.5**2 * gradient
        self.assertGreater(
            float(jnp.linalg.norm(original_full - original_halves)), 0.01
        )
        field = lambda x, t: controlled_velocity(
            jnp.zeros_like(x), gradient, eta
        )
        for steps in (1, 2, 8, 16):
            endpoint = rk2_path(field, jnp.zeros_like(gradient), steps)[-1]
            np.testing.assert_allclose(endpoint, eta * gradient, atol=1e-6)
            self.assertGreater(float(jnp.sum(endpoint * gradient)), 0)

    def test_semigroup_uses_one_global_grid(self):
        field = lambda x, t: jnp.sin(x) + 0.2 * t
        x = jnp.array([[0.4, -0.2]])
        whole = rk2_path(field, x, 8)[-1]
        middle = rk2_path(field, x, 8, end_index=3)[-1]
        composed = rk2_path(field, middle, 8, start_index=3)[-1]
        np.testing.assert_allclose(whole, composed, atol=1e-7)
        np.testing.assert_allclose(rk2_path(field, x, 8, 3, 3)[-1], x)

    def test_affine_adjoint_matches_finite_difference(self):
        matrix = jnp.array([[0.8, 0.3], [-0.2, 1.1]])
        endpoint = lambda x: x @ matrix.T + 0.1
        reward = lambda a: -0.5 * jnp.sum((a - 0.4) ** 2, axis=-1)
        x = jnp.array([[0.3, -0.1]])
        value, adjoint = endpoint_adjoint(endpoint, reward, x)
        np.testing.assert_allclose(adjoint, (0.4 - value) @ matrix, atol=1e-6)
        direction = jnp.array([[1.0, -1.0]])
        numerical = directional_difference(
            lambda z: reward(endpoint(z)), x, direction
        )
        np.testing.assert_allclose(
            numerical, jnp.sum(adjoint * direction, axis=-1), atol=2e-5
        )

    def test_curved_flow_and_suffix_jacobian(self):
        field = lambda x, t: jnp.stack([-x[:, 1], x[:, 0]], axis=-1)
        c, s = jnp.cos(1.0), jnp.sin(1.0)
        matrix = jnp.array([[c, s], [-s, c]])
        exact = lambda x: x @ matrix.T
        x = jnp.array([[0.5, -0.3]])
        errors = []
        for steps in (8, 16, 32):
            errors.append(float(jnp.linalg.norm(rk2_path(field, x, steps)[-1] - exact(x))))
        self.assertLess(errors[1], errors[0] / 3)
        self.assertLess(errors[2], errors[1] / 3)

        initial = jnp.array([[0.4, -0.2], [0.3, 0.7], [-0.2, 0.8]])
        path = rk2_path(lambda z, t: jnp.sin(z) + 0.2 * t, initial, 8)
        indices = jnp.array([0, 4, 7])
        states = path[indices, jnp.arange(3)]
        suffix = lambda z: rk2_endpoint_from_grid(
            lambda state, time: jnp.sin(state) + 0.2 * time,
            z,
            indices,
            8,
        )
        np.testing.assert_allclose(suffix(states), path[-1], atol=1e-7)
        direction = jnp.ones_like(states)
        exact_jvp = jax.jvp(suffix, (states,), (direction,))[1]
        finite_difference = directional_difference(suffix, states, direction)
        np.testing.assert_allclose(exact_jvp, finite_difference, atol=1e-4)

    def test_endpoint_and_jacobian_errors_are_distinct(self):
        x = jnp.zeros((1, 2))
        identity = lambda z: z
        reward = lambda a: jnp.sum(a, axis=-1)
        map_bias = sensitivity_errors(lambda z: z + 0.2, identity, x, reward)
        jacobian_bias = sensitivity_errors(lambda z: 2 * z, identity, x, reward)
        self.assertGreater(float(map_bias["endpoint_rms"]), 0.19)
        self.assertEqual(float(map_bias["jacobian_rms"]), 0.0)
        self.assertEqual(float(jacobian_bias["endpoint_rms"]), 0.0)
        self.assertGreater(float(jacobian_bias["adjoint_relative"]), 0.99)

    def test_clip_chain_and_bound_gradient(self):
        x = jnp.array([[2.0, 0.5, -2.0]])
        _, adjoint = endpoint_adjoint(
            lambda z: z,
            lambda a: jnp.clip(a, -1, 1).sum(axis=-1),
            x,
        )
        np.testing.assert_allclose(adjoint, [[0.0, 1.0, 0.0]])
        bound = lambda z: (
            jax.nn.relu(z - 1) + jax.nn.relu(-1 - z)
        ).sum()
        np.testing.assert_allclose(jax.grad(bound)(x), [[1.0, 0.0, -1.0]])
        metrics = clipping_metrics(x)
        self.assertEqual(float(metrics["any_oob"]), 1.0)
        self.assertAlmostEqual(float(metrics["coordinate_oob"]), 2 / 3)


if __name__ == "__main__":
    unittest.main()
