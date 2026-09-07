import unittest

import jax
import jax.numpy as jnp
import numpy as np

from utils.native_meanflow import (
    endpoint_from_average_velocity,
    meanflow_jvp_target,
    reverse_transport_step,
    sample_time_pairs,
)


class NativeMeanFlowMathTest(unittest.TestCase):
    def test_time_pairs_are_ordered(self):
        t, r, _ = sample_time_pairs(
            jax.random.PRNGKey(0),
            batch_size=1024,
            flow_ratio=0.25,
            distribution="uniform",
        )
        self.assertEqual(t.shape, (1024, 1))
        self.assertEqual(r.shape, (1024, 1))
        self.assertTrue(np.all(np.asarray(r) <= np.asarray(t)))
        self.assertTrue(np.all(np.asarray(r) >= 0.0))
        self.assertTrue(np.all(np.asarray(t) <= 1.0))

    def test_flow_ratio_one_forces_instantaneous_samples(self):
        t, r, mask = sample_time_pairs(
            jax.random.PRNGKey(1), batch_size=32, flow_ratio=1.0
        )
        np.testing.assert_allclose(r, t)
        self.assertTrue(np.all(np.asarray(mask)))

    def test_invalid_time_sampling_arguments_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "flow_ratio"):
            sample_time_pairs(
                jax.random.PRNGKey(2), batch_size=4, flow_ratio=1.1
            )
        with self.assertRaisesRegex(ValueError, "batch_size"):
            sample_time_pairs(jax.random.PRNGKey(2), batch_size=0)
        with self.assertRaisesRegex(ValueError, "logit_std"):
            sample_time_pairs(
                jax.random.PRNGKey(2), batch_size=4, logit_std=0.0
            )
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            sample_time_pairs(
                jax.random.PRNGKey(2),
                batch_size=4,
                distribution="invalid",
            )

    def test_affine_jvp_matches_closed_form(self):
        x_t = jnp.array([[0.2, -0.4], [0.7, 0.1]])
        velocity = jnp.array([[1.0, -2.0], [-0.5, 0.3]])
        r = jnp.array([[0.1], [0.2]])
        t = jnp.array([[0.8], [0.9]])
        matrix = jnp.array([[2.0, 0.3], [-0.4, 1.5]])
        time_coefficient = jnp.array([0.7, -0.2])
        r_coefficient = jnp.array([0.1, 0.5])

        def field(x, target_time, current_time):
            return (
                x @ matrix.T
                + current_time * time_coefficient
                + target_time * r_coefficient
            )

        u_pred, u_target, du_dt = meanflow_jvp_target(
            field, x_t, r, t, velocity
        )
        expected_u = field(x_t, r, t)
        expected_derivative = velocity @ matrix.T + time_coefficient
        expected_target = velocity - (t - r) * expected_derivative

        np.testing.assert_allclose(u_pred, expected_u, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            du_dt, expected_derivative, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            u_target, expected_target, rtol=1e-6, atol=1e-6
        )

    def test_instantaneous_target_is_conditional_velocity(self):
        x_t = jnp.ones((4, 2))
        velocity = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
        t = jnp.full((4, 1), 0.6)

        def nonlinear_field(x, target_time, current_time):
            return x**2 + current_time + target_time

        _, target, _ = meanflow_jvp_target(
            nonlinear_field, x_t, t, t, velocity
        )
        np.testing.assert_allclose(target, velocity)

    def test_jvp_target_is_stop_gradient(self):
        x_t = jnp.array([[0.3, -0.2]])
        velocity = jnp.array([[0.5, 0.7]])
        r = jnp.array([[0.1]])
        t = jnp.array([[0.8]])

        def target_sum(scale):
            def field(x, target_time, current_time):
                return scale * (x + target_time + current_time)

            _, target, _ = meanflow_jvp_target(
                field, x_t, r, t, velocity
            )
            return target.sum()

        gradient = jax.grad(target_sum)(jnp.array(2.0))
        np.testing.assert_allclose(gradient, 0.0)

    def test_one_step_transport_matches_endpoint_map(self):
        noise = jnp.array([[0.4, -0.9], [0.1, 0.7]])
        velocity = jnp.array([[0.2, -0.3], [-0.5, 0.4]])
        r = jnp.zeros((2, 1))
        t = jnp.ones((2, 1))
        transported = reverse_transport_step(noise, velocity, r, t)
        expected = endpoint_from_average_velocity(noise, velocity)
        np.testing.assert_allclose(transported, expected)

    def test_constant_field_has_identical_k1_and_k4_endpoint(self):
        noise = jnp.array([[1.0, -0.5]])
        velocity = jnp.array([[0.25, -0.75]])
        k1 = endpoint_from_average_velocity(noise, velocity)

        state = noise
        intervals = (
            (1.0, 0.75),
            (0.75, 0.5),
            (0.5, 0.25),
            (0.25, 0.0),
        )
        for current, target in intervals:
            state = reverse_transport_step(
                state,
                velocity,
                jnp.full((1, 1), target),
                jnp.full((1, 1), current),
            )
        np.testing.assert_allclose(state, k1, rtol=1e-6, atol=1e-6)

    def test_direct_q_gradient_improves_quadratic_q(self):
        noise = jnp.array([[0.5, -0.25]])
        target_action = jnp.array([[0.1, 0.4]])

        def q_loss(average_velocity):
            action = endpoint_from_average_velocity(noise, average_velocity)
            q_value = -jnp.square(action - target_action).sum()
            return -q_value

        average_velocity = jnp.zeros_like(noise)
        before = q_loss(average_velocity)
        gradient = jax.grad(q_loss)(average_velocity)
        after = q_loss(average_velocity - 0.1 * gradient)
        self.assertLess(float(after), float(before))


if __name__ == "__main__":
    unittest.main()
