import unittest

import jax
import jax.numpy as jnp
import numpy as np

from utils.am_meanflow import (
    adjoint_corrected_velocity,
    alpha_flow_schedule,
    compose_am_target,
    critical_alpha,
    endpoint_map,
    endpoint_reward_adjoint,
    generative_velocity_from_native,
    meanflow_endpoint_jvp_target,
    native_interval_from_generative,
    positive_alpha_regression_loss,
)


class AMMeanFlowMathTest(unittest.TestCase):
    def test_native_to_generative_time_mapping_and_sign(self):
        start = jnp.asarray([[0.2]])
        end = jnp.asarray([[0.7]])
        native_r, native_t = native_interval_from_generative(start, end)

        np.testing.assert_allclose(native_r, [[0.3]])
        np.testing.assert_allclose(native_t, [[0.8]])
        np.testing.assert_allclose(
            generative_velocity_from_native(jnp.asarray([[2.0, -3.0]])),
            [[-2.0, 3.0]],
        )

    def test_endpoint_map_uses_remaining_interval(self):
        endpoint = endpoint_map(
            jnp.asarray([[1.0, 2.0]]),
            jnp.asarray([[0.25]]),
            jnp.asarray([[4.0, -2.0]]),
        )
        np.testing.assert_allclose(endpoint, [[4.0, 0.5]])

    def test_alpha_schedule_starts_at_one_and_reaches_floor(self):
        start = alpha_flow_schedule(20, 20, 10, 100, 0.05, 8.0)
        warmup = alpha_flow_schedule(29, 20, 10, 100, 0.05, 8.0)
        middle = alpha_flow_schedule(80, 20, 10, 100, 0.05, 8.0)
        final = alpha_flow_schedule(130, 20, 10, 100, 0.05, 8.0)

        self.assertAlmostEqual(float(start), 1.0, places=6)
        self.assertAlmostEqual(float(warmup), 1.0, places=6)
        self.assertGreater(float(middle), 0.05)
        self.assertLess(float(middle), 1.0)
        self.assertAlmostEqual(float(final), 0.05, places=6)

    def test_invalid_alpha_schedule_arguments_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "start_step"):
            alpha_flow_schedule(0, -1, 0, 10, 0.1, 1.0)
        with self.assertRaisesRegex(ValueError, "warmup_steps"):
            alpha_flow_schedule(0, 0, -1, 10, 0.1, 1.0)
        with self.assertRaisesRegex(ValueError, "transition_steps"):
            alpha_flow_schedule(0, 0, 0, 0, 0.1, 1.0)
        with self.assertRaisesRegex(ValueError, "floor"):
            alpha_flow_schedule(0, 0, 0, 10, 1.1, 1.0)
        with self.assertRaisesRegex(ValueError, "gamma"):
            alpha_flow_schedule(0, 0, 0, 10, 0.1, 0.0)

    def test_adjoint_matches_closed_form_and_finite_difference(self):
        # F(x)=2x and R(a)=a^2/2 at x=1: J_F^T grad R = 2*2=4.
        def endpoint_fn(state):
            return 2.0 * state

        def reward_fn(action):
            return 0.5 * jnp.square(action)

        state = jnp.asarray([1.0])
        endpoint, adjoint = endpoint_reward_adjoint(
            endpoint_fn, reward_fn, state
        )

        epsilon = 1e-3
        def composed(x):
            return float(
                jnp.sum(reward_fn(endpoint_fn(jnp.asarray([x]))))
            )

        finite_difference = (
            composed(1.0 + epsilon) - composed(1.0 - epsilon)
        ) / (2.0 * epsilon)
        np.testing.assert_allclose(endpoint, [2.0])
        np.testing.assert_allclose(adjoint, [4.0])
        np.testing.assert_allclose(
            float(adjoint[0]), finite_difference, rtol=1e-3
        )

    def test_interval_scaled_adjoint_correction_is_exact(self):
        corrected = adjoint_corrected_velocity(
            jnp.asarray([[1.0, -2.0]]),
            jnp.asarray([[2.0, 3.0]]),
            eta=0.5,
            interval=jnp.asarray([[0.2]]),
        )
        np.testing.assert_allclose(corrected, [[1.2, -1.7]])

    def test_zero_alpha_endpoint_jvp_target_is_exact(self):
        target = meanflow_endpoint_jvp_target(
            local_velocity=jnp.asarray([[2.0, -1.0]]),
            start_time=jnp.asarray([[0.25]]),
            total_derivative=jnp.asarray([[4.0, 2.0]]),
        )
        np.testing.assert_allclose(target, [[5.0, 0.5]])

    def test_target_switch_has_exact_alpha_endpoints(self):
        reward = jnp.asarray([[2.0, 4.0]])
        bootstrap = jnp.asarray([[10.0, 20.0]])
        np.testing.assert_allclose(
            compose_am_target(reward, bootstrap, 1.0), reward
        )
        np.testing.assert_allclose(
            compose_am_target(reward, bootstrap, 0.0), bootstrap
        )
        np.testing.assert_allclose(
            compose_am_target(reward, bootstrap, 0.25), [[8.0, 16.0]]
        )

    def test_normalized_and_unnormalized_loss_are_explicit(self):
        prediction = jnp.asarray([[0.0, 0.0]])
        reward = jnp.asarray([[2.0, 2.0]])
        bootstrap = jnp.asarray([[0.0, 0.0]])
        normalized, target = positive_alpha_regression_loss(
            prediction,
            reward,
            bootstrap,
            alpha=0.25,
            normalize_by_alpha=True,
        )
        unnormalized, _ = positive_alpha_regression_loss(
            prediction,
            reward,
            bootstrap,
            alpha=0.25,
            normalize_by_alpha=False,
        )

        np.testing.assert_allclose(target, [[0.5, 0.5]])
        self.assertAlmostEqual(float(unnormalized), 0.25, places=6)
        self.assertAlmostEqual(float(normalized), 1.0, places=6)

    def test_am_target_is_stopped(self):
        prediction = jnp.asarray([[0.0]])
        bootstrap = jnp.asarray([[1.0]])

        def loss_from_reward(reward):
            return positive_alpha_regression_loss(
                prediction,
                reward,
                bootstrap,
                alpha=0.5,
            )[0]

        reward_gradient = jax.grad(loss_from_reward)(jnp.asarray([[2.0]]))
        np.testing.assert_allclose(reward_gradient, 0.0)

    def test_critical_alpha_is_bounded_and_has_fallback(self):
        reward_error = jnp.asarray([[1.0, -2.0], [0.5, 1.0]])
        bootstrap_error = jnp.asarray([[0.0, 3.0], [1.0, -1.0]])
        alpha, _, _, _ = critical_alpha(reward_error, bootstrap_error)
        self.assertGreaterEqual(float(alpha), 0.0)
        self.assertLessEqual(float(alpha), 1.0)

        fallback, _, _, _ = critical_alpha(
            reward_error, reward_error, fallback=0.7
        )
        self.assertAlmostEqual(float(fallback), 0.7, places=6)


if __name__ == "__main__":
    unittest.main()
