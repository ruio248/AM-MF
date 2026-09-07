"""AM-MF refinement stage built on the native MeanFlow parameterization.

This agent intentionally keeps :class:`NativeMeanFlowAgent` unchanged.  The
native JVP identity is used for pretraining; the AM stage replaces the actor's
main target with a frozen-reference/adjoint target mixed with an EMA bootstrap
target.  Direct-Q is not applied to the student endpoint in the strict AM
objective: Q enters through the stopped adjoint target instead.
"""

import copy
from typing import Any, ClassVar, Mapping

from flax.core import FrozenDict
import jax
import jax.numpy as jnp
import ml_collections

from agents.native_meanflow import (
    NativeMeanFlowAgent,
    get_config as get_native_config,
)
from utils.am_meanflow import (
    adjoint_corrected_velocity,
    alpha_flow_schedule,
    critical_alpha,
    endpoint_map,
    endpoint_reward_adjoint,
    generative_velocity_from_native,
    meanflow_endpoint_jvp_target,
    native_interval_from_generative,
    positive_alpha_regression_loss,
)
from utils.native_meanflow import sample_time_pairs


class AMMeanFlowAgent(NativeMeanFlowAgent):
    """AM-MF with the original Note target ``eta * (t-s) * lambda``.

    ``pre_actor_params`` is the frozen behavior/reference actor.  The actor and
    target actor still live in the inherited ``ModuleDict`` so existing native
    sampling, critic training, and optimizer state remain reusable.
    """

    AGENT_NAME: ClassVar[str] = "am_meanflow"
    TARGET_VARIANT: ClassVar[str] = "note_average"

    pre_actor_params: Any = None
    am_updates: Any = 0

    @classmethod
    def _validate_am_config(cls, config: Mapping[str, Any]) -> None:
        target_variant = config.get(
            "am_target_variant", cls.TARGET_VARIANT
        )
        if target_variant != cls.TARGET_VARIANT:
            raise ValueError(
                f"{cls.AGENT_NAME} fixes am_target_variant to "
                f"{cls.TARGET_VARIANT!r}, got {target_variant!r}."
            )

        alpha_mode = config.get("alpha_mode", "anneal")
        if alpha_mode not in ("fixed", "anneal"):
            raise ValueError(
                "alpha_mode must be 'fixed' or 'anneal', got "
                f"{alpha_mode!r}."
            )
        alpha_value = float(config.get("alpha_value", 1.0))
        if not 0.0 <= alpha_value <= 1.0:
            raise ValueError(
                f"alpha_value must be in [0, 1], got {alpha_value}."
            )
        alpha_eps = float(config.get("alpha_eps", 1e-4))
        if alpha_eps <= 0.0:
            raise ValueError(
                f"alpha_eps must be positive, got {alpha_eps}."
            )
        if float(config.get("adjoint_eta", 0.1)) < 0.0:
            raise ValueError("adjoint_eta must be non-negative.")
        for key in (
            "am_loss_coef",
            "critic_coef",
            "native_regularizer_coef",
        ):
            if float(config.get(key, 0.0)) < 0.0:
                raise ValueError(f"{key} must be non-negative.")

        if alpha_mode == "anneal":
            alpha_floor = float(config.get("alpha_floor", 0.05))
            if alpha_floor <= alpha_eps:
                raise ValueError(
                    "annealed AM training requires alpha_floor > alpha_eps; "
                    "use alpha_mode='fixed', alpha_value=0 for the explicit "
                    "JVP target experiment."
                )
            # Run validation once outside JIT.  The returned value is ignored.
            alpha_flow_schedule(
                0,
                int(config.get("alpha_start_step", 0)),
                int(config.get("alpha_warmup_steps", 50000)),
                int(config.get("alpha_transition_steps", 400000)),
                alpha_floor,
                float(config.get("alpha_gamma", 8.0)),
            )

    def _alpha(self):
        if self.config["alpha_mode"] == "fixed":
            return jnp.asarray(
                self.config["alpha_value"], dtype=jnp.float32
            )
        return alpha_flow_schedule(
            self.am_updates,
            self.config["alpha_start_step"],
            self.config["alpha_warmup_steps"],
            self.config["alpha_transition_steps"],
            self.config["alpha_floor"],
            self.config["alpha_gamma"],
        )

    def _sample_am_times(self, rng, batch_size):
        native_t, native_r, _ = sample_time_pairs(
            rng,
            batch_size=batch_size,
            flow_ratio=0.0,
            distribution=self.config["time_distribution"],
            logit_mean=self.config["time_logit_mean"],
            logit_std=self.config["time_logit_std"],
        )
        # The same ordered scalars are interpreted in generative time here.
        return native_r, native_t

    def _pre_velocity(self, observations, state, native_r, native_t):
        params = dict(self.network.params)
        params["modules_actor_bc_flow"] = self.pre_actor_params
        return self._velocity(
            observations,
            state,
            native_r,
            native_t,
            params=params,
        )

    def _generative_velocity(
        self,
        observations,
        state,
        start_time,
        end_time,
        params=None,
        role="online",
    ):
        """Call a native actor through the noise-to-action AM convention."""

        native_r, native_t = native_interval_from_generative(
            start_time, end_time
        )
        if role == "online":
            native_velocity = self._velocity(
                observations,
                state,
                native_r,
                native_t,
                params=params,
            )
        elif role == "target":
            if params is not None:
                raise ValueError("Target velocity does not accept grad params.")
            native_velocity = self._velocity(
                observations,
                state,
                native_r,
                native_t,
                target=True,
            )
        elif role == "pre":
            if params is not None:
                raise ValueError("Frozen pre velocity does not accept params.")
            native_velocity = self._pre_velocity(
                observations, state, native_r, native_t
            )
        else:
            raise ValueError(
                f"Unknown actor role {role!r}; expected online, target, or pre."
            )
        return generative_velocity_from_native(native_velocity)

    def _target_q(self, observations, actions):
        critic_actions = (
            jnp.clip(actions, -1.0, 1.0)
            if self.config["clip_actions_for_critic"]
            else actions
        )
        q_values = self.network.select("target_critic")(
            observations, actions=critic_actions
        )
        return self._aggregate_q(q_values)

    def _endpoint_and_adjoint(
        self, observations, state, start_time
    ):
        one = jnp.ones_like(start_time)

        def endpoint_fn(current_state):
            endpoint_velocity = self._generative_velocity(
                observations,
                current_state,
                start_time,
                one,
                role="target",
            )
            return endpoint_map(
                current_state, start_time, endpoint_velocity
            )

        def reward_fn(actions):
            return self._target_q(observations, actions)

        endpoint, adjoint = endpoint_reward_adjoint(
            endpoint_fn, reward_fn, state
        )
        return (
            jax.lax.stop_gradient(endpoint),
            jax.lax.stop_gradient(adjoint),
        )

    def _correct_reward_velocity(
        self, base_velocity, adjoint, interval
    ):
        """Original Note target: ``u_pre + eta * interval * lambda``."""

        return adjoint_corrected_velocity(
            base_velocity,
            adjoint,
            self.config["adjoint_eta"],
            interval,
        )

    def _positive_alpha_loss(
        self, batch, grad_params, rng, alpha
    ):
        observations = batch["observations"]
        batch_size, action_dim = batch["actions"].shape
        noise_rng, time_rng = jax.random.split(rng)
        x_0 = self.sample_noise(noise_rng, (batch_size, action_dim))
        r, t = self._sample_am_times(time_rng, batch_size)
        s = alpha * r + (1.0 - alpha) * t
        zero = jnp.zeros_like(r)

        # The current policy chooses the states at which the AM target is
        # evaluated, but trajectory construction itself is detached.
        x_r = x_0 + r * self._generative_velocity(
            observations, x_0, zero, r, role="online"
        )
        x_s = x_0 + s * self._generative_velocity(
            observations, x_0, zero, s, role="online"
        )
        x_r = jax.lax.stop_gradient(x_r)
        x_s = jax.lax.stop_gradient(x_s)

        rollout_interval = t - s
        x_t = x_s + rollout_interval * self._generative_velocity(
            observations, x_s, s, t, role="target"
        )
        x_t = jax.lax.stop_gradient(x_t)
        endpoint, adjoint = self._endpoint_and_adjoint(
            observations, x_t, t
        )

        base_velocity = self._generative_velocity(
            observations, x_s, s, t, role="pre"
        )
        reward_velocity = self._correct_reward_velocity(
            base_velocity, adjoint, rollout_interval
        )
        bootstrap_velocity = self._generative_velocity(
            observations, x_r, r, s, role="target"
        )
        prediction = self._generative_velocity(
            observations,
            x_r,
            r,
            t,
            params=grad_params,
            role="online",
        )

        loss, target = positive_alpha_regression_loss(
            prediction,
            reward_velocity,
            bootstrap_velocity,
            alpha,
            alpha_eps=self.config["alpha_eps"],
            normalize_by_alpha=self.config[
                "normalize_am_loss_by_alpha"
            ],
        )
        reward_error = prediction - jax.lax.stop_gradient(reward_velocity)
        bootstrap_error = prediction - jax.lax.stop_gradient(
            bootstrap_velocity
        )
        alpha_c, tfm_a, tfm_b, tc_c = critical_alpha(
            reward_error, bootstrap_error
        )
        correction = reward_velocity - base_velocity

        return loss, {
            "am_loss": loss,
            "alpha": alpha,
            "critical_alpha": alpha_c,
            "reward_target_loss": jnp.mean(jnp.square(reward_error)),
            "bootstrap_loss": jnp.mean(jnp.square(bootstrap_error)),
            "mixed_target_loss": jnp.mean(jnp.square(prediction - target)),
            "jvp_loss": jnp.asarray(0.0),
            "jvp_branch": jnp.asarray(0.0),
            "adjoint_norm": jnp.linalg.norm(adjoint, axis=-1).mean(),
            "correction_norm": jnp.linalg.norm(
                correction, axis=-1
            ).mean(),
            "prediction_norm": jnp.linalg.norm(
                prediction, axis=-1
            ).mean(),
            "target_norm": jnp.linalg.norm(target, axis=-1).mean(),
            "rollout_interval": rollout_interval.mean(),
            "sample_interval": (t - r).mean(),
            "endpoint_q": self._target_q(observations, endpoint).mean(),
            "endpoint_out_of_bounds_fraction": (
                jnp.abs(endpoint) > 1.0
            ).mean(),
            "tfm_a": tfm_a,
            "tfm_b": tfm_b,
            "tc_c": tc_c,
        }

    def _zero_alpha_loss(self, batch, grad_params, rng, alpha):
        """Use the explicit endpoint-MeanFlow JVP limit at ``alpha=0``."""

        observations = batch["observations"]
        batch_size, action_dim = batch["actions"].shape
        noise_rng, time_rng = jax.random.split(rng)
        x_0 = self.sample_noise(noise_rng, (batch_size, action_dim))
        eps = self.config["alpha_eps"]
        time = jax.random.uniform(
            time_rng,
            (batch_size, 1),
            minval=eps,
            maxval=1.0 - eps,
        )
        zero_time = jnp.zeros_like(time)
        state = x_0 + time * self._generative_velocity(
            observations, x_0, zero_time, time, role="online"
        )
        state = jax.lax.stop_gradient(state)

        endpoint, adjoint = self._endpoint_and_adjoint(
            observations, state, time
        )
        base_velocity = self._generative_velocity(
            observations, state, time, time, role="pre"
        )
        zero_interval = jnp.zeros_like(time)
        local_velocity = self._correct_reward_velocity(
            base_velocity, adjoint, zero_interval
        )
        one = jnp.ones_like(time)

        def target_field(current_state, current_time):
            return self._generative_velocity(
                observations,
                current_state,
                current_time,
                one,
                role="target",
            )

        _, total_derivative = jax.jvp(
            target_field,
            (state, time),
            (local_velocity, jnp.ones_like(time)),
        )
        target = jax.lax.stop_gradient(
            meanflow_endpoint_jvp_target(
                local_velocity, time, total_derivative
            )
        )
        prediction = self._generative_velocity(
            observations,
            state,
            time,
            one,
            params=grad_params,
            role="online",
        )
        error = prediction - target
        loss = jnp.mean(jnp.square(error))
        reward_error = prediction - jax.lax.stop_gradient(local_velocity)
        correction = local_velocity - base_velocity
        zeros = jnp.asarray(0.0)

        return loss, {
            "am_loss": loss,
            "alpha": alpha,
            "critical_alpha": zeros,
            "reward_target_loss": jnp.mean(jnp.square(reward_error)),
            "bootstrap_loss": zeros,
            "mixed_target_loss": loss,
            "jvp_loss": loss,
            "jvp_branch": jnp.asarray(1.0),
            "adjoint_norm": jnp.linalg.norm(adjoint, axis=-1).mean(),
            "correction_norm": jnp.linalg.norm(
                correction, axis=-1
            ).mean(),
            "prediction_norm": jnp.linalg.norm(
                prediction, axis=-1
            ).mean(),
            "target_norm": jnp.linalg.norm(target, axis=-1).mean(),
            "rollout_interval": zeros,
            "sample_interval": (1.0 - time).mean(),
            "endpoint_q": self._target_q(observations, endpoint).mean(),
            "endpoint_out_of_bounds_fraction": (
                jnp.abs(endpoint) > 1.0
            ).mean(),
            "tfm_a": zeros,
            "tfm_b": zeros,
            "tc_c": zeros,
        }

    def am_loss(self, batch, grad_params, rng, alpha):
        """Use the AM target for positive alpha and explicit JVP at zero."""

        # This branch is intentionally static.  Differentiating a lax.cond
        # containing a nested JVP can contaminate the inactive positive-alpha
        # gradient on older JAX versions.  Annealed runs are validated to stay
        # above alpha_eps; the exact zero/JVP experiment uses fixed alpha=0.
        if (
            self.config["alpha_mode"] == "fixed"
            and self.config["alpha_value"] <= self.config["alpha_eps"]
        ):
            return self._zero_alpha_loss(batch, grad_params, rng, alpha)
        return self._positive_alpha_loss(batch, grad_params, rng, alpha)

    def total_loss(self, batch, grad_params, rng, current_step=0):
        """Train the critic and the AM actor with explicitly separated terms."""

        del current_step
        am_rng, critic_rng, native_rng = jax.random.split(rng, 3)
        alpha = self._alpha()
        am_loss, am_info = self.am_loss(batch, grad_params, am_rng, alpha)

        if self.config["critic_coef"] > 0.0:
            critic_loss, critic_info = self.critic_loss(
                batch, grad_params, critic_rng
            )
        else:
            critic_loss = jnp.asarray(0.0)
            critic_info = {
                "critic_loss": critic_loss,
                "q_mean": critic_loss,
                "q_max": critic_loss,
                "q_min": critic_loss,
                "target_q_mean": critic_loss,
            }

        if self.config["native_regularizer_coef"] > 0.0:
            native_loss, native_info = self.meanflow_loss(
                batch, grad_params, native_rng
            )
        else:
            native_loss = jnp.asarray(0.0)
            native_info = {
                "mean_flow_loss": native_loss,
                "identity_residual": native_loss,
                "instantaneous_fraction": native_loss,
                "mean_interval": native_loss,
                "velocity_norm": native_loss,
                "derivative_norm": native_loss,
            }

        total_loss = (
            self.config["am_loss_coef"] * am_loss
            + self.config["critic_coef"] * critic_loss
            + self.config["native_regularizer_coef"] * native_loss
        )
        info = {f"am/{key}": value for key, value in am_info.items()}
        info.update(
            {f"critic/{key}": value for key, value in critic_info.items()}
        )
        info.update(
            {f"native/{key}": value for key, value in native_info.items()}
        )
        info["am_loss_coef"] = self.config["am_loss_coef"]
        info["critic_coef"] = self.config["critic_coef"]
        info["native_regularizer_coef"] = self.config[
            "native_regularizer_coef"
        ]
        info["total_loss"] = total_loss
        return total_loss, info

    @jax.jit
    def update(self, batch, current_step=0):
        """Run one AM update, then update actor and critic EMA targets."""

        new_rng, loss_rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(
                batch, grad_params, loss_rng, current_step=current_step
            )

        network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        network = self._update_ema_module(
            network, "modules_critic", "modules_target_critic"
        )
        network = self._update_ema_module(
            network,
            "modules_actor_bc_flow",
            "modules_target_actor_bc_flow",
        )
        alpha = self._alpha()
        return self.replace(
            network=network,
            rng=new_rng,
            current_alpha=alpha,
            am_updates=self.am_updates + 1,
        ), info

    @jax.jit
    def pretrain(self, batch, current_step=None):
        """Pretrain native MeanFlow and hard-sync all AM actor roles."""

        del current_step
        new_rng, loss_rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.meanflow_loss(batch, grad_params, loss_rng)

        network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        params = dict(network.params)
        online_actor = params["modules_actor_bc_flow"]
        params["modules_target_actor_bc_flow"] = online_actor
        network = network.replace(params=params)
        return self.replace(
            network=network,
            rng=new_rng,
            pre_actor_params=online_actor,
            am_updates=jnp.asarray(0, dtype=jnp.int32),
            current_alpha=self._initial_alpha(),
        ), info

    def _initial_alpha(self):
        if self.config["alpha_mode"] == "fixed":
            return jnp.asarray(self.config["alpha_value"], dtype=jnp.float32)
        return jnp.asarray(1.0, dtype=jnp.float32)

    def initialize_am_stage(self):
        """Reset pre/target roles at an explicit Native -> AM boundary."""

        params = dict(self.network.params)
        online_actor = params["modules_actor_bc_flow"]
        online_critic = params["modules_critic"]
        params["modules_target_actor_bc_flow"] = copy.deepcopy(online_actor)
        params["modules_target_critic"] = copy.deepcopy(online_critic)
        return self.replace(
            network=self.network.replace(params=params),
            pre_actor_params=copy.deepcopy(online_actor),
            am_updates=jnp.asarray(0, dtype=jnp.int32),
            current_alpha=self._initial_alpha(),
        )

    @classmethod
    def from_native_agent(cls, native_agent, config=None):
        """Convert a loaded NativeMeanFlowAgent into a fresh AM stage.

        Network parameters and optimizer state are retained.  The online actor
        initializes both the frozen reference and EMA actor; the online critic
        initializes the target critic.  This makes the target-role transition
        explicit when importing a native checkpoint.
        """

        defaults = get_config().to_dict()
        defaults["agent_name"] = cls.AGENT_NAME
        defaults["am_target_variant"] = cls.TARGET_VARIANT
        defaults.update(dict(native_agent.config))
        if config is not None:
            if hasattr(config, "to_dict"):
                config = config.to_dict()
            defaults.update(dict(config))
        defaults["agent_name"] = cls.AGENT_NAME
        cls._validate_am_config(defaults)

        params = dict(native_agent.network.params)
        online_actor = params["modules_actor_bc_flow"]
        online_critic = params["modules_critic"]
        params["modules_target_actor_bc_flow"] = copy.deepcopy(online_actor)
        params["modules_target_critic"] = copy.deepcopy(online_critic)
        network = native_agent.network.replace(params=params)
        initial_alpha = (
            defaults["alpha_value"]
            if defaults["alpha_mode"] == "fixed"
            else 1.0
        )
        return cls(
            rng=native_agent.rng,
            network=network,
            config=FrozenDict(defaults),
            current_alpha=jnp.asarray(initial_alpha, dtype=jnp.float32),
            loss_history=native_agent.loss_history,
            valid_count=native_agent.valid_count,
            pre_actor_params=copy.deepcopy(online_actor),
            am_updates=jnp.asarray(0, dtype=jnp.int32),
        )

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        local_config = copy.deepcopy(config)
        local_config["agent_name"] = cls.AGENT_NAME
        cls._validate_am_config(local_config)
        agent = super().create(
            seed, ex_observations, ex_actions, local_config
        )
        online_actor = agent.network.params["modules_actor_bc_flow"]
        return agent.replace(
            pre_actor_params=copy.deepcopy(online_actor),
            am_updates=jnp.asarray(0, dtype=jnp.int32),
            current_alpha=agent._initial_alpha(),
        )


def get_config():
    """Return Native MeanFlow settings plus the strict AM target controls."""

    config = get_native_config()
    config.agent_name = "am_meanflow"

    # AM replaces the native JVP/direct-Q actor objective by default.  The
    # native identity can be reintroduced only as an explicitly named
    # regularizer/ablation.
    config.meanflow_coef = 0.0
    config.q_coef = 0.0
    config.bound_loss_weight = 0.0
    config.am_loss_coef = 1.0
    config.native_regularizer_coef = 0.0

    config.am_target_variant = "note_average"
    config.adjoint_eta = 0.1
    config.clip_actions_for_critic = True
    config.normalize_am_loss_by_alpha = True

    config.alpha_mode = "anneal"
    config.alpha_value = 1.0
    config.alpha_start_step = 0
    config.alpha_warmup_steps = 50000
    config.alpha_transition_steps = 400000
    config.alpha_floor = 0.05
    config.alpha_gamma = 8.0
    config.alpha_eps = 1e-4
    return ml_collections.ConfigDict(config)
