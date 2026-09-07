"""AM guidance integrated into MeanFlowQL's reformulated target.

The original repository's :class:`MeanFlowQL_Agent` predicts the direct map
``g(o, x_t, t)`` rather than a native interval-average velocity.  This agent
keeps that network, sampler, critic, and adaptive MFI weighting, but replaces
the behavior path velocity inside MeanFlowQL's regression target by an
endpoint-Q-guided AM velocity.
"""

import copy
from typing import ClassVar, Mapping

from flax.core import FrozenDict
import jax
import jax.numpy as jnp
import ml_collections

from agents.meanflowql import MeanFlowQL_Agent, get_config as get_meanflowql_config
from utils.am_meanflow import (
    endpoint_reward_adjoint,
    meanflowql_adjoint_guided_velocity,
    meanflowql_endpoint_map,
    meanflowql_reformulated_target,
)


class AMMeanFlowTargetChangedAgent(MeanFlowQL_Agent):
    """MeanFlowQL with AM embedded in its reformulated ``g`` target.

    For ``b=0`` the changed target is

    ``v_AM = v - eta * t * lambda``

    ``g_target = x_t + (t - 1) * v_AM - t * D_t^[v_AM] g``.

    ``eta=0`` exactly recovers the repository's MeanFlowQL target.  Q enters
    through the stopped adjoint by default; the inherited direct-Q actor term
    is exposed only as an explicit hybrid ablation.
    """

    AGENT_NAME: ClassVar[str] = "am_meanflow_target_changed"
    TARGET_VARIANT: ClassVar[str] = "meanflowql_reformulated_adjoint"

    @classmethod
    def _validate_am_config(cls, config: Mapping) -> None:
        variant = config.get("am_target_variant", cls.TARGET_VARIANT)
        if variant != cls.TARGET_VARIANT:
            raise ValueError(
                f"{cls.AGENT_NAME} fixes am_target_variant to "
                f"{cls.TARGET_VARIANT!r}, got {variant!r}."
            )
        if float(config.get("adjoint_eta", 0.1)) < 0.0:
            raise ValueError("adjoint_eta must be non-negative.")
        if float(config.get("meanflowql_direct_q_coef", 0.0)) < 0.0:
            raise ValueError(
                "meanflowql_direct_q_coef must be non-negative."
            )
        if float(config.get("bound_loss_weight", 1.0)) < 0.0:
            raise ValueError("bound_loss_weight must be non-negative.")

    def _aggregate_target_q(self, observations, actions):
        critic_actions = (
            jnp.clip(actions, -1.0, 1.0)
            if self.config["clip_actions_for_critic"]
            else actions
        )
        q_values = self.network.select("target_critic")(
            observations, actions=critic_actions
        )
        if self.config["q_agg"] == "min":
            return q_values.min(axis=0)
        if self.config["q_agg"] == "max":
            return q_values.max(axis=0)
        if self.config["q_agg"] == "mean":
            return q_values.mean(axis=0)
        raise ValueError(
            f"Unsupported q_agg {self.config['q_agg']!r}; expected "
            "'min', 'max', or 'mean'."
        )

    def _meanflowql_endpoint_and_adjoint(
        self, observations, state, current_time
    ):
        """Differentiate target-Q through MeanFlowQL's implied endpoint."""

        def endpoint_fn(current_state):
            # No grad_params: this is the frozen current actor snapshot held by
            # self.network while apply_loss_fn optimizes a candidate tree.
            direct_map = self.network.select("actor_bc_flow")(
                observations, current_state, current_time
            )
            return meanflowql_endpoint_map(
                current_state, current_time, direct_map
            )

        def reward_fn(actions):
            return self._aggregate_target_q(observations, actions)

        endpoint, adjoint = endpoint_reward_adjoint(
            endpoint_fn, reward_fn, state
        )
        return (
            jax.lax.stop_gradient(endpoint),
            jax.lax.stop_gradient(adjoint),
        )

    def _sample_meanflowql_time(self, rng, batch_size):
        time_steps = self.config.get("time_steps", 10)
        if time_steps <= 1000:
            time_values = jnp.linspace(1 / time_steps, 1.0, time_steps)
            indices = jax.random.randint(rng, (batch_size,), 0, time_steps)
            return time_values[indices].reshape(-1, 1)
        return jax.random.uniform(rng, (batch_size, 1))

    def _meanflowql_loss(self, batch, grad_params, rng, use_am):
        """Evaluate baseline or AM-guided reformulated MeanFlowQL MFI."""

        batch_size = batch["actions"].shape[0]
        consistency_rng, time_rng, _, noise_rng = jax.random.split(rng, 4)
        current_time = self._sample_meanflowql_time(time_rng, batch_size)
        target_time = jnp.zeros_like(current_time)

        actions = batch["actions"]
        noise = self.sample_noise(noise_rng, actions.shape)
        state = (1.0 - current_time) * actions + current_time * noise
        conditional_velocity = noise - actions

        if use_am:
            endpoint, adjoint = self._meanflowql_endpoint_and_adjoint(
                batch["observations"], state, current_time
            )
        else:
            endpoint = actions
            adjoint = jnp.zeros_like(conditional_velocity)

        guided_velocity = meanflowql_adjoint_guided_velocity(
            conditional_velocity,
            adjoint,
            self.config["adjoint_eta"] if use_am else 0.0,
            current_time - target_time,
        )

        actor = self.network.select("actor_bc_flow")

        def direct_map(current_state, time):
            return actor(
                batch["observations"],
                current_state,
                time,
                params=grad_params,
            )

        prediction, total_derivative = jax.jvp(
            direct_map,
            (state, current_time),
            (guided_velocity, jnp.ones_like(current_time)),
        )
        target = meanflowql_reformulated_target(
            state,
            target_time,
            current_time,
            guided_velocity,
            total_derivative,
        )
        target = jnp.clip(jax.lax.stop_gradient(target), -5.0, 5.0)

        # Compute the exact unguided MeanFlowQL target with the same sample and
        # actor.  This is a diagnostic only and is stopped before logging.
        _, baseline_derivative = jax.jvp(
            direct_map,
            (state, current_time),
            (conditional_velocity, jnp.ones_like(current_time)),
        )
        baseline_target = meanflowql_reformulated_target(
            state,
            target_time,
            current_time,
            conditional_velocity,
            baseline_derivative,
        )
        baseline_target = jnp.clip(
            jax.lax.stop_gradient(baseline_target), -5.0, 5.0
        )

        error = prediction - target
        meanflow_loss = self.adaptive_l2_loss(
            error, current_time, mode="normal"
        )
        consistency_loss = self.consistency_loss(
            batch, grad_params, consistency_rng
        )
        flow_loss = (
            meanflow_loss
            + self.config.get("consistency_alpha", 0.0)
            * consistency_loss
        )
        correction = guided_velocity - conditional_velocity
        target_q = (
            self._aggregate_target_q(batch["observations"], endpoint).mean()
            if use_am
            else jnp.asarray(0.0)
        )

        return flow_loss, {
            "mean_flow_loss": meanflow_loss,
            "consistency_loss": consistency_loss,
            "flow_loss": flow_loss,
            "am_enabled": jnp.asarray(float(use_am)),
            "adjoint_norm": jnp.linalg.norm(adjoint, axis=-1).mean(),
            "correction_norm": jnp.linalg.norm(
                correction, axis=-1
            ).mean(),
            "conditional_velocity_norm": jnp.linalg.norm(
                conditional_velocity, axis=-1
            ).mean(),
            "guided_velocity_norm": jnp.linalg.norm(
                guided_velocity, axis=-1
            ).mean(),
            "derivative_norm": jnp.linalg.norm(
                total_derivative, axis=-1
            ).mean(),
            "target_norm": jnp.linalg.norm(target, axis=-1).mean(),
            "baseline_target_norm": jnp.linalg.norm(
                baseline_target, axis=-1
            ).mean(),
            "target_shift_norm": jnp.linalg.norm(
                target - baseline_target, axis=-1
            ).mean(),
            "remaining_time": current_time.mean(),
            "endpoint_q": target_q,
            "endpoint_out_of_bounds_fraction": (
                jnp.abs(endpoint) > 1.0
            ).mean(),
        }

    def meanflow_loss(self, batch, grad_params, rng):
        """Train with AM inside the reformulated MeanFlowQL target."""

        return self._meanflowql_loss(
            batch, grad_params, rng, use_am=True
        )

    def behavior_meanflow_loss(self, batch, grad_params, rng):
        """Use the exact original MeanFlowQL target during pretraining."""

        return self._meanflowql_loss(
            batch, grad_params, rng, use_am=False
        )

    def actor_loss(self, batch, grad_params, rng):
        """Keep bounds; expose direct-Q only as an explicit hybrid."""

        batch_size, action_dim = batch["actions"].shape
        _, noise_rng = jax.random.split(rng)
        time = jnp.ones((batch_size, 1))
        noise = self.sample_noise(noise_rng, (batch_size, action_dim))
        raw_actions = self.network.select("actor_bc_flow")(
            batch["observations"], noise, time, params=grad_params
        )
        bound_loss = (
            jax.nn.relu(raw_actions - 1.0).mean()
            + jax.nn.relu(-1.0 - raw_actions).mean()
        )
        actions = jnp.clip(raw_actions, -1.0, 1.0)

        direct_q_coef = self.config["meanflowql_direct_q_coef"]
        if direct_q_coef > 0.0:
            q_values = self.network.select("critic")(
                batch["observations"], actions=actions
            )
            q = q_values.mean(axis=0)
            q_loss = -q.mean()
            if self.config["normalize_q_loss"]:
                scale = jax.lax.stop_gradient(
                    1.0 / jnp.maximum(jnp.abs(q).mean(), 1e-6)
                )
                q_loss = scale * q_loss
        else:
            q = jnp.zeros((batch_size,))
            q_loss = jnp.asarray(0.0)

        actor_loss = (
            direct_q_coef * q_loss
            + self.config["bound_loss_weight"] * bound_loss
        )
        return actor_loss, {
            "actor_loss": actor_loss,
            "q_loss": q_loss,
            "q": q.mean(),
            "bound_loss": bound_loss,
            "mse": jnp.mean(jnp.square(actions - batch["actions"])),
        }

    @jax.jit
    def pretrain(self, batch, current_step=None):
        """Pretrain the unmodified MeanFlowQL target without critic guidance."""

        new_rng, loss_rng = jax.random.split(self.rng)

        def pretrain_loss(grad_params):
            return self.behavior_meanflow_loss(
                batch, grad_params, rng=loss_rng
            )

        network, info = self.network.apply_loss_fn(loss_fn=pretrain_loss)
        if current_step is not None:
            actor_schedule = self.config.get("actor_lr_schedule")
            critic_schedule = self.config.get("critic_lr_schedule")
            if actor_schedule is not None:
                info["metrics/actor_learning_rate"] = actor_schedule(
                    current_step
                )
            if critic_schedule is not None:
                info["metrics/critic_learning_rate"] = critic_schedule(
                    current_step
                )
        return self.replace(network=network, rng=new_rng), info

    @classmethod
    def from_meanflowql_agent(cls, meanflowql_agent, config=None):
        """Switch a restored MeanFlowQL state to the AM target in place."""

        merged = dict(meanflowql_agent.config)
        changed_defaults = get_config().to_dict()
        for key in (
            "am_target_variant",
            "adjoint_eta",
            "clip_actions_for_critic",
            "meanflowql_direct_q_coef",
        ):
            merged[key] = changed_defaults[key]
        if config is not None:
            if hasattr(config, "to_dict"):
                config = config.to_dict()
            merged.update(dict(config))
        merged["agent_name"] = cls.AGENT_NAME
        merged["am_target_variant"] = cls.TARGET_VARIANT
        cls._validate_am_config(merged)
        return cls(
            rng=meanflowql_agent.rng,
            network=meanflowql_agent.network,
            config=FrozenDict(merged),
            current_alpha=meanflowql_agent.current_alpha,
            loss_history=meanflowql_agent.loss_history,
            valid_count=meanflowql_agent.valid_count,
        )

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        local_config = copy.deepcopy(config)
        local_config["agent_name"] = cls.AGENT_NAME
        cls._validate_am_config(local_config)
        return super().create(
            seed, ex_observations, ex_actions, local_config
        )


def get_config():
    """Return MeanFlowQL settings plus AM target controls."""

    config = get_meanflowql_config()
    config.agent_name = "am_meanflow_target_changed"
    config.am_target_variant = "meanflowql_reformulated_adjoint"
    config.adjoint_eta = 0.1
    config.clip_actions_for_critic = True

    # Q enters through the AM target.  A nonzero value deliberately recreates
    # a hybrid with MeanFlowQL's original direct policy-gradient term.
    config.meanflowql_direct_q_coef = 0.0
    return ml_collections.ConfigDict(config)
