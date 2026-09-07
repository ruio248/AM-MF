"""AM guidance integrated into MeanFlowQL's reformulated target.

The original repository's :class:`MeanFlowQL_Agent` predicts the direct map
``g(o, x_t, t)`` rather than a native interval-average velocity.  This agent
keeps that network, sampler, critic, and adaptive MFI weighting, embeds the
endpoint-Q-guided AM velocity in the reformulated reward target, and then
applies the note's frozen-pre/EMA-target AlphaFlow mixture in velocity space.
"""

import copy
from typing import Any, ClassVar, Mapping

from flax.core import FrozenDict, unfreeze
import jax
import jax.numpy as jnp
import ml_collections

from agents.meanflowql import MeanFlowQL_Agent, get_config as get_meanflowql_config
from utils.am_meanflow import (
    alpha_flow_schedule,
    critical_alpha,
    endpoint_reward_adjoint,
    meanflowql_adjoint_guided_velocity,
    meanflowql_alphaflow_state,
    meanflowql_alphaflow_target,
    meanflowql_endpoint_map,
    meanflowql_reformulated_target,
)


class AMMeanFlowTargetChangedAgent(MeanFlowQL_Agent):
    """MeanFlowQL with AM embedded in its reformulated ``g`` target.

    For ``b=0`` the changed target is

    ``v_AM = v - eta * t * lambda``

    ``g_target = x_t + (t - 1) * v_AM - t * D_t^[v_AM] g``.

    At the ``alpha_AF=1`` boundary, ``eta=0`` exactly recovers the repository's
    MeanFlowQL target.  Q enters through the stopped adjoint by default; the
    inherited direct-Q actor term is exposed only as an explicit hybrid
    ablation.
    """

    AGENT_NAME: ClassVar[str] = "am_meanflow_target_changed"
    TARGET_VARIANT: ClassVar[str] = "meanflowql_reformulated_adjoint"

    # AlphaFlow actor roles are stored outside MeanFlowQL's optimizer tree.
    # This preserves exact compatibility with existing MeanFlowQL checkpoints
    # while still checkpointing the frozen behavior and EMA snapshots.
    pre_actor_params: Any = None
    target_actor_params: Any = None
    alphaflow_updates: Any = 0
    current_alphaflow_alpha: Any = 1.0

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

        alpha_mode = config.get("alphaflow_alpha_mode", "anneal")
        if alpha_mode not in ("fixed", "anneal"):
            raise ValueError(
                "alphaflow_alpha_mode must be 'fixed' or 'anneal', got "
                f"{alpha_mode!r}."
            )
        alpha_value = float(config.get("alphaflow_alpha_value", 1.0))
        if not 0.0 <= alpha_value <= 1.0:
            raise ValueError(
                "alphaflow_alpha_value must be in [0, 1], got "
                f"{alpha_value}."
            )
        alpha_eps = float(config.get("alphaflow_alpha_eps", 1e-4))
        if alpha_eps <= 0.0:
            raise ValueError("alphaflow_alpha_eps must be positive.")
        target_tau = float(config.get("alphaflow_target_tau", 0.005))
        if not 0.0 < target_tau <= 1.0:
            raise ValueError("alphaflow_target_tau must be in (0, 1].")
        if alpha_mode == "anneal":
            alpha_floor = float(
                config.get("alphaflow_alpha_floor", 0.05)
            )
            if alpha_floor <= alpha_eps:
                raise ValueError(
                    "annealed AlphaFlow requires alphaflow_alpha_floor > "
                    "alphaflow_alpha_eps; use fixed alpha=0 for the exact "
                    "JVP consistency limit."
                )
            alpha_flow_schedule(
                0,
                int(config.get("alphaflow_start_step", 0)),
                int(config.get("alphaflow_warmup_steps", 50000)),
                int(config.get("alphaflow_transition_steps", 400000)),
                alpha_floor,
                float(config.get("alphaflow_gamma", 8.0)),
            )

    @staticmethod
    def _actor_snapshot(params):
        """Extract every actor/actor-encoder subtree from MeanFlowQL params."""

        return {
            key: copy.deepcopy(unfreeze(value))
            for key, value in dict(params).items()
            if key.startswith("modules_actor_bc_flow")
        }

    def _call_actor_snapshot(
        self, snapshot, observations, state, current_time
    ):
        params = dict(self.network.params)
        params.update(snapshot)
        return self.network.select("actor_bc_flow")(
            observations, state, current_time, params=params
        )

    def _alphaflow_alpha(self):
        if self.config["alphaflow_alpha_mode"] == "fixed":
            return jnp.asarray(
                self.config["alphaflow_alpha_value"], dtype=jnp.float32
            )
        return alpha_flow_schedule(
            self.alphaflow_updates,
            self.config["alphaflow_start_step"],
            self.config["alphaflow_warmup_steps"],
            self.config["alphaflow_transition_steps"],
            self.config["alphaflow_alpha_floor"],
            self.config["alphaflow_gamma"],
        )

    def _initial_alphaflow_alpha(self):
        if self.config["alphaflow_alpha_mode"] == "fixed":
            return jnp.asarray(
                self.config["alphaflow_alpha_value"], dtype=jnp.float32
            )
        return jnp.asarray(1.0, dtype=jnp.float32)

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
            direct_map = self._call_actor_snapshot(
                self.target_actor_params,
                observations,
                current_state,
                current_time,
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

    def _sample_path(self, batch, time_rng, noise_rng):
        batch_size = batch["actions"].shape[0]
        current_time = self._sample_meanflowql_time(time_rng, batch_size)
        actions = batch["actions"]
        noise = self.sample_noise(noise_rng, actions.shape)
        state = (1.0 - current_time) * actions + current_time * noise
        return state, current_time, noise - actions

    def _baseline_target(
        self, observations, state, current_time, velocity, grad_params
    ):
        actor = self.network.select("actor_bc_flow")

        def direct_map(current_state, time):
            return actor(
                observations,
                current_state,
                time,
                params=grad_params,
            )

        prediction, derivative = jax.jvp(
            direct_map,
            (state, current_time),
            (velocity, jnp.ones_like(current_time)),
        )
        target = meanflowql_reformulated_target(
            state,
            jnp.zeros_like(current_time),
            current_time,
            velocity,
            derivative,
        )
        return prediction, jnp.clip(
            jax.lax.stop_gradient(target), -5.0, 5.0
        )

    def _positive_alphaflow_loss(
        self, batch, grad_params, rng, alpha
    ):
        """Wrap the AM changed target in the note's AlphaFlow mixture."""

        consistency_rng, time_rng, _, noise_rng = jax.random.split(rng, 4)
        state, current_time, conditional_velocity = self._sample_path(
            batch, time_rng, noise_rng
        )
        observations = batch["observations"]

        bootstrap_direct_map = self._call_actor_snapshot(
            self.target_actor_params,
            observations,
            state,
            current_time,
        )
        intermediate_time, intermediate_state, bootstrap_velocity = (
            meanflowql_alphaflow_state(
                state,
                current_time,
                alpha,
                bootstrap_direct_map,
            )
        )
        intermediate_state = jax.lax.stop_gradient(intermediate_state)

        endpoint, adjoint = self._meanflowql_endpoint_and_adjoint(
            observations, intermediate_state, intermediate_time
        )
        guided_velocity = meanflowql_adjoint_guided_velocity(
            conditional_velocity,
            adjoint,
            self.config["adjoint_eta"],
            intermediate_time,
        )

        def pre_direct_map(current_state, time):
            return self._call_actor_snapshot(
                self.pre_actor_params,
                observations,
                current_state,
                time,
            )

        _, reward_derivative = jax.jvp(
            pre_direct_map,
            (intermediate_state, intermediate_time),
            (guided_velocity, jnp.ones_like(intermediate_time)),
        )
        reward_direct_target = meanflowql_reformulated_target(
            intermediate_state,
            jnp.zeros_like(intermediate_time),
            intermediate_time,
            guided_velocity,
            reward_derivative,
        )
        reward_direct_target = jnp.clip(
            jax.lax.stop_gradient(reward_direct_target), -5.0, 5.0
        )
        target, reward_velocity, bootstrap_velocity = (
            meanflowql_alphaflow_target(
                state,
                intermediate_state,
                reward_direct_target,
                bootstrap_direct_map,
                alpha,
            )
        )
        target = jnp.clip(jax.lax.stop_gradient(target), -5.0, 5.0)

        prediction = self.network.select("actor_bc_flow")(
            observations, state, current_time, params=grad_params
        )
        error = prediction - target
        meanflow_loss = self.adaptive_l2_loss(
            error, current_time, mode="normal"
        )
        if self.config["normalize_alphaflow_loss_by_alpha"]:
            meanflow_loss = meanflow_loss / jnp.maximum(
                alpha, self.config["alphaflow_alpha_eps"]
            )

        consistency_loss = self.consistency_loss(
            batch, grad_params, consistency_rng
        )
        flow_loss = (
            meanflow_loss
            + self.config.get("consistency_alpha", 0.0)
            * consistency_loss
        )
        _, baseline_target = self._baseline_target(
            observations,
            state,
            current_time,
            conditional_velocity,
            grad_params,
        )
        student_velocity = state - prediction
        reward_error = student_velocity - jax.lax.stop_gradient(
            reward_velocity
        )
        bootstrap_error = student_velocity - jax.lax.stop_gradient(
            bootstrap_velocity
        )
        critical, tfm_a, tfm_b, tc_c = critical_alpha(
            reward_error, bootstrap_error
        )
        correction = guided_velocity - conditional_velocity

        return flow_loss, {
            "mean_flow_loss": meanflow_loss,
            "consistency_loss": consistency_loss,
            "flow_loss": flow_loss,
            "am_enabled": jnp.asarray(1.0),
            "alphaflow_alpha": alpha,
            "critical_alpha": critical,
            "jvp_branch": jnp.asarray(0.0),
            "reward_target_loss": jnp.mean(jnp.square(reward_error)),
            "bootstrap_loss": jnp.mean(jnp.square(bootstrap_error)),
            "mixed_target_loss": jnp.mean(jnp.square(error)),
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
                reward_derivative, axis=-1
            ).mean(),
            "target_norm": jnp.linalg.norm(target, axis=-1).mean(),
            "baseline_target_norm": jnp.linalg.norm(
                baseline_target, axis=-1
            ).mean(),
            "target_shift_norm": jnp.linalg.norm(
                target - baseline_target, axis=-1
            ).mean(),
            "remaining_time": intermediate_time.mean(),
            "bootstrap_interval": (
                current_time - intermediate_time
            ).mean(),
            "endpoint_q": self._aggregate_target_q(
                observations, endpoint
            ).mean(),
            "endpoint_out_of_bounds_fraction": (
                jnp.abs(endpoint) > 1.0
            ).mean(),
            "tfm_a": tfm_a,
            "tfm_b": tfm_b,
            "tc_c": tc_c,
        }

    def _zero_alphaflow_loss(self, batch, grad_params, rng, alpha):
        """Use the exact EMA MeanFlowQL JVP consistency limit at alpha=0."""

        consistency_rng, time_rng, _, noise_rng = jax.random.split(rng, 4)
        state, current_time, conditional_velocity = self._sample_path(
            batch, time_rng, noise_rng
        )
        observations = batch["observations"]

        def target_direct_map(current_state, time):
            return self._call_actor_snapshot(
                self.target_actor_params,
                observations,
                current_state,
                time,
            )

        bootstrap_direct_map = target_direct_map(state, current_time)
        bootstrap_velocity = state - bootstrap_direct_map
        _, derivative = jax.jvp(
            target_direct_map,
            (state, current_time),
            (bootstrap_velocity, jnp.ones_like(current_time)),
        )
        target = meanflowql_reformulated_target(
            state,
            jnp.zeros_like(current_time),
            current_time,
            bootstrap_velocity,
            derivative,
        )
        target = jnp.clip(jax.lax.stop_gradient(target), -5.0, 5.0)
        prediction = self.network.select("actor_bc_flow")(
            observations, state, current_time, params=grad_params
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
        _, baseline_target = self._baseline_target(
            observations,
            state,
            current_time,
            conditional_velocity,
            grad_params,
        )
        endpoint = meanflowql_endpoint_map(
            state, current_time, bootstrap_direct_map
        )
        zeros = jnp.asarray(0.0)
        return flow_loss, {
            "mean_flow_loss": meanflow_loss,
            "consistency_loss": consistency_loss,
            "flow_loss": flow_loss,
            "am_enabled": zeros,
            "alphaflow_alpha": alpha,
            "critical_alpha": zeros,
            "jvp_branch": jnp.asarray(1.0),
            "reward_target_loss": zeros,
            "bootstrap_loss": jnp.mean(
                jnp.square(prediction - bootstrap_direct_map)
            ),
            "mixed_target_loss": jnp.mean(jnp.square(error)),
            "adjoint_norm": zeros,
            "correction_norm": zeros,
            "conditional_velocity_norm": jnp.linalg.norm(
                conditional_velocity, axis=-1
            ).mean(),
            "guided_velocity_norm": jnp.linalg.norm(
                bootstrap_velocity, axis=-1
            ).mean(),
            "derivative_norm": jnp.linalg.norm(
                derivative, axis=-1
            ).mean(),
            "target_norm": jnp.linalg.norm(target, axis=-1).mean(),
            "baseline_target_norm": jnp.linalg.norm(
                baseline_target, axis=-1
            ).mean(),
            "target_shift_norm": jnp.linalg.norm(
                target - baseline_target, axis=-1
            ).mean(),
            "remaining_time": zeros,
            "bootstrap_interval": current_time.mean(),
            "endpoint_q": self._aggregate_target_q(
                observations, endpoint
            ).mean(),
            "endpoint_out_of_bounds_fraction": (
                jnp.abs(endpoint) > 1.0
            ).mean(),
            "tfm_a": zeros,
            "tfm_b": zeros,
            "tc_c": zeros,
        }

    def _meanflowql_loss(self, batch, grad_params, rng, use_am):
        """Evaluate baseline pretraining or full AM-AlphaFlow refinement."""

        if not use_am:
            loss, info = MeanFlowQL_Agent.meanflow_loss(
                self, batch, grad_params, rng
            )
            info = dict(info)
            info.update(
                {
                    "am_enabled": jnp.asarray(0.0),
                    "alphaflow_alpha": jnp.asarray(1.0),
                    "jvp_branch": jnp.asarray(0.0),
                }
            )
            return loss, info

        alpha = self._alphaflow_alpha()
        if (
            self.config["alphaflow_alpha_mode"] == "fixed"
            and self.config["alphaflow_alpha_value"]
            <= self.config["alphaflow_alpha_eps"]
        ):
            return self._zero_alphaflow_loss(
                batch, grad_params, rng, alpha
            )
        return self._positive_alphaflow_loss(
            batch, grad_params, rng, alpha
        )

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
    def update(self, batch, current_step=0):
        """Run MeanFlowQL update, then advance the AlphaFlow EMA actor."""

        updated, info = MeanFlowQL_Agent.update(
            self, batch, current_step=current_step
        )
        online_actor = self._actor_snapshot(updated.network.params)
        target_actor = jax.tree_util.tree_map(
            lambda online, target: (
                self.config["alphaflow_target_tau"] * online
                + (1.0 - self.config["alphaflow_target_tau"]) * target
            ),
            online_actor,
            self.target_actor_params,
        )
        alpha = self._alphaflow_alpha()
        return updated.replace(
            target_actor_params=target_actor,
            alphaflow_updates=self.alphaflow_updates + 1,
            current_alphaflow_alpha=alpha,
        ), info

    @jax.jit
    def pretrain(self, batch, current_step=None):
        """Pretrain the unmodified MeanFlowQL target without critic guidance."""

        new_rng, loss_rng = jax.random.split(self.rng)

        def pretrain_loss(grad_params):
            return self.behavior_meanflow_loss(
                batch, grad_params, rng=loss_rng
            )

        network, info = self.network.apply_loss_fn(loss_fn=pretrain_loss)
        online_actor = self._actor_snapshot(network.params)
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
        return self.replace(
            network=network,
            rng=new_rng,
            pre_actor_params=online_actor,
            target_actor_params=copy.deepcopy(online_actor),
            alphaflow_updates=jnp.asarray(0, dtype=jnp.int32),
            current_alphaflow_alpha=self._initial_alphaflow_alpha(),
        ), info

    def initialize_alphaflow_stage(self):
        """Freeze behavior actor and initialize the EMA actor explicitly."""

        online_actor = self._actor_snapshot(self.network.params)
        return self.replace(
            pre_actor_params=copy.deepcopy(online_actor),
            target_actor_params=copy.deepcopy(online_actor),
            alphaflow_updates=jnp.asarray(0, dtype=jnp.int32),
            current_alphaflow_alpha=self._initial_alphaflow_alpha(),
        )

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
            "alphaflow_alpha_mode",
            "alphaflow_alpha_value",
            "alphaflow_start_step",
            "alphaflow_warmup_steps",
            "alphaflow_transition_steps",
            "alphaflow_alpha_floor",
            "alphaflow_gamma",
            "alphaflow_alpha_eps",
            "alphaflow_target_tau",
            "normalize_alphaflow_loss_by_alpha",
        ):
            merged[key] = changed_defaults[key]
        if config is not None:
            if hasattr(config, "to_dict"):
                config = config.to_dict()
            merged.update(dict(config))
        merged["agent_name"] = cls.AGENT_NAME
        merged["am_target_variant"] = cls.TARGET_VARIANT
        cls._validate_am_config(merged)
        online_actor = cls._actor_snapshot(meanflowql_agent.network.params)
        return cls(
            rng=meanflowql_agent.rng,
            network=meanflowql_agent.network,
            config=FrozenDict(merged),
            current_alpha=meanflowql_agent.current_alpha,
            loss_history=meanflowql_agent.loss_history,
            valid_count=meanflowql_agent.valid_count,
            pre_actor_params=copy.deepcopy(online_actor),
            target_actor_params=copy.deepcopy(online_actor),
            alphaflow_updates=jnp.asarray(0, dtype=jnp.int32),
            current_alphaflow_alpha=jnp.asarray(1.0, dtype=jnp.float32),
        )

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        local_config = copy.deepcopy(config)
        local_config["agent_name"] = cls.AGENT_NAME
        cls._validate_am_config(local_config)
        agent = super().create(
            seed, ex_observations, ex_actions, local_config
        )
        online_actor = cls._actor_snapshot(agent.network.params)
        return agent.replace(
            pre_actor_params=copy.deepcopy(online_actor),
            target_actor_params=copy.deepcopy(online_actor),
            alphaflow_updates=jnp.asarray(0, dtype=jnp.int32),
            current_alphaflow_alpha=agent._initial_alphaflow_alpha(),
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

    # This alpha is deliberately separate from MeanFlowQL's `alpha`, which
    # remains the outer MFI/BC coefficient used by the paper configuration.
    config.alphaflow_alpha_mode = "anneal"
    config.alphaflow_alpha_value = 1.0
    config.alphaflow_start_step = 0
    config.alphaflow_warmup_steps = 50000
    config.alphaflow_transition_steps = 400000
    config.alphaflow_alpha_floor = 0.05
    config.alphaflow_gamma = 8.0
    config.alphaflow_alpha_eps = 1e-4
    config.alphaflow_target_tau = 0.005
    config.normalize_alphaflow_loss_by_alpha = True

    # The note-style EMA bootstrap is the default consistency mechanism.
    # Keep the older pairwise endpoint MSE available only as an explicit
    # supplemental ablation.
    config.consistency_alpha = 0.0
    return ml_collections.ConfigDict(config)
