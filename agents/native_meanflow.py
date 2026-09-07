"""Native MeanFlow policy with direct one-step Q-learning.

Unlike agents/meanflowql.py, which predicts a reformulated endpoint map, this
agent explicitly learns the interval-average velocity u(o, x_t, r, t). It is
kept as an isolated baseline so the original MeanFlowQL checkpoint tree and
training behavior remain unchanged.
"""

import copy
from functools import partial

import flax
from flax.core import FrozenDict, unfreeze
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.meanflowql import MeanFlowQL_Agent
from utils.dit_jax import MFDiT
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.native_meanflow import (
    endpoint_from_average_velocity,
    meanflow_jvp_target,
    reverse_transport_step,
    sample_time_pairs,
)
from utils.networks import Value


class NativeMeanFlowAgent(MeanFlowQL_Agent):
    """OFQL-style native MeanFlow actor in the MeanFlowQL RL pipeline."""

    def _velocity(
        self,
        observations,
        x_t,
        r,
        t,
        params=None,
        target=False,
    ):
        module_name = "target_actor_bc_flow" if target else "actor_bc_flow"
        return self.network.select(module_name)(
            observations,
            x_t,
            r=r,
            t=t,
            params=params,
        )

    def _sample_time_pairs(self, rng, batch_size):
        return sample_time_pairs(
            rng,
            batch_size=batch_size,
            flow_ratio=self.config["flow_ratio"],
            distribution=self.config["time_distribution"],
            logit_mean=self.config["time_logit_mean"],
            logit_std=self.config["time_logit_std"],
        )

    def meanflow_loss(self, batch, grad_params, rng):
        """Fit the full interval-average velocity with the MeanFlow identity."""

        batch_size = batch["actions"].shape[0]
        time_rng, noise_rng = jax.random.split(rng)
        t, r, instantaneous_mask = self._sample_time_pairs(
            time_rng, batch_size
        )

        actions = batch["actions"]
        noise = self.sample_noise(noise_rng, actions.shape)
        x_t = (1.0 - t) * actions + t * noise
        conditional_velocity = noise - actions

        def field_fn(x, target_time, current_time):
            return self._velocity(
                batch["observations"],
                x,
                target_time,
                current_time,
                params=grad_params,
            )

        u_pred, u_target, du_dt = meanflow_jvp_target(
            field_fn,
            x_t,
            r,
            t,
            conditional_velocity,
        )
        squared_error = jnp.square(u_pred - u_target)
        meanflow_loss = squared_error.mean()

        return meanflow_loss, {
            "mean_flow_loss": meanflow_loss,
            "identity_residual": squared_error.mean(),
            "instantaneous_fraction": instantaneous_mask.mean(),
            "mean_interval": (t - r).mean(),
            "velocity_norm": jnp.linalg.norm(conditional_velocity, axis=-1).mean(),
            "derivative_norm": jnp.linalg.norm(du_dt, axis=-1).mean(),
        }

    def _endpoint_actions(
        self,
        observations,
        rng,
        target=False,
        num_candidates=1,
        select_best=True,
    ):
        """Generate one-step endpoint actions, optionally with/choose best-of-N."""

        batch_size = observations.shape[0]
        action_dim = self.config["action_dim"]
        candidate_count = int(num_candidates)
        if candidate_count < 1:
            raise ValueError(
                f"num_candidates must be positive, got {candidate_count}."
            )

        candidate_rngs = jax.random.split(rng, candidate_count)
        noise = jax.vmap(
            lambda key: self.sample_noise(key, (batch_size, action_dim))
        )(candidate_rngs)

        observations_expanded = jnp.tile(
            observations[None, ...],
            (candidate_count, 1) + (1,) * (observations.ndim - 1),
        )
        flat_observations = observations_expanded.reshape(
            candidate_count * batch_size, *observations.shape[1:]
        )
        flat_noise = noise.reshape(candidate_count * batch_size, action_dim)
        flat_r = jnp.zeros((candidate_count * batch_size, 1))
        flat_t = jnp.ones((candidate_count * batch_size, 1))

        average_velocity = self._velocity(
            flat_observations,
            flat_noise,
            flat_r,
            flat_t,
            target=target,
        )
        raw_actions = endpoint_from_average_velocity(
            flat_noise, average_velocity
        )
        candidate_actions = jnp.clip(raw_actions, -1.0, 1.0).reshape(
            candidate_count, batch_size, action_dim
        )

        if candidate_count == 1 or not select_best:
            return candidate_actions[0]

        flat_actions = candidate_actions.reshape(
            candidate_count * batch_size, action_dim
        )
        q_values = self.network.select("target_critic")(
            flat_observations, actions=flat_actions
        )
        q_values = self._aggregate_q(q_values).reshape(
            candidate_count, batch_size
        )
        best_indices = jnp.argmax(q_values, axis=0)
        return candidate_actions[best_indices, jnp.arange(batch_size)]

    def _aggregate_q(self, q_values):
        if self.config["q_agg"] == "min":
            return q_values.min(axis=0)
        if self.config["q_agg"] == "max":
            return q_values.max(axis=0)
        if self.config["q_agg"] == "mean":
            return q_values.mean(axis=0)
        raise ValueError(
            f"Unsupported q_agg {self.config['q_agg']!r}; "
            "expected 'min', 'max', or 'mean'."
        )

    def critic_loss(self, batch, grad_params, rng):
        """TD critic loss bootstrapped with the EMA native MeanFlow actor."""

        candidate_count = max(1, int(self.config["num_candidates"]) // 2)
        next_actions = self._endpoint_actions(
            batch["next_observations"],
            rng,
            target=True,
            num_candidates=candidate_count,
        )
        next_qs = self.network.select("target_critic")(
            batch["next_observations"], actions=next_actions
        )
        next_q = self._aggregate_q(next_qs)
        target_q = (
            batch["rewards"]
            + self.config["discount"] * batch["masks"] * next_q
        )
        target_q = jax.lax.stop_gradient(target_q)

        q_values = self.network.select("critic")(
            batch["observations"],
            actions=batch["actions"],
            params=grad_params,
        )
        critic_loss = jnp.square(q_values - target_q).mean()
        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q_values.mean(),
            "q_max": q_values.max(),
            "q_min": q_values.min(),
            "target_q_mean": target_q.mean(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Apply direct Q-learning to the native one-step endpoint."""

        batch_size, action_dim = batch["actions"].shape
        noise = self.sample_noise(rng, (batch_size, action_dim))
        r = jnp.zeros((batch_size, 1))
        t = jnp.ones((batch_size, 1))
        average_velocity = self._velocity(
            batch["observations"],
            noise,
            r,
            t,
            params=grad_params,
        )
        raw_actions = endpoint_from_average_velocity(
            noise, average_velocity
        )

        bound_loss = (
            jax.nn.relu(raw_actions - 1.0).mean()
            + jax.nn.relu(-1.0 - raw_actions).mean()
        )
        actions = jnp.clip(raw_actions, -1.0, 1.0)
        if self.config["q_coef"] > 0.0:
            q_values = self.network.select("critic")(
                batch["observations"], actions=actions
            )
            q_values = self._aggregate_q(q_values)
            q_loss = -q_values.mean()
            if self.config["normalize_q_loss"]:
                q_scale = jax.lax.stop_gradient(
                    1.0 / jnp.maximum(jnp.abs(q_values).mean(), 1e-6)
                )
                q_loss = q_scale * q_loss
        else:
            q_values = jnp.zeros((batch_size,))
            q_loss = jnp.array(0.0)

        actor_loss = (
            self.config["q_coef"] * q_loss
            + self.config["bound_loss_weight"] * bound_loss
        )
        return actor_loss, {
            "actor_loss": actor_loss,
            "q_loss": q_loss,
            "q": q_values.mean(),
            "bound_loss": bound_loss,
            "action_std": actions.std(axis=0).mean(),
            "action_clip_fraction": (jnp.abs(raw_actions) > 1.0).mean(),
        }

    def total_loss(self, batch, grad_params, rng, current_step=0):
        meanflow_rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        if self.config["critic_coef"] > 0.0:
            critic_loss, critic_info = self.critic_loss(
                batch, grad_params, critic_rng
            )
        else:
            critic_loss = jnp.array(0.0)
            critic_info = {
                "critic_loss": critic_loss,
                "q_mean": critic_loss,
                "q_max": critic_loss,
                "q_min": critic_loss,
                "target_q_mean": critic_loss,
            }
        meanflow_loss, meanflow_info = self.meanflow_loss(
            batch, grad_params, meanflow_rng
        )
        actor_loss, actor_info = self.actor_loss(
            batch, grad_params, actor_rng
        )

        total_loss = (
            self.config["critic_coef"] * critic_loss
            + self.config["meanflow_coef"] * meanflow_loss
            + actor_loss
        )
        info = {f"critic/{key}": value for key, value in critic_info.items()}
        info.update(
            {f"meanflow/{key}": value for key, value in meanflow_info.items()}
        )
        info.update({f"actor/{key}": value for key, value in actor_info.items()})
        info["meanflow_coef"] = self.config["meanflow_coef"]
        info["total_loss"] = total_loss
        return total_loss, info

    def _update_ema_module(self, network, online_name, target_name):
        params = dict(network.params)
        params[target_name] = jax.tree_util.tree_map(
            lambda online, target: (
                self.config["tau"] * online
                + (1.0 - self.config["tau"]) * target
            ),
            params[online_name],
            params[target_name],
        )
        return network.replace(params=params)

    @jax.jit
    def update(self, batch, current_step=0):
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
        return self.replace(network=network, rng=new_rng), info

    @jax.jit
    def pretrain(self, batch, current_step=None):
        """Update only the native MeanFlow actor using behavior data."""

        new_rng, loss_rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.meanflow_loss(batch, grad_params, loss_rng)

        network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        network = self._update_ema_module(
            network,
            "modules_actor_bc_flow",
            "modules_target_actor_bc_flow",
        )
        return self.replace(network=network, rng=new_rng), info

    @partial(jax.jit, static_argnames=("num_candidates",))
    def sample_actions(
        self,
        observations,
        temperature=1,
        seed=None,
        num_candidates=None,
    ):
        """Generate K1 actions; best-of-N is used only when N is greater than 1."""

        del temperature
        candidate_count = (
            int(self.config["num_candidates"])
            if num_candidates is None
            else int(num_candidates)
        )
        return self._endpoint_actions(
            observations,
            seed,
            target=self.config["eval_target_actor"],
            num_candidates=candidate_count,
        )

    @partial(jax.jit, static_argnames=("num_steps", "target"))
    def sample_actions_nfe(
        self,
        observations,
        seed,
        num_steps=1,
        target=False,
    ):
        """Generate actions with K reverse MeanFlow transport intervals."""

        if num_steps < 1:
            raise ValueError(f"num_steps must be positive, got {num_steps}.")
        action_shape = (observations.shape[0], self.config["action_dim"])
        state = self.sample_noise(seed, action_shape)
        step_size = 1.0 / num_steps
        for index in range(num_steps, 0, -1):
            t = jnp.full((observations.shape[0], 1), index * step_size)
            r = jnp.full(
                (observations.shape[0], 1), (index - 1) * step_size
            )
            average_velocity = self._velocity(
                observations, state, r, t, target=target
            )
            state = reverse_transport_step(
                state, average_velocity, r, t
            )
        return jnp.clip(state, -1.0, 1.0)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        batch_size = ex_observations.shape[0]
        ex_t = jnp.ones((batch_size, 1))
        ex_r = jnp.zeros((batch_size, 1))
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        encoders = {}
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["critic"] = encoder_module()
            encoders["actor_bc_flow"] = encoder_module()

        critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=2,
            encoder=encoders.get("critic"),
        )
        actor_def = MFDiT(
            hidden_dim=config["actor_hidden_dims"],
            depth=config["actor_depth"],
            num_heads=config["actor_num_heads"],
            output_dim=action_dim,
            encoder=encoders.get("actor_bc_flow"),
            tanh_squash=config["tanh_squash"],
            use_output_layernorm=config["use_output_layernorm"],
        )

        network_info = {
            "critic": (critic_def, (ex_observations, ex_actions)),
            "target_critic": (
                copy.deepcopy(critic_def),
                (ex_observations, ex_actions),
            ),
            "actor_bc_flow": (
                actor_def,
                (ex_observations, ex_actions, ex_r, ex_t),
            ),
            "target_actor_bc_flow": (
                copy.deepcopy(actor_def),
                (ex_observations, ex_actions, ex_r, ex_t),
            ),
        }
        networks = {name: value[0] for name, value in network_info.items()}
        network_args = {name: value[1] for name, value in network_info.items()}
        network_def = ModuleDict(networks)

        network_params = network_def.init(init_rng, **network_args)["params"]
        params = unfreeze(network_params)
        params["modules_target_critic"] = copy.deepcopy(
            params["modules_critic"]
        )
        params["modules_target_actor_bc_flow"] = copy.deepcopy(
            params["modules_actor_bc_flow"]
        )

        def parameter_labels(parameter_tree):
            flat_params = flax.traverse_util.flatten_dict(parameter_tree)
            labels = {}
            for key in flat_params:
                path = "/".join(key)
                labels[key] = "critic" if "critic" in path else "actor"
            return flax.traverse_util.unflatten_dict(labels)

        actor_tx = optax.chain(
            optax.clip_by_global_norm(config["grad_clip_norm"]),
            optax.adam(learning_rate=config["lr"]),
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(config["grad_clip_norm"]),
            optax.adam(learning_rate=config["critic_lr"]),
        )
        labels = parameter_labels(params)
        network_tx = optax.multi_transform(
            {"actor": actor_tx, "critic": critic_tx}, labels
        )
        network = TrainState.create(
            network_def, params, tx=network_tx
        )

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim
        history_size = config.get("loss_history_window_size", 1)
        return cls(
            rng=rng,
            network=network,
            config=FrozenDict(**config),
            current_alpha=config["meanflow_coef"],
            loss_history=jnp.zeros(history_size),
        )


def get_config():
    return ml_collections.ConfigDict(
        {
            "agent_name": "native_meanflow",
            "ob_dims": ml_collections.config_dict.placeholder(list),
            "action_dim": ml_collections.config_dict.placeholder(int),
            "encoder": ml_collections.config_dict.placeholder(str),
            "batch_size": 256,
            "sigma": 1.0,
            "noise_type": "gaussian",
            "time_distribution": "logit_normal",
            "time_logit_mean": -0.4,
            "time_logit_std": 1.0,
            "flow_ratio": 0.5,
            "meanflow_coef": 10.0,
            "q_coef": 1.0,
            "critic_coef": 1.0,
            "normalize_q_loss": True,
            "bound_loss_weight": 1.0,
            "value_hidden_dims": (512, 512, 512, 512),
            "layer_norm": True,
            "q_agg": "mean",
            "discount": 0.99,
            "tau": 0.005,
            "lr": 1e-4,
            "critic_lr": 3e-4,
            "grad_clip_norm": 1.0,
            "actor_hidden_dims": 256,
            "actor_depth": 3,
            "actor_num_heads": 2,
            "tanh_squash": False,
            "use_output_layernorm": False,
            "num_candidates": 1,
            "eval_target_actor": False,
            "loss_history_window_size": 1,
        }
    )
