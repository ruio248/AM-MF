"""Note adjoint control implemented as an upstream-runner-compatible agent.

The public MeanFlowQL main loop sees the same create/update/sample_actions
interface as the baseline.  Behavior initialization, prior freezing, EMA,
adjoint control, interval transport, and Jacobian matching all live here so
the upstream runner and evaluator do not need method-specific branches.
"""

import copy
from functools import partial
from typing import Any

import flax
import flax.linen as nn
from flax.core import freeze, unfreeze
import jax
import jax.numpy as jnp
import optax

from agents.meanflowql import MeanFlowQL_Agent, get_config as base_config
from utils.dit_jax import MFDiT
from utils.flax_utils import ModuleDict, TrainState
from utils.note_flow import (
    clipping_metrics,
    controlled_velocity,
    directional_difference,
    endpoint_adjoint,
    interval_map,
    rk2_endpoint_from_grid,
    rk2_path,
)


class IntervalDirectMap(nn.Module):
    """Double-time residual map with the legacy positional-t interface."""

    hidden_dim: int
    depth: int
    num_heads: int
    output_dim: int

    @nn.compact
    def __call__(self, observations, actions, t=None, r=None, **kwargs):
        r = jnp.zeros_like(t) if r is None else r
        return MFDiT(
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            output_dim=self.output_dim,
            name="interval_core",
        )(observations, actions, r=r, t=t)


class AMMeanFlowNoteAgent(MeanFlowQL_Agent):
    """Interval MeanFlow agent with endpoint-reward adjoint credit assignment."""

    pre_actor_params: Any = None
    target_actor_params: Any = None
    completed_updates: Any = 0

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        if config["variant"] not in ("note_adjoint", "direct_q_control"):
            raise ValueError("Unknown Note variant")
        if config["encoder"] is not None or config["consistency_alpha"] != 0:
            raise ValueError("Note v1 supports vector observations and consistency_alpha=0")
        if (
            config["behavior_warmup_updates"] < 0
            or config["control_eta_ramp_updates"] < 0
            or config["control_uncertainty_end_update"] < -1
        ):
            raise ValueError("warmup and control-ramp updates must be nonnegative")
        for key in (
            "teacher_steps",
            "teacher_batch_size",
            "jacobian_interval",
            "jacobian_batch_size",
        ):
            if config[key] < 1:
                raise ValueError(f"{key} must be positive")
        if config["teacher_solver"] != "midpoint_rk2" or config["jacobian_fd_eps"] <= 0:
            raise ValueError("Invalid teacher or finite-difference configuration")
        if min(
            config["control_eta"],
            config["control_adjoint_clip"],
            config["control_uncertainty_scale"],
            config["jacobian_coef"],
            config["transport_coef"],
        ) < 0:
            raise ValueError("Loss/control coefficients must be nonnegative")
        if not 0 < config["target_actor_tau"] <= 1:
            raise ValueError("Invalid EMA coefficient")

        base = MeanFlowQL_Agent.create(seed, ex_observations, ex_actions, config)
        modules = dict(base.network.model_def.modules)
        modules["actor_bc_flow"] = IntervalDirectMap(
            config["actor_hidden_dims"],
            config["actor_depth"],
            config["actor_num_heads"],
            ex_actions.shape[-1],
        )
        definition = ModuleDict(modules)
        t = jnp.ones((len(ex_actions), 1))
        init_rng = jax.random.split(jax.random.PRNGKey(seed))[1]
        params = unfreeze(
            definition.init(
                init_rng,
                critic=(ex_observations, ex_actions),
                target_critic=(ex_observations, ex_actions),
                actor_bc_flow=(ex_observations, ex_actions, t),
            )["params"]
        )

        # Preserve the baseline critic initialization exactly.  Only the actor
        # architecture differs between B0 and N.
        for name in ("modules_critic", "modules_target_critic"):
            params[name] = unfreeze(base.network.params[name])
        labels = {
            name: jax.tree_util.tree_map(
                lambda _: "critic" if "critic" in name else "actor", subtree
            )
            for name, subtree in params.items()
        }
        tx = optax.multi_transform(
            {
                "actor": optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adam(base.config["actor_lr_schedule"]),
                ),
                "critic": optax.chain(
                    optax.clip_by_global_norm(1.0), optax.adam(3e-4)
                ),
            },
            labels,
        )
        network = TrainState(
            step=1,
            apply_fn=definition.apply,
            model_def=definition,
            params=freeze(params),
            tx=tx,
            opt_state=tx.init(params),
        )
        actor = params["modules_actor_bc_flow"]
        return cls(
            rng=base.rng,
            network=network,
            config=base.config,
            current_alpha=base.current_alpha,
            loss_history=base.loss_history,
            pre_actor_params=copy.deepcopy(actor),
            target_actor_params=copy.deepcopy(actor),
            completed_updates=jnp.asarray(0),
        )

    def g(self, observations, x, r, t, params=None, snapshot=None):
        if snapshot is not None:
            params = dict(self.network.params)
            params["modules_actor_bc_flow"] = snapshot
        return self.network.select("actor_bc_flow")(
            observations, x, t=t, r=r, params=params
        )

    def endpoint(self, observations, x, r, t, params=None, snapshot=None):
        return interval_map(x, r, t, self.g(observations, x, r, t, params, snapshot))

    def target_reward(self, observations, raw_actions):
        return self.network.select("target_critic")(
            observations, actions=jnp.clip(raw_actions, -1, 1)
        ).mean(axis=0)

    def adjoint(self, observations, x, t):
        return endpoint_adjoint(
            lambda z: self.endpoint(
                observations,
                z,
                jnp.zeros_like(t),
                t,
                snapshot=self.target_actor_params,
            ),
            lambda a: self.target_reward(observations, a),
            x,
        )

    def scheduled_control_eta(self, current_step):
        """Ramp the AM control after behavior initialization when requested."""
        ramp_updates = self.config["control_eta_ramp_updates"]
        if ramp_updates == 0:
            return jnp.asarray(self.config["control_eta"])
        guided_updates = jnp.maximum(
            jnp.asarray(current_step) - self.config["behavior_warmup_updates"], 0
        )
        ramp = jnp.minimum(guided_updates / ramp_updates, 1.0)
        return jnp.asarray(self.config["control_eta"]) * ramp

    def scheduled_uncertainty_scale(self, current_step):
        """Disable the uncertainty gate after a configured offline boundary."""
        end_update = self.config["control_uncertainty_end_update"]
        scale = jnp.asarray(self.config["control_uncertainty_scale"])
        if end_update < 0:
            return scale
        return jnp.where(jnp.asarray(current_step) < end_update, scale, 0.0)

    def control_terms(self, observations, x, t, uncertainty_scale=None):
        """Return a clipped, uncertainty-gated endpoint-Q adjoint.

        The gate is based on the relative disagreement of the two target
        critics at the endpoint. A zero scale disables it exactly, which
        preserves the original N behavior for all existing experiments.
        """
        endpoint, adjoint = self.adjoint(observations, x, t)
        raw_norm = jnp.linalg.norm(adjoint, axis=-1)
        clip = self.config["control_adjoint_clip"]
        if clip == 0:
            clip_scale = jnp.ones_like(raw_norm)
        else:
            clip_scale = jnp.minimum(1.0, clip / jnp.maximum(raw_norm, 1e-8))
        clipped_adjoint = adjoint * clip_scale[:, None]

        q_values = self.network.select("target_critic")(
            observations, actions=jnp.clip(endpoint, -1, 1)
        )
        q_mean = jnp.mean(q_values, axis=0)
        q_std = jnp.std(q_values, axis=0)
        relative_disagreement = q_std / jnp.maximum(jnp.abs(q_mean), 1.0)
        scale = (
            jnp.asarray(self.config["control_uncertainty_scale"])
            if uncertainty_scale is None
            else uncertainty_scale
        )
        gated = 1.0 / (
            1.0 + relative_disagreement / jnp.maximum(scale, 1e-8)
        )
        gate = jnp.where(scale > 0, gated, jnp.ones_like(relative_disagreement))
        return clipped_adjoint, gate, {
            "raw_norm": raw_norm,
            "clipped_norm": jnp.linalg.norm(clipped_adjoint, axis=-1),
            "gate": gate,
            "relative_disagreement": relative_disagreement,
            "clipped_fraction": 1.0 - clip_scale,
        }

    def teacher_field(
        self, observations, x, t, control_eta=None, uncertainty_scale=None
    ):
        prior = x - self.g(
            observations, x, t, t, snapshot=self.pre_actor_params
        )
        if self.config["variant"] == "direct_q_control" or self.config["control_eta"] == 0:
            return prior
        eta = (
            jnp.asarray(self.config["control_eta"])
            if control_eta is None
            else control_eta
        )
        adjoint, gate, _ = self.control_terms(
            observations, x, t, uncertainty_scale
        )
        return controlled_velocity(prior, gate[:, None] * adjoint, eta)

    def teacher_path(
        self, observations, noise, steps=None, control_eta=None, uncertainty_scale=None
    ):
        return rk2_path(
            lambda x, t: self.teacher_field(
                observations, x, t, control_eta, uncertainty_scale
            ),
            noise,
            steps=self.config["teacher_steps"] if steps is None else steps,
        )

    def meanflow_loss(self, batch, grad_params, rng):
        """Warmup MFI: 25% r=0, 25% diagonal, 50% proper intervals."""
        batch_size = len(batch["actions"])
        tk, rk, ek = jax.random.split(rng, 3)
        t_indices = jax.random.randint(
            tk, (batch_size, 1), 1, self.config["time_steps"] + 1
        )
        t = t_indices / self.config["time_steps"]
        r = (
            jnp.floor(jax.random.uniform(rk, (batch_size, 1)) * t_indices)
            / self.config["time_steps"]
        )
        rows = jnp.arange(batch_size)[:, None]
        r = jnp.where(rows < batch_size // 4, 0, r)
        r = jnp.where(
            (rows >= batch_size // 4) & (rows < batch_size // 2), t, r
        )
        noise = self.sample_noise(ek, batch["actions"].shape)
        velocity = noise - batch["actions"]
        x = (1 - t) * batch["actions"] + t * noise
        prediction, derivative = jax.jvp(
            lambda z, rt, tt: self.g(
                batch["observations"], z, rt, tt, params=grad_params
            ),
            (x, r, t),
            (velocity, jnp.zeros_like(r), jnp.ones_like(t)),
        )
        target = x + (t - r - 1) * velocity - (t - r) * derivative
        target = jax.lax.stop_gradient(jnp.clip(target, -5, 5))
        loss = self.adaptive_l2_loss(prediction - target, t)
        return loss, {
            "mean_flow_loss": loss,
            "flow_loss": loss,
            "consistency_loss": jnp.asarray(0.0),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Warmup actor loss: boundary regularization, with no reward signal."""
        _, noise_rng = jax.random.split(rng)
        noise = self.sample_noise(noise_rng, batch["actions"].shape)
        one = jnp.ones((len(noise), 1))
        raw = self.g(
            batch["observations"],
            noise,
            jnp.zeros_like(one),
            one,
            params=grad_params,
        )
        bound = (jax.nn.relu(raw - 1) + jax.nn.relu(-1 - raw)).mean()
        return bound * self.config["bound_loss_weight"], {
            "bound_loss": bound,
            "q_loss": jnp.asarray(0.0),
        }

    def transport_batch(
        self, observations, rng, control_eta=None, uncertainty_scale=None
    ):
        noise_key, diagonal_key, pair_key = jax.random.split(rng, 3)
        count = min(self.config["teacher_batch_size"], len(observations))
        obs = observations[:count]
        noise = self.sample_noise(
            noise_key, (count, self.config["action_dim"])
        )
        path = self.teacher_path(
            obs, noise, control_eta=control_eta,
            uncertainty_scale=uncertainty_scale
        )
        steps = self.config["teacher_steps"]
        diagonal = jax.random.randint(diagonal_key, (count, 2), 0, steps + 1)
        starts, ends = jnp.triu_indices(steps + 1, k=1)
        choices = jax.random.randint(pair_key, (count, 5), 0, len(starts))
        i = jnp.concatenate(
            [
                jnp.zeros((count, 1), dtype=jnp.int32),
                diagonal,
                starts[choices],
            ],
            axis=1,
        )
        j = jnp.concatenate(
            [
                jnp.full((count, 1), steps, dtype=jnp.int32),
                diagonal,
                ends[choices],
            ],
            axis=1,
        )
        rows = jnp.arange(count)[:, None]
        x = path[i, rows]
        y = path[j, rows]
        t = (1 - i / steps)[..., None]
        r = (1 - j / steps)[..., None]
        average = (x - y) / jnp.maximum(t - r, 1 / steps)
        diagonal_obs = jnp.repeat(obs, 2, axis=0)
        diagonal_x = path[diagonal, rows].reshape(-1, self.config["action_dim"])
        diagonal_t = (1 - diagonal / steps).reshape(-1, 1)
        diagonal_velocity = self.teacher_field(
            diagonal_obs, diagonal_x, diagonal_t, control_eta, uncertainty_scale
        ).reshape(count, 2, -1)
        average = average.at[:, 1:3].set(diagonal_velocity)
        data = (
            jnp.repeat(obs, 8, axis=0),
            x.reshape(-1, x.shape[-1]),
            r.reshape(-1, 1),
            t.reshape(-1, 1),
            average.reshape(-1, x.shape[-1]),
        )
        return jax.tree_util.tree_map(jax.lax.stop_gradient, data), noise, path

    def jacobian_loss(
        self, observations, noise, grad_params, rng, control_eta=None,
        uncertainty_scale=None
    ):
        count = min(self.config["jacobian_batch_size"], len(noise))
        obs = observations[:count]
        index_key, direction_key = jax.random.split(rng)
        steps = self.config["teacher_steps"]
        indices = jax.random.randint(index_key, (count,), 0, steps).at[0].set(0)
        path = self.teacher_path(
            obs, noise[:count], control_eta=control_eta,
            uncertainty_scale=uncertainty_scale
        )
        x = jax.lax.stop_gradient(path[indices, jnp.arange(count)])
        t = (1 - indices / steps)[:, None]
        direction = jax.random.rademacher(direction_key, x.shape, dtype=x.dtype)
        epsilon = self.config["jacobian_fd_eps"]
        student = directional_difference(
            lambda z: self.endpoint(
                obs, z, jnp.zeros_like(t), t, params=grad_params
            ),
            x,
            direction,
            epsilon,
        )
        teacher = directional_difference(
            lambda z: rk2_endpoint_from_grid(
                lambda state, time: self.teacher_field(
                    obs, state, time, control_eta, uncertainty_scale
                ),
                z,
                indices,
                steps,
            ),
            x,
            direction,
            epsilon,
        )
        return jnp.mean((student - jax.lax.stop_gradient(teacher)) ** 2)

    def guided_actor_loss(
        self, batch, grad_params, rng, jacobian_active, control_eta=None,
        uncertainty_scale=None
    ):
        transport_key, jacobian_key, bound_key = jax.random.split(rng, 3)
        (obs, x, r, t, target), noise, path = self.transport_batch(
            batch["observations"], transport_key, control_eta, uncertainty_scale
        )
        prediction = x - self.g(obs, x, r, t, params=grad_params)
        transport = jnp.mean((prediction - target) ** 2)
        jacobian = (
            self.jacobian_loss(
                batch["observations"], noise, grad_params, jacobian_key, control_eta,
                uncertainty_scale
            )
            if jacobian_active
            else jnp.asarray(0.0)
        )
        endpoint_noise = self.sample_noise(bound_key, batch["actions"].shape)
        one = jnp.ones((len(endpoint_noise), 1))
        raw = self.g(
            batch["observations"],
            endpoint_noise,
            jnp.zeros_like(one),
            one,
            params=grad_params,
        )
        bound = (jax.nn.relu(raw - 1) + jax.nn.relu(-1 - raw)).mean()
        q = self.network.select("critic")(
            batch["observations"], actions=jnp.clip(raw, -1, 1)
        ).mean()
        direct_q_active = (
            self.config["variant"] == "direct_q_control"
            and self.config["direct_q_coef"] != 0
        )
        q_loss = (
            -q * self.config["direct_q_coef"]
            if direct_q_active
            else jnp.asarray(0.0)
        )
        loss = (
            self.config["transport_coef"] * transport
            + self.config["jacobian_coef"]
            * self.config["jacobian_interval"]
            * jacobian
            + self.config["bound_loss_weight"] * bound
            + q_loss
        )
        teacher_q = self.target_reward(
            batch["observations"][: len(noise)], path[-1]
        ).mean()
        diagnostic_t = jnp.ones((len(noise), 1))
        _, _, control_info = self.control_terms(
            batch["observations"][: len(noise)], noise, diagnostic_t,
            uncertainty_scale
        )
        eta = (
            jnp.asarray(self.config["control_eta"])
            if control_eta is None
            else control_eta
        )
        info = {
            "actor/loss": loss,
            "actor/transport": transport,
            "actor/jacobian": jacobian,
            "actor/bound_loss": bound,
            "actor/q_loss": q_loss,
            "actor/q": q,
            "teacher/q": teacher_q,
            "teacher/endpoint_variance": jnp.var(path[-1], axis=0).mean(),
            "control/eta": eta,
            "control/gate": jnp.mean(control_info["gate"]),
            "control/adjoint_norm": jnp.mean(control_info["raw_norm"]),
            "control/applied_adjoint_norm": jnp.mean(control_info["clipped_norm"]),
            "control/adjoint_clip_fraction": jnp.mean(control_info["clipped_fraction"]),
            "control/relative_critic_disagreement": jnp.mean(
                control_info["relative_disagreement"]
            ),
        }
        info.update(
            {f"actor/{key}": value for key, value in clipping_metrics(raw).items()}
        )
        return loss, info

    @partial(jax.jit, static_argnames=("jacobian_active",))
    def _guided_update(self, batch, current_step, jacobian_active):
        new_rng, root = jax.random.split(self.rng)
        actor_rng, critic_rng = jax.random.split(root)

        def loss_fn(params):
            critic, critic_info = self.critic_loss(batch, params, critic_rng)
            control_eta = self.scheduled_control_eta(current_step)
            uncertainty_scale = self.scheduled_uncertainty_scale(current_step)
            actor, actor_info = self.guided_actor_loss(
                batch, params, actor_rng, jacobian_active, control_eta,
                uncertainty_scale
            )
            info = {f"critic/{key}": value for key, value in critic_info.items()}
            info.update(actor_info)
            info["total_loss"] = actor + critic
            return actor + critic, info

        network, info = self.network.apply_loss_fn(loss_fn)
        self.target_update(network, "critic")
        return self.replace(network=network, rng=new_rng), info

    @jax.jit
    def _finish_update(self, updated):
        count = self.completed_updates + 1
        actor = updated.network.params["modules_actor_bc_flow"]
        pre = jax.tree_util.tree_map(
            lambda new, old: jnp.where(
                count <= self.config["behavior_warmup_updates"], new, old
            ),
            actor,
            self.pre_actor_params,
        )
        tau = self.config["target_actor_tau"]
        ema = jax.tree_util.tree_map(
            lambda new, old: tau * new + (1 - tau) * old,
            actor,
            self.target_actor_params,
        )
        return updated.replace(
            pre_actor_params=pre,
            target_actor_params=ema,
            completed_updates=count,
        )

    def update(self, batch, current_step=0):
        """Select the internal phase without changing the upstream host loop."""
        if current_step < self.config["behavior_warmup_updates"]:
            updated, info = MeanFlowQL_Agent.update(self, batch, current_step)
        else:
            active = (
                current_step - self.config["behavior_warmup_updates"]
            ) % self.config["jacobian_interval"] == 0
            updated, info = self._guided_update(
                batch, current_step, jacobian_active=active
            )
        return self._finish_update(updated), info


def get_config():
    config = base_config()
    config.update(
        dict(
            agent_name="am_meanflow_note",
            variant="note_adjoint",
            behavior_warmup_updates=500000,
            control_eta=0.1,
            control_eta_ramp_updates=0,
            control_adjoint_clip=0.0,
            control_uncertainty_scale=0.0,
            control_uncertainty_end_update=-1,
            teacher_steps=8,
            teacher_solver="midpoint_rk2",
            teacher_batch_size=32,
            transport_coef=1.0,
            jacobian_coef=0.01,
            jacobian_interval=100,
            jacobian_batch_size=4,
            jacobian_fd_eps=1e-3,
            target_actor_tau=0.005,
            direct_q_coef=1.0,
            alpha=10000.0,
            time_steps=50,
            consistency_alpha=0.0,
        )
    )
    return config
