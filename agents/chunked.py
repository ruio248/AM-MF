"""Shared chunk-Q adapters; the frozen historical agents remain untouched."""

import copy
from functools import partial

import flax
from flax.core import freeze, unfreeze
import jax
import jax.numpy as jnp
import optax
from absl import flags
from absl.testing import flagsaver

from agents.meanflowql import MeanFlowQL_Agent, get_config as b0_config
from agents.am_meanflow_note import AMMeanFlowNoteAgent, get_config as n_config


def make_config(method, overrides, chunk_size, env_action_dim):
    if method not in ("b0", "n"):
        raise ValueError("method must be b0 or n")
    config = (b0_config if method == "b0" else n_config)()
    config.update(critic_lr=3e-4, target_num_candidates=5)
    for key, value in overrides.items():
        if key not in config:
            raise ValueError(f"Unknown {method} agent setting: {key}")
        config[key] = tuple(value) if key == "value_hidden_dims" else value
    config.update(chunk_size=int(chunk_size), env_action_dim=int(env_action_dim))
    if chunk_size < 1 or env_action_dim < 1:
        raise ValueError("chunk_size and env_action_dim must be positive")
    if config.encoder is not None:
        raise ValueError("Chunk v1 supports vector observations only")
    for key in ("num_candidates", "target_num_candidates", "time_steps", "batch_size"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config.critic_lr <= 0:
        raise ValueError("critic_lr must be positive")
    return config


def create_agent(method, seed, observations, actions, config, offline_steps, online_steps):
    """Scope legacy absl budget flags to construction, without importing its runner."""
    for name, default in (("offline_steps", 1000000), ("online_steps", 1000000)):
        if name not in flags.FLAGS:
            flags.DEFINE_integer(name, default, "Legacy schedule construction budget")
    if "pretrain_factor" not in flags.FLAGS:
        flags.DEFINE_float("pretrain_factor", 0.0, "Legacy pretrain budget")
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["chunked"])
    cls = ChunkB0Agent if method == "b0" else ChunkNAgent
    with flagsaver.flagsaver(offline_steps=offline_steps, online_steps=online_steps, pretrain_factor=0.0):
        return cls.create(seed, observations, actions, copy.deepcopy(config))


class ChunkAgentMixin:
    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        expected = config["chunk_size"] * config["env_action_dim"]
        if ex_actions.ndim != 2 or ex_actions.shape[-1] != expected:
            raise ValueError(f"Expected flat chunk actions [B,{expected}], got {ex_actions.shape}")
        agent = super().create(seed, ex_observations, ex_actions, config)
        # Both upstream factories create their own optimizer. Replace it before
        # the first update so B0 and N obey the same explicit critic setting.
        effective = dict(agent.config)
        critic_lr = float(config["critic_lr"])
        effective["critic_lr_schedule"] = lambda step: jnp.asarray(critic_lr)
        params = unfreeze(agent.network.params)
        labels = {
            name: jax.tree_util.tree_map(
                lambda _: "critic" if name in ("modules_critic", "modules_target_critic") else "actor",
                subtree,
            )
            for name, subtree in params.items()
        }
        tx = optax.multi_transform(
            {
                "critic": optax.chain(optax.clip_by_global_norm(1.0), optax.adam(critic_lr)),
                "actor": optax.chain(
                    optax.clip_by_global_norm(1.0), optax.adam(effective["actor_lr_schedule"])
                ),
            },
            labels,
        )
        return agent.replace(
            network=agent.network.replace(tx=tx, opt_state=tx.init(params)),
            config=freeze(effective),
        )

    def critic_loss(self, batch, grad_params, rng):
        _, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(
            batch["next_observations"], seed=sample_rng,
            num_candidates=int(self.config["target_num_candidates"]),
        )
        next_qs = self.network.select("target_critic")(
            batch["next_observations"], actions=next_actions
        )
        aggregation = self.config["q_agg"]
        next_q = (next_qs.min(axis=0) if aggregation == "min" else
                  next_qs.max(axis=0) if aggregation == "max" else next_qs.mean(axis=0))
        discount = self.config["discount"] ** self.config["chunk_size"]
        target = jax.lax.stop_gradient(batch["rewards"] + discount * batch["masks"] * next_q)
        q = self.network.select("critic")(
            batch["observations"], actions=batch["actions"], params=grad_params
        )
        loss = jnp.square(q - target).mean()
        return loss, dict(critic_loss=loss, q_mean=q.mean(), q_max=q.max(), q_min=q.min(),
                          target_q_mean=target.mean(), bootstrap_discount=jnp.asarray(discount))

    @partial(jax.jit, static_argnames=("num_candidates",))
    def sample_actions(self, observations, temperature=1, seed=None, num_candidates=None):
        # Temperature is retained for compatibility with the upstream interface;
        # the noise scale is explicitly configured by sigma, as in the parent.
        del temperature
        if observations.ndim != 2:
            raise ValueError("Expected vector observations [B, observation_dim]")
        count = self.config["num_candidates"] if num_candidates is None else num_candidates
        if not isinstance(count, int) or count < 1:
            raise ValueError("num_candidates must be a positive static Python integer")
        mode = self.config["action_mode"]
        if mode not in ("normal", "best", "mean"):
            raise ValueError(f"Unsupported action mode {mode}")
        batch_size = observations.shape[0]
        action_seed, _ = jax.random.split(seed)
        if mode == "normal":
            noises = self.sample_noise(action_seed, (batch_size, self.config["action_dim"]))
            return jnp.clip(self.network.select("actor_bc_flow")(
                observations, noises, jnp.ones((batch_size, 1))
            ), -1, 1)
        keys = jax.random.split(action_seed, count)
        noises = jax.vmap(lambda key: self.sample_noise(
            key, (batch_size, self.config["action_dim"])
        ))(keys)
        flat_obs = jnp.tile(observations[None], (count, 1, 1)).reshape(-1, observations.shape[-1])
        actions = self.network.select("actor_bc_flow")(
            flat_obs, noises.reshape(-1, self.config["action_dim"]),
            jnp.ones((count * batch_size, 1)),
        ).reshape(count, batch_size, self.config["action_dim"])
        if mode == "mean":
            return jnp.clip(actions.mean(axis=0), -1, 1)
        actions = jnp.clip(actions, -1, 1)
        qs = self.network.select("target_critic")(
            flat_obs, actions=actions.reshape(-1, self.config["action_dim"])
        )
        # Value has [ensemble, batch] layout, even when batch happens to be 2.
        scores = qs.mean(axis=0).reshape(count, batch_size)
        return actions[jnp.argmax(scores, axis=0), jnp.arange(batch_size)]


class ChunkB0Agent(ChunkAgentMixin, MeanFlowQL_Agent):
    """MeanFlowQL with a whole-chunk actor and H-step chunk critic."""


class ChunkNAgent(ChunkAgentMixin, AMMeanFlowNoteAgent):
    """The original global interval-control algorithm in H*d action space."""

    @jax.jit
    def adjoint_metrics(self, observations):
        count = min(len(observations), self.config["teacher_batch_size"])
        noise = self.sample_noise(jax.random.fold_in(self.rng, 731),
                                  (count, self.config["action_dim"]))
        _, adjoint = self.adjoint(observations[:count], noise, jnp.full((count, 1), 0.5))
        chunks = adjoint.reshape(count, self.config["chunk_size"], self.config["env_action_dim"])
        metrics = {
            "adjoint/norm": jnp.linalg.norm(adjoint, axis=-1).mean(),
            "adjoint/coordinate_rms": jnp.sqrt(jnp.mean(adjoint ** 2)),
            "adjoint/control_norm": self.config["control_eta"] * jnp.linalg.norm(adjoint, axis=-1).mean(),
        }
        for position in range(self.config["chunk_size"]):
            metrics[f"adjoint/position_{position}_norm"] = jnp.linalg.norm(chunks[:, position], axis=-1).mean()
        return metrics
