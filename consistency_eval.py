"""Evaluate consistency of a saved Native MeanFlow/MeanFlowQL checkpoint."""

import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from absl import app, flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.consistency_eval import (
    ConsistencyThresholds,
    evaluate_agent_consistency,
    judge_consistency,
)
from utils.datasets import Dataset
from utils.flax_utils import restore_agent


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "run_dir",
    None,
    "Training run directory containing flags.json and params_STEP.pkl.",
)
flags.DEFINE_integer("restore_epoch", None, "Checkpoint step to restore.")
flags.DEFINE_integer(
    "validation_states", 1024, "Number of validation observations."
)
flags.DEFINE_integer(
    "noises_per_state", 4, "Fixed noise samples per validation state."
)
flags.DEFINE_integer("eval_seed", 20260824, "Deterministic probe seed.")
flags.DEFINE_string(
    "eval_nfes", "1,2,4,10", "Comma-separated rollout NFE values."
)
flags.DEFINE_integer(
    "trajectory_steps", 10, "Segments in the reference trajectory."
)
flags.DEFINE_integer(
    "jacobian_probe_pairs",
    128,
    "State-noise pairs used by the endpoint JVP check.",
)
flags.DEFINE_integer(
    "inference_batch_size",
    256,
    "Maximum state-noise pairs per actor forward pass.",
)
flags.DEFINE_bool(
    "use_target_actor",
    False,
    "Use the EMA target actor for native MeanFlow agents.",
)
flags.DEFINE_string(
    "probe_path",
    None,
    "Optional NPZ containing observations and noises for a shared probe.",
)
flags.DEFINE_string(
    "output_path", None, "Optional output JSON path; defaults inside run_dir."
)
flags.DEFINE_float(
    "max_endpoint_map_nmse",
    -1.0,
    "Optional endpoint-map NMSE judgement threshold; negative disables it.",
)
flags.DEFINE_float(
    "max_endpoint_jvp_nmse",
    -1.0,
    "Optional endpoint-JVP NMSE judgement threshold; negative disables it.",
)
flags.DEFINE_float(
    "max_split_consistency_nmse",
    -1.0,
    "Optional Native MeanFlow split NMSE threshold; negative disables it.",
)
flags.DEFINE_float(
    "max_k1_k_reference_mse",
    -1.0,
    "Optional K1-vs-largest-K MSE threshold; negative disables it.",
)

# MeanFlowQL.create reads these global training flags when rebuilding the
# optimizer tree.  Their values are replaced from the saved flags.json before
# checkpoint restoration.
flags.DEFINE_integer("offline_steps", 1_000_000, "Saved offline schedule.")
flags.DEFINE_integer("online_steps", 0, "Saved online schedule.")
flags.DEFINE_float("pretrain_factor", 0.0, "Saved pretraining ratio.")


def _optional_threshold(value):
    return None if value < 0.0 else float(value)


def _parse_nfes(value):
    try:
        nfes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("--eval_nfes must contain comma-separated integers.") from error
    if not nfes or any(nfe < 1 for nfe in nfes):
        raise ValueError("--eval_nfes must contain positive integers.")
    return nfes


def _load_metadata(run_dir):
    flags_path = run_dir / "flags.json"
    if not flags_path.is_file():
        raise FileNotFoundError(f"Missing training metadata: {flags_path}")
    metadata = json.loads(flags_path.read_text())
    if "agent" not in metadata or "env_name" not in metadata:
        raise ValueError(
            "flags.json must contain both 'agent' and 'env_name'."
        )
    return metadata


def _prepare_dataset(metadata):
    _, _, train_dataset, val_dataset = make_env_and_datasets(
        metadata["env_name"], frame_stack=metadata.get("frame_stack")
    )
    if not isinstance(train_dataset, Dataset):
        train_dataset = Dataset.create(**train_dataset)
    if val_dataset is not None and not isinstance(val_dataset, Dataset):
        val_dataset = Dataset.create(**val_dataset)

    if metadata.get("use_observation_normalization", True):
        train_dataset.compute_normalization_stats()
        train_dataset.enable_normalization(True)
        if val_dataset is not None:
            val_dataset.obs_mean = train_dataset.obs_mean
            val_dataset.obs_std = train_dataset.obs_std
            val_dataset.enable_normalization(True)
    return train_dataset, val_dataset or train_dataset


def _restore_from_run(run_dir, restore_epoch, metadata, train_dataset):
    config = ml_collections.ConfigDict(metadata["agent"])
    agent_name = str(config["agent_name"])
    if agent_name not in agents:
        raise ValueError(
            f"Unsupported saved agent {agent_name!r}; available: {sorted(agents)}"
        )

    FLAGS.offline_steps = int(metadata.get("offline_steps", FLAGS.offline_steps))
    FLAGS.online_steps = int(metadata.get("online_steps", FLAGS.online_steps))
    FLAGS.pretrain_factor = float(
        metadata.get("pretrain_factor", FLAGS.pretrain_factor)
    )
    example_batch = train_dataset.sample(1)
    agent = agents[agent_name].create(
        int(metadata.get("seed", 0)),
        example_batch["observations"],
        example_batch["actions"],
        config,
    )
    return restore_agent(agent, str(run_dir), restore_epoch)


def _build_probe(agent, dataset):
    if FLAGS.probe_path is not None:
        probe_path = Path(FLAGS.probe_path).expanduser().resolve()
        with np.load(probe_path) as probe:
            observations = np.asarray(probe["observations"], dtype=np.float32)
            noises = np.asarray(probe["noises"], dtype=np.float32)
        if observations.shape[0] != noises.shape[0]:
            raise ValueError(
                "Shared probe observations and noises have different lengths."
            )
        return jnp.asarray(observations), jnp.asarray(noises)

    if FLAGS.validation_states < 1 or FLAGS.noises_per_state < 1:
        raise ValueError(
            "--validation_states and --noises_per_state must be positive."
        )
    rng = np.random.default_rng(FLAGS.eval_seed)
    replace = dataset.size < FLAGS.validation_states
    indices = rng.choice(
        dataset.size, size=FLAGS.validation_states, replace=replace
    )
    batch = dataset.sample(FLAGS.validation_states, idxs=indices)
    observations = jnp.repeat(
        jnp.asarray(batch["observations"]), FLAGS.noises_per_state, axis=0
    )
    noises = agent.sample_noise(
        jax.random.PRNGKey(FLAGS.eval_seed),
        (observations.shape[0], int(agent.config["action_dim"])),
    )
    return observations, noises


def main(_):
    if FLAGS.run_dir is None or FLAGS.restore_epoch is None:
        raise ValueError("--run_dir and --restore_epoch are required.")
    run_dir = Path(FLAGS.run_dir).expanduser().resolve()
    metadata = _load_metadata(run_dir)
    train_dataset, validation_dataset = _prepare_dataset(metadata)
    agent = _restore_from_run(
        run_dir,
        int(FLAGS.restore_epoch),
        metadata,
        train_dataset,
    )
    observations, noises = _build_probe(agent, validation_dataset)
    metrics = evaluate_agent_consistency(
        agent,
        observations,
        noises,
        nfe_values=_parse_nfes(FLAGS.eval_nfes),
        trajectory_steps=FLAGS.trajectory_steps,
        jacobian_probe_pairs=FLAGS.jacobian_probe_pairs,
        inference_batch_size=FLAGS.inference_batch_size,
        use_target_actor=FLAGS.use_target_actor,
        rng=jax.random.fold_in(jax.random.PRNGKey(FLAGS.eval_seed), 913),
    )
    thresholds = ConsistencyThresholds(
        endpoint_map_nmse=_optional_threshold(
            FLAGS.max_endpoint_map_nmse
        ),
        endpoint_jvp_nmse=_optional_threshold(
            FLAGS.max_endpoint_jvp_nmse
        ),
        split_consistency_nmse=_optional_threshold(
            FLAGS.max_split_consistency_nmse
        ),
        k1_k_reference_mse=_optional_threshold(
            FLAGS.max_k1_k_reference_mse
        ),
    )
    result = {
        "run_dir": str(run_dir),
        "checkpoint": int(FLAGS.restore_epoch),
        "eval_seed": int(FLAGS.eval_seed),
        "probe": {
            "num_pairs": int(observations.shape[0]),
            "validation_states": int(FLAGS.validation_states),
            "noises_per_state": int(FLAGS.noises_per_state),
            "inference_batch_size": int(FLAGS.inference_batch_size),
            "shared_probe_path": FLAGS.probe_path,
        },
        "metrics": metrics,
        "judgement": judge_consistency(metrics, thresholds),
    }

    output_path = (
        Path(FLAGS.output_path).expanduser().resolve()
        if FLAGS.output_path is not None
        else run_dir
        / f"consistency_eval_step{FLAGS.restore_epoch}_seed{FLAGS.eval_seed}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    print(f"Saved consistency evaluation to {output_path}")


if __name__ == "__main__":
    app.run(main)
