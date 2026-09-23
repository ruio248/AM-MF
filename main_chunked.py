"""Fair primitive-step B0/N chunk experiments, independent of the legacy runner."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import yaml

from agents.chunked import create_agent, make_config
from utils.chunk_data import ChunkReplayBuffer, ObservationNormalizer
from utils.chunk_runtime import (
    ChunkCollector, atomic_json, checkpoint_contract, evaluate_chunked,
    evaluation_seeds, jsonable, restore_offline_checkpoint, save_checkpoint,
)


BASE_COMMIT = "31e4b77defc97f16b6cc6229c906790b16cbfc44"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar_metrics(metrics):
    output = {key: float(np.asarray(value)) for key, value in metrics.items()}
    if not all(np.isfinite(value) for value in output.values()):
        raise FloatingPointError("Nonfinite training metric")
    return output


def write_row(file, row):
    file.write(json.dumps(jsonable(row), sort_keys=True, allow_nan=False) + "\n")
    file.flush()


def load_settings(args):
    with Path(args.config).open() as file:
        settings = yaml.safe_load(file)
    if settings.get("protocol_version") != 1:
        raise ValueError("Unsupported configuration protocol version")
    if settings["env_name"] not in ("door-cloned-v1", "pen-cloned-v1"):
        raise ValueError("Chunk v1 task profiles are door-cloned-v1 and pen-cloned-v1")
    settings = copy.deepcopy(settings)
    training = settings["training"]
    if args.smoke:
        training.update(offline_steps=40, online_steps=20, eval_interval=40,
                        eval_episodes=2, log_interval=10, save_interval=40)
        settings["agent"].update(actor_hidden_dims=32, actor_depth=1, actor_num_heads=2,
                                 value_hidden_dims=[32, 32], batch_size=16, time_steps=8)
        settings["n_agent"].update(behavior_warmup_updates=4, teacher_steps=2,
                                   teacher_batch_size=4, jacobian_batch_size=2, jacobian_interval=4)
    for key in ("offline_steps", "online_steps", "eval_episodes", "eval_interval", "log_interval", "save_interval"):
        value = getattr(args, key, None)
        if value is not None:
            training[key] = value
    for key in ("offline_steps", "eval_episodes", "eval_interval", "log_interval", "save_interval"):
        if not isinstance(training[key], int) or training[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if training["online_steps"] < 0 or args.chunk_size < 1 or args.seed < 0:
        raise ValueError("Invalid step budget, horizon or seed")
    if args.critic_lr is not None:
        settings["agent"]["critic_lr"] = args.critic_lr
    if args.target_num_candidates is not None:
        settings["agent"]["target_num_candidates"] = args.target_num_candidates
    if args.num_candidates is not None:
        settings["agent"]["num_candidates"] = args.num_candidates
    if args.control_eta is not None:
        settings["n_agent"]["control_eta"] = args.control_eta
    return settings


def run(args):
    settings = load_settings(args)
    training = settings["training"]
    if args.phase == "online" and not args.restore:
        raise ValueError("--phase online requires an offline-boundary --restore checkpoint")
    if args.restore and args.phase != "online":
        raise ValueError("--restore is only supported with --phase online")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite nonempty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "requested_config.json", dict(settings=settings, arguments=vars(args)))
    started = time.monotonic()

    from envs.env_utils import make_env_and_datasets
    env, eval_env, dataset, _ = make_env_and_datasets(settings["env_name"])
    try:
        normalizer = ObservationNormalizer.fit(dataset["observations"])
        replay = ChunkReplayBuffer(dataset, training["buffer_size"], args.chunk_size,
                                   settings["agent"]["discount"], normalizer)
        if replay.pool_size == 0:
            raise ValueError("Offline data has no valid complete chunks")
        data_dir = Path(os.environ.get("D4RL_DATASET_DIR", str(Path.home() / ".d4rl/datasets")))
        data_file = data_dir / f"{settings['env_name']}.hdf5"
        identity = dict(path=str(data_file), sha256=sha256(data_file),
                        valid_transitions=len(dataset["observations"]))
        action_dim = int(np.prod(env.action_space.shape))
        overrides = dict(settings["agent"])
        if args.method == "n":
            overrides.update(settings["n_agent"])
        config = make_config(args.method, overrides, args.chunk_size, action_dim)
        # Initialization does not consume the training sampler's RNG stream.
        example = replay.batch_from_starts(replay.pool[:1])
        agent = create_agent(args.method, args.seed, example["observations"], example["actions"],
                             config, training["offline_steps"], training["online_steps"])
        sample_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 2001]))
        contract = checkpoint_contract(settings["env_name"], args.method, args.seed,
                                       agent.config, normalizer, identity)
        offline_updates = 0
        if args.restore:
            agent, normalizer, sample_rng = restore_offline_checkpoint(
                args.restore, agent, contract, training["offline_steps"]
            )
            replay.normalizer = normalizer
            offline_updates = training["offline_steps"]
        seeds = evaluation_seeds(args.seed, training["eval_episodes"])
        repository = Path(__file__).resolve().parent
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True)
        manifest = dict(
            protocol_version=1, base_commit=BASE_COMMIT, source_commit=commit, source_status=status,
            env_name=settings["env_name"], method=args.method, chunk_size=args.chunk_size,
            seed=args.seed, smoke=args.smoke, phase=args.phase, restore=args.restore,
            settings=settings, effective_agent_config=jsonable(agent.config), contract=contract,
            evaluation_seeds=seeds, replay_initial=replay.metrics(),
            normalization="valid_offline_frozen", online_utd=1,
            jax_version=jax.__version__, devices=[str(device) for device in jax.devices()],
        )
        atomic_json(output / "manifest.json", manifest)
        np.savez(output / "normalization.npz", **normalizer.state_dict())
        print(json.dumps({"output": str(output), "task": settings["env_name"],
                          "method": args.method, "H": args.chunk_size,
                          "valid_windows": replay.pool_size, "devices": manifest["devices"]}), flush=True)
        collector = ChunkCollector(env, normalizer, args.chunk_size, args.seed)
        online_updates, guided_updates = 0, 0
        evaluation_rows = []
        with (output / "train.jsonl").open("w") as train_log, (output / "eval.jsonl").open("w") as eval_log:
            def evaluate(phase, cached=None):
                result = cached if cached is not None else evaluate_chunked(
                    agent, eval_env, normalizer, args.chunk_size, seeds
                )
                row = dict(phase=phase, offline_updates=offline_updates,
                           online_env_steps=collector.env_steps, learner_updates=offline_updates + online_updates,
                           **result)
                write_row(eval_log, row)
                evaluation_rows.append(row)
                print(json.dumps({key: value for key, value in row.items() if key != "episodes"}), flush=True)
                return result

            def log_training(info, batch, phase, completed):
                row = dict(phase=phase, offline_updates=offline_updates,
                           online_env_steps=collector.env_steps, learner_updates=offline_updates + online_updates,
                           elapsed_seconds=time.monotonic() - started, **scalar_metrics(info), **replay.metrics())
                if args.method == "n" and completed > agent.config["behavior_warmup_updates"]:
                    row.update(scalar_metrics(agent.adjoint_metrics(batch["observations"])))
                row["effective/critic_lr"] = float(agent.config["critic_lr"])
                row["effective/actor_lr"] = float(agent.config["actor_lr_schedule"](completed - 1))
                write_row(train_log, row)

            offline_endpoint = None
            if args.phase != "online":
                evaluate("offline")
                for update_index in range(training["offline_steps"]):
                    batch = replay.sample(config["batch_size"], sample_rng)
                    agent, info = agent.update(batch, current_step=update_index)
                    offline_updates += 1
                    guided_updates += int(args.method == "n" and update_index >= config["behavior_warmup_updates"])
                    if offline_updates % training["log_interval"] == 0 or offline_updates == training["offline_steps"]:
                        log_training(info, batch, "offline", offline_updates)
                    if offline_updates % training["eval_interval"] == 0 or offline_updates == training["offline_steps"]:
                        offline_endpoint = evaluate("offline")
                    if offline_updates % training["save_interval"] == 0 and offline_updates < training["offline_steps"]:
                        save_checkpoint(output / "checkpoints" / f"offline_{offline_updates}.pkl", agent,
                                        normalizer, sample_rng, contract, offline_updates, 0, "offline_partial")
                save_checkpoint(output / "checkpoints/offline_complete.pkl", agent, normalizer,
                                sample_rng, contract, offline_updates, 0, "offline_complete")

            if args.phase != "offline":
                evaluate("online", cached=offline_endpoint)
                for _ in range(training["online_steps"]):
                    transition, _ = collector.step(agent)
                    replay.add_transition(transition)
                    batch = replay.sample(config["batch_size"], sample_rng)
                    update_index = offline_updates + online_updates
                    agent, info = agent.update(batch, current_step=update_index)
                    online_updates += 1
                    guided_updates += int(args.method == "n" and update_index >= config["behavior_warmup_updates"])
                    if online_updates % training["log_interval"] == 0 or online_updates == training["online_steps"]:
                        log_training(info, batch, "online", offline_updates + online_updates)
                    if online_updates % training["eval_interval"] == 0 or online_updates == training["online_steps"]:
                        evaluate("online")
                    if online_updates % training["save_interval"] == 0 and online_updates < training["online_steps"]:
                        save_checkpoint(output / "checkpoints" / f"online_{online_updates}.pkl", agent,
                                        normalizer, sample_rng, contract, offline_updates, online_updates, "online_partial")
                save_checkpoint(output / "checkpoints/online_complete.pkl", agent, normalizer,
                                sample_rng, contract, offline_updates, online_updates, "online_complete")
            if online_updates != collector.env_steps:
                raise AssertionError("UTD=1 budget violation")
            if args.smoke and args.method == "n" and guided_updates == 0:
                raise AssertionError("Smoke run did not exercise adjoint-guided updates")
            completion = dict(offline_updates=offline_updates, online_env_steps=collector.env_steps,
                              online_updates=online_updates, learner_updates=offline_updates + online_updates,
                              guided_updates_this_process=guided_updates,
                              policy_calls=collector.policy_calls, elapsed_seconds=time.monotonic() - started,
                              last_evaluation=evaluation_rows[-1], replay_final=replay.metrics())
            atomic_json(output / "COMPLETED.json", completion)
            print(json.dumps({"completed": str(output), **{k: v for k, v in completion.items()
                              if k not in ("last_evaluation", "replay_final")}}), flush=True)
    finally:
        env.close()
        eval_env.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=("b0", "n"), required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase", choices=("both", "offline", "online"), default="both")
    parser.add_argument("--restore")
    parser.add_argument("--smoke", action="store_true", help="Explicitly use small networks and budgets for validation")
    for name in ("offline-steps", "online-steps", "eval-episodes", "eval-interval", "log-interval", "save-interval",
                 "num-candidates", "target-num-candidates"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--critic-lr", type=float)
    parser.add_argument("--control-eta", type=float)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
