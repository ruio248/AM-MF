from collections import defaultdict

import jax
import numpy as np
from tqdm import trange


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=1,
    train_dataset=None,  # Add parameters for the training dataset to obtain normalization statistical information.
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        train_dataset: Training dataset with normalization statistics.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    # 检查是否需要进行归一化
    normalize_obs = False
    if train_dataset is not None and hasattr(train_dataset, 'normalize_obs') and train_dataset.normalize_obs:
        normalize_obs = True
        obs_mean = train_dataset.obs_mean
        obs_std = train_dataset.obs_std
        print("Using observation normalization during evaluation")

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset()
        done = False
        step = 0
        render = []
        while not done:
            obs_batch = (
                np.expand_dims(observation, axis=0)
                if observation.ndim == 1
                else observation
            )

            if normalize_obs:
                obs_batch = (obs_batch - obs_mean) / obs_std

            chunk_size = int(config.get('chunk_size', 1)) if config is not None else 1
            action_dim = int(np.prod(env.action_space.shape))
            action = np.asarray(
                actor_fn(observations=obs_batch, temperature=eval_temperature)
            )
            if action.ndim > 1 and action.shape[0] == 1:
                action = action[0]
            action = action.reshape(-1)
            expected_action_dim = chunk_size * action_dim
            if action.size != expected_action_dim:
                raise ValueError(
                    f'Policy returned {action.size} values, expected '
                    f'{expected_action_dim} for chunk_size={chunk_size}'
                )
            action_chunk = np.clip(
                action.reshape(chunk_size, action_dim), -1, 1
            )

            if i >= num_eval_episodes - 1:
                print(f"Step {i}: The agent generated action chunk is {action_chunk}")

            for chunk_action in action_chunk:
                next_observation, reward, terminated, truncated, info = env.step(
                    np.asarray(chunk_action).copy()
                )
                done = bool(terminated or truncated)
                step += 1

                if should_render and (step % video_frame_skip == 0 or done):
                    frame = env.render().copy()
                    render.append(frame)

                transition = dict(
                    observation=observation,
                    next_observation=next_observation,
                    action=np.asarray(chunk_action),
                    reward=reward,
                    done=done,
                    info=info,
                )
                add_to(traj, transition)
                observation = next_observation
                if done:
                    break

        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders
