from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Dataset class."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None  # Number of frames to stack; set outside the class.
        self.p_aug = None  # Image augmentation probability; set outside the class.
        self.return_next_actions = False  # Whether to additionally return next actions; set outside the class.
        
        # Normalization statistics
        self.obs_mean = None
        self.obs_std = None
        self.normalize_obs = False

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
    
    def compute_normalization_stats(self):
        """Compute mean and standard deviation of observations for normalization."""
        observations = self['observations']
        self.obs_mean = np.mean(observations, axis=0, keepdims=True)
        self.obs_std = np.std(observations, axis=0, keepdims=True)
        # Avoid division by zero
        self.obs_std = np.where(self.obs_std < 1e-6, 1.0, self.obs_std)
        
        # Calculate max and min values for each feature dimension
        self.obs_max = np.max(observations, axis=0, keepdims=True)
        self.obs_min = np.min(observations, axis=0, keepdims=True)
        
        print(f"Observation normalization stats computed. Shape: {observations.shape}")
        print(f"Mean shape: {self.obs_mean.shape}, Std shape: {self.obs_std.shape}")
        print(f"Mean range: [{np.min(self.obs_mean)}, {np.max(self.obs_mean)}]")
        print(f"Std range: [{np.min(self.obs_std)}, {np.max(self.obs_std)}]")
        
        # Print max and min values for each feature dimension
        print(f"Max values for each feature dimension: {self.obs_max.flatten()}")
        print(f"Min values for each feature dimension: {self.obs_min.flatten()}")
        # Optionally, print the range for each dimension
        print(f"Range for each feature dimension: {(self.obs_max - self.obs_min).flatten()}")
        
        return self.obs_mean, self.obs_std
    
    def enable_normalization(self, enable=True):
        """Enable or disable observation normalization."""
        if enable and (self.obs_mean is None or self.obs_std is None):
            self.compute_normalization_stats()
        self.normalize_obs = enable
        print(f"Observation normalization {'enabled' if enable else 'disabled'}")
    
    def normalize_observations(self, observations):
        """Normalize observations using precomputed statistics."""
        if self.obs_mean is None or self.obs_std is None:
            raise ValueError("Normalization statistics not computed. Call compute_normalization_stats() first.")
        return (observations - self.obs_mean) / self.obs_std
    
    def denormalize_observations(self, normalized_observations):
        """Denormalize observations using precomputed statistics."""
        if self.obs_mean is None or self.obs_std is None:
            raise ValueError("Normalization statistics not computed. Call compute_normalization_stats() first.")
        return normalized_observations * self.obs_std + self.obs_mean

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        if self.frame_stack is not None:
            # Stack frames.
            initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
            obs = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]].
            next_obs = []  # Will be [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]].
            for i in reversed(range(self.frame_stack)):
                # Use the initial state if the index is out of bounds.
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
                if i != self.frame_stack - 1:
                    next_obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
            next_obs.append(jax.tree_util.tree_map(lambda arr: arr[idxs], self['next_observations']))

            batch['observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *obs)
            batch['next_observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *next_obs)
        
        # Apply normalization if enabled
        if self.normalize_obs:
            batch['observations'] = self.normalize_observations(batch['observations'])
            batch['next_observations'] = self.normalize_observations(batch['next_observations'])
            
        if self.p_aug is not None:
            # Apply random-crop image augmentation.
            if np.random.rand() < self.p_aug:
                self.augment(batch, ['observations', 'next_observations'])
        return batch

    def to_chunked(self, chunk_size: int, discount: float):
        """Convert one-step transitions into fixed-length action chunks.

        A chunk is formed from ``chunk_size`` consecutive actions that do not
        cross an episode boundary before the final action. The action vector
        is flattened, rewards are discounted within the chunk, and the next
        observation is taken after the final action. For ``chunk_size=1``
        this preserves the original transition semantics exactly while adding
        metadata used by the online collector.
        """
        chunk_size = int(chunk_size)
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if self.frame_stack is not None:
            raise ValueError("Chunked sampling is incompatible with frame stacking")

        required = (
            "observations",
            "actions",
            "rewards",
            "masks",
            "terminals",
            "next_observations",
        )
        missing = [key for key in required if key not in self]
        if missing:
            raise KeyError(f"Cannot build chunks; missing fields: {missing}")

        data = {key: np.asarray(value) for key, value in self.items()}
        size = len(data["observations"])
        if size < chunk_size:
            raise ValueError(
                f"Dataset has {size} transitions, smaller than chunk_size={chunk_size}"
            )

        starts = np.arange(size - chunk_size + 1, dtype=np.int64)
        if chunk_size > 1:
            terminals = data["terminals"] > 0
            prefix = np.concatenate(
                [np.zeros(1, dtype=np.int64), np.cumsum(terminals, dtype=np.int64)]
            )
            # A terminal at the final action is allowed; a terminal in the
            # preceding H-1 actions would make the chunk cross an episode.
            valid = (prefix[starts + chunk_size - 1] - prefix[starts]) == 0
            starts = starts[valid]
        if len(starts) == 0:
            raise ValueError(f"No valid chunks found for chunk_size={chunk_size}")

        offsets = np.arange(chunk_size, dtype=np.int64)[None, :]
        indices = starts[:, None] + offsets
        end_indices = starts + chunk_size - 1
        reward_weights = np.power(
            np.float32(discount), np.arange(chunk_size, dtype=np.float32)
        )

        chunk_data = {
            "observations": data["observations"][starts],
            "actions": data["actions"][indices].reshape(len(starts), -1),
            "rewards": np.sum(
                data["rewards"][indices] * reward_weights[None, :], axis=1
            ).astype(np.float32),
            "masks": np.prod(data["masks"][indices], axis=1).astype(np.float32),
            "terminals": data["terminals"][end_indices].astype(np.float32),
            "next_observations": data["next_observations"][end_indices],
            "executed_steps": np.full(
                (len(starts), 1), chunk_size, dtype=np.float32
            ),
            "valid_action_mask": np.ones(
                (len(starts), chunk_size), dtype=np.float32
            ),
        }
        return Dataset.create(**chunk_data)

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            # WARNING: This is incorrect at the end of the trajectory. Use with caution.
            result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]
        return result

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0
