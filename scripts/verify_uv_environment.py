#!/usr/bin/env python3
"""Verify that an AM-MF uv environment can import its selected stack."""

from __future__ import annotations

import argparse
import importlib
import os
from importlib import metadata


CORE_MODULES = (
    "absl",
    "chex",
    "distrax",
    "dm_control",
    "flax",
    "gymnasium",
    "h5py",
    "jax",
    "ml_collections",
    "mujoco",
    "numpy",
    "ogbench",
    "optax",
    "PIL",
    "scipy",
    "tqdm",
    "wandb",
)


def import_modules(names: tuple[str, ...]) -> None:
    for name in names:
        importlib.import_module(name)
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = "imported"
        print(f"[ok] {name}: {version}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--check-d4rl", action="store_true")
    parser.add_argument("--check-toy", action="store_true")
    args = parser.parse_args()

    import_modules(CORE_MODULES)

    import jax
    import jax.numpy as jnp

    value = jnp.sum(jnp.arange(4, dtype=jnp.float32))
    if float(value) != 6.0:
        raise RuntimeError(f"Unexpected JAX smoke-test result: {value}")

    devices = jax.devices()
    print("[ok] JAX devices:", ", ".join(str(device) for device in devices))
    if args.require_gpu and not any(device.platform == "gpu" for device in devices):
        raise RuntimeError(
            "CUDA packages are installed, but JAX cannot see a GPU. Check the "
            "NVIDIA driver and remove conflicting CUDA paths from LD_LIBRARY_PATH."
        )

    if args.check_d4rl:
        os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
        import_modules(("gym", "mujoco_py", "d4rl"))

    if args.check_toy:
        import_modules(("matplotlib", "ot"))

    print("AM-MF uv environment verification passed.")


if __name__ == "__main__":
    main()

