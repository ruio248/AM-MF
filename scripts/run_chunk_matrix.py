"""Explicit matrix launcher: one serial lane per supplied GPU, no implicit jobs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def matrix_jobs(output_root, smoke=False):
    for task, method, horizon, seed in itertools.product(
        ("door-cloned-v1", "pen-cloned-v1"), ("b0", "n"), (1, 2, 5, 10), (1,) if smoke else (1, 2)
    ):
        output = Path(output_root) / task / method / f"H{horizon}" / f"seed{seed}"
        command = ["bash", str(ROOT / "scripts/chunk_env.sh"), str(ROOT / "main_chunked.py"),
                   "--config", str(ROOT / "configs/chunk" / f"{task}.yaml"),
                   "--method", method, "--chunk-size", str(horizon), "--seed", str(seed),
                   "--output-dir", str(output)]
        if smoke:
            command.append("--smoke")
        yield dict(task=task, method=method, horizon=horizon, seed=seed, output=str(output), command=command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpus", help="Explicit comma-separated physical GPU IDs")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    jobs = list(matrix_jobs(Path(args.output_root).resolve(), args.smoke))
    if args.dry_run:
        for job in jobs:
            print(shlex.join(job["command"]))
        print(f"# {len(jobs)} jobs; {'validation only' if args.smoke else 'formal 1M offline + 1M env steps'}")
        return
    if not args.gpus:
        parser.error("--gpus is required to launch; inspect --dry-run first")
    gpus = [gpu.strip() for gpu in args.gpus.split(",")]
    if any(not gpu.isdigit() for gpu in gpus) or len(set(gpus)) != len(gpus):
        parser.error("GPU IDs must be unique nonnegative integers")
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    launches = []

    def lane(gpu, lane_jobs):
        rows = []
        for job in lane_jobs:
            output = Path(job["output"])
            if (output / "COMPLETED.json").is_file():
                rows.append(dict(**job, gpu=gpu, status="skipped_complete"))
                continue
            if output.exists() and any(output.iterdir()):
                rows.append(dict(**job, gpu=gpu, status="blocked_nonempty"))
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            log_path = output.parent / f"{output.name}.stdout.log"
            started = time.time()
            print(json.dumps({"starting": str(output), "gpu": gpu}), flush=True)
            with log_path.open("a") as log:
                result = subprocess.run(job["command"], env=env, stdout=log, stderr=subprocess.STDOUT)
            row = dict(**job, gpu=gpu, status="completed" if result.returncode == 0 else "failed",
                       exit_code=result.returncode, seconds=time.time() - started, log=str(log_path))
            rows.append(row)
            print(json.dumps({"finished": str(output), "status": row["status"], "seconds": row["seconds"]}), flush=True)
        return rows

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(lane, gpu, jobs[index::len(gpus)]) for index, gpu in enumerate(gpus)]
        for future in futures:
            launches.extend(future.result())
    with (output_root / "matrix_status.json").open("w") as file:
        json.dump(launches, file, indent=2)
    if any(row["status"] not in ("completed", "skipped_complete") for row in launches):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
