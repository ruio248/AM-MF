"""Endpoint-only chunk comparison, primitive-step AUC and paired N-minus-B0 gaps."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics


METRIC = "evaluation/episode.normalized_return"


def auc(points):
    points = sorted(points)
    if len({step for step, _ in points}) != len(points):
        raise ValueError("Duplicate online evaluation steps")
    return sum((right[0] - left[0]) * (left[1] + right[1]) / 2
               for left, right in zip(points, points[1:]))


def read_run(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    completion = json.loads((directory / "COMPLETED.json").read_text())
    evaluations = [json.loads(line) for line in (directory / "eval.jsonl").read_text().splitlines() if line]
    budget = manifest["settings"]["training"]
    expected_offline, expected_online = budget["offline_steps"], budget["online_steps"]
    if completion["offline_updates"] != expected_offline:
        raise ValueError("Offline endpoint incomplete")
    if completion["online_env_steps"] != expected_online or completion["online_updates"] != expected_online:
        raise ValueError("Online endpoint incomplete or UTD != 1")
    offline = [row for row in evaluations if row["phase"] == "offline" and row["offline_updates"] == expected_offline]
    online = sorted((row for row in evaluations if row["phase"] == "online"), key=lambda row: row["online_env_steps"])
    if not online or online[0]["online_env_steps"] != 0 or online[-1]["online_env_steps"] != expected_online:
        raise ValueError("Missing online origin or exact endpoint")
    points = [(row["online_env_steps"], row[METRIC]) for row in online]
    if not all(math.isfinite(value) for _, value in points):
        raise ValueError("Nonfinite evaluation values")
    start, end = online[0][METRIC], online[-1][METRIC]
    offline_end = offline[-1][METRIC] if offline else start
    area = auc(points)
    return dict(
        path=str(directory), env_name=manifest["env_name"], method=manifest["method"],
        chunk_size=manifest["chunk_size"], seed=manifest["seed"], smoke=manifest["smoke"],
        source_commit=manifest["source_commit"], offline_steps=expected_offline, online_steps=expected_online,
        norm=manifest["contract"]["normalization_fingerprint"],
        settings_signature=json.dumps(manifest["settings"], sort_keys=True),
        offline_endpoint=offline_end, online_start=start, online_endpoint=end,
        online_auc=area, online_auc_mean=area / expected_online if expected_online else None,
        curve=points,
    )


def mean_std(values):
    return dict(mean=statistics.mean(values), std=statistics.stdev(values) if len(values) > 1 else None, n=len(values))


def aggregate(runs):
    groups = defaultdict(list)
    for run in runs:
        key = (run["env_name"], run["method"], run["chunk_size"])
        groups[key].append(run)
    summaries = []
    for (task, method, horizon), group in sorted(groups.items()):
        if len({run["seed"] for run in group}) != len(group):
            raise ValueError(f"Duplicate seed for {task}/{method}/H{horizon}; select one matrix root")
        entry = dict(env_name=task, method=method, chunk_size=horizon, seeds=sorted(run["seed"] for run in group))
        for field in ("offline_endpoint", "online_start", "online_endpoint", "online_auc", "online_auc_mean"):
            values = [run[field] for run in group if run[field] is not None]
            entry[field] = mean_std(values) if values else None
        summaries.append(entry)
    paired = []
    lookup = {(run["env_name"], run["chunk_size"], run["seed"], run["method"]): run for run in runs}
    for task in sorted({run["env_name"] for run in runs}):
        for horizon in sorted({run["chunk_size"] for run in runs if run["env_name"] == task}):
            seeds = sorted(seed for env, h, seed, method in lookup
                           if env == task and h == horizon and method == "n" and (env, h, seed, "b0") in lookup)
            if not seeds:
                continue
            row = dict(env_name=task, chunk_size=horizon, paired_seeds=seeds)
            for field in ("offline_endpoint", "online_endpoint", "online_auc_mean"):
                deltas, interactions = [], []
                for seed in seeds:
                    n = lookup[task, horizon, seed, "n"][field]
                    b0 = lookup[task, horizon, seed, "b0"][field]
                    if n is None or b0 is None:
                        continue
                    delta = n - b0
                    deltas.append(delta)
                    if (task, 1, seed, "n") in lookup and (task, 1, seed, "b0") in lookup:
                        reference = lookup[task, 1, seed, "n"][field] - lookup[task, 1, seed, "b0"][field]
                        interactions.append(delta - reference)
                row[f"delta_{field}"] = mean_std(deltas) if deltas else None
                row[f"delta_minus_H1_{field}"] = mean_std(interactions) if interactions else None
            paired.append(row)
    return summaries, paired


def format_stat(stat):
    if stat is None:
        return "—"
    return f"{stat['mean']:.3f}" + (f" ± {stat['std']:.3f}" if stat["std"] is not None else " (1 seed)")


def plot_curves(runs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    for task in sorted({run["env_name"] for run in runs}):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        for axis, horizon in zip(axes.flat, (1, 2, 5, 10)):
            for method, color in (("b0", "#3366aa"), ("n", "#d06b20")):
                selected = [run for run in runs if run["env_name"] == task and run["chunk_size"] == horizon and run["method"] == method]
                if not selected:
                    continue
                steps = sorted(set.intersection(*(set(step for step, _ in run["curve"]) for run in selected)))
                values = np.array([[dict(run["curve"])[step] for step in steps] for run in selected])
                mean = values.mean(axis=0)
                axis.plot(steps, mean, color=color, label=f"{method.upper()} (n={len(selected)})")
                if len(selected) > 1:
                    std = values.std(axis=0, ddof=1)
                    axis.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)
            axis.set_title(f"H={horizon}")
            axis.set_xlabel("Online environment steps")
            axis.set_ylabel("D4RL normalized return")
            axis.grid(alpha=0.2)
            handles, _ = axis.get_legend_handles_labels()
            if handles:
                axis.legend()
        label = "SMOKE VALIDATION — not performance evidence" if all(run["smoke"] for run in runs) else "mean ± seed standard deviation"
        fig.suptitle(f"{task}: {label}")
        fig.tight_layout()
        fig.savefig(output / f"{task}_online.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    runs, warnings = [], []
    for path in sorted(Path(args.root).rglob("manifest.json")):
        try:
            run = read_run(path.parent)
            if run["smoke"] and not args.include_smoke:
                continue
            runs.append(run)
        except (OSError, ValueError, KeyError) as error:
            warnings.append(f"{path.parent}: {error}")
    if not runs:
        raise SystemExit("No complete comparable runs; smoke runs require --include-smoke")
    for key in ("smoke", "source_commit", "offline_steps", "online_steps"):
        if len({run[key] for run in runs}) != 1:
            raise ValueError(f"Refusing to pool runs with different {key}")
    for task in {run["env_name"] for run in runs}:
        if len({run["norm"] for run in runs if run["env_name"] == task}) != 1:
            raise ValueError(f"Normalization differs across {task} runs")
        if len({run["settings_signature"] for run in runs if run["env_name"] == task}) != 1:
            raise ValueError(f"Training settings differ across {task} runs")
    summaries, paired = aggregate(runs)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = dict(runs=runs, aggregate=summaries, paired=paired, warnings=warnings,
                   auc_units="normalized-return * environment-steps", auc_mean="AUC / online budget")
    (output / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    fields = [key for key in runs[0] if key not in ("curve", "settings_signature")]
    with (output / "per_seed.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: run[key] for key in fields} for run in runs)
    lines = ["# Chunk-size comparison", "", "SMOKE VALIDATION ONLY; these short runs do not measure algorithm quality." if runs[0]["smoke"] else "Endpoint results, not best-checkpoint scores.", "",
             "| Task | Method | H | Seeds | Offline endpoint | Online start | Online endpoint | AUC / online steps |",
             "|---|---|---:|---|---:|---:|---:|---:|"]
    for row in summaries:
        values = [format_stat(row[key]) for key in ("offline_endpoint", "online_start", "online_endpoint", "online_auc_mean")]
        lines.append(f"| {row['env_name']} | {row['method']} | {row['chunk_size']} | {row['seeds']} | " + " | ".join(values) + " |")
    lines += ["", "## Paired N − B0", "", "| Task | H | Paired seeds | Offline Δ | Online endpoint Δ | Online Δ − Δ(H=1) |", "|---|---:|---|---:|---:|---:|"]
    for row in paired:
        lines.append(f"| {row['env_name']} | {row['chunk_size']} | {row['paired_seeds']} | {format_stat(row['delta_offline_endpoint'])} | {format_stat(row['delta_online_endpoint'])} | {format_stat(row['delta_minus_H1_online_endpoint'])} |")
    if warnings:
        lines += ["", "## Excluded / incomplete runs", ""] + [f"- {warning}" for warning in warnings]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    if not args.no_plots:
        plot_curves(runs, output)
    print(json.dumps({"runs": len(runs), "groups": len(summaries), "warnings": warnings, "output": str(output)}))


if __name__ == "__main__":
    main()
