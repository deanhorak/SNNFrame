#!/usr/bin/env python3
"""Run small substrate-level sweeps over retinal spike learning knobs.

The runner creates temporary config copies under build/substrate_sweeps so
checked-in SONATA configs stay unchanged. It is intentionally conservative:
small defaults make it useful as a smoke gate, while CLI ranges can expand it
for longer optimization passes.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
BINARY = BUILD / "experiments" / "emnist_retina_letters"
OUT_DIR = BUILD / "substrate_sweeps"


DATASETS = {
    "mnist": {
        "config": ROOT / "configs" / "mnist_retina_bilateral_experimental.sonata.json",
        "train_images": [ROOT / "data/MNIST/train-images-idx3-ubyte", ROOT.parent / "data/MNIST/raw/train-images-idx3-ubyte"],
        "train_labels": [ROOT / "data/MNIST/train-labels-idx1-ubyte", ROOT.parent / "data/MNIST/raw/train-labels-idx1-ubyte"],
        "test_images": [ROOT / "data/MNIST/t10k-images-idx3-ubyte", ROOT.parent / "data/MNIST/raw/t10k-images-idx3-ubyte"],
        "test_labels": [ROOT / "data/MNIST/t10k-labels-idx1-ubyte", ROOT.parent / "data/MNIST/raw/t10k-labels-idx1-ubyte"],
    },
    "emnist": {
        "config": ROOT / "configs" / "emnist_retina_bilateral_experimental.sonata.json",
        "train_images": [ROOT / "data/EMNIST/emnist-letters-train-images-idx3-ubyte"],
        "train_labels": [ROOT / "data/EMNIST/emnist-letters-train-labels-idx1-ubyte"],
        "test_images": [ROOT / "data/EMNIST/emnist-letters-test-images-idx3-ubyte"],
        "test_labels": [ROOT / "data/EMNIST/emnist-letters-test-labels-idx1-ubyte"],
    },
    "cifar10": {
        "config": ROOT / "configs" / "cifar10_retina_bilateral_natural_features_experimental.sonata.json",
        "train_images": [
            ROOT / "data/cifar-10-batches-bin",
            ROOT / "data/CIFAR_10/cifar-10-binary/cifar-10-batches-bin",
            ROOT.parent / "data/cifar-10-batches-bin",
            ROOT.parent / "data/CIFAR_10/cifar-10-binary/cifar-10-batches-bin",
        ],
        "train_labels": [
            ROOT / "data/cifar-10-batches-bin",
            ROOT / "data/CIFAR_10/cifar-10-binary/cifar-10-batches-bin",
            ROOT.parent / "data/cifar-10-batches-bin",
            ROOT.parent / "data/CIFAR_10/cifar-10-binary/cifar-10-batches-bin",
        ],
        "test_images": [
            ROOT / "data/cifar-10-batches-bin/test_batch.bin",
            ROOT / "data/CIFAR_10/cifar-10-binary/cifar-10-batches-bin/test_batch.bin",
            ROOT.parent / "data/cifar-10-batches-bin/test_batch.bin",
            ROOT.parent / "data/CIFAR_10/cifar-10-binary/cifar-10-batches-bin/test_batch.bin",
        ],
        "test_labels": [
            ROOT / "data/cifar-10-batches-bin/test_batch.bin",
            ROOT / "data/CIFAR_10/cifar-10-binary/cifar-10-batches-bin/test_batch.bin",
            ROOT.parent / "data/cifar-10-batches-bin/test_batch.bin",
            ROOT.parent / "data/CIFAR_10/cifar-10-binary/cifar-10-batches-bin/test_batch.bin",
        ],
    },
}


def parse_csv(value: str, cast: Any) -> list[Any]:
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def mutate_adapters(node: Any, temporal_window: float, threshold: float, max_patterns: int) -> None:
    if isinstance(node, dict):
        if node.get("type") == "retina":
            node["temporal_window_ms"] = temporal_window
            node.setdefault("double_params", {})["neuron_window_size"] = temporal_window
            node.setdefault("double_params", {})["neuron_threshold"] = threshold
            node.setdefault("int_params", {})["neuron_max_patterns"] = max_patterns
        for value in node.values():
            mutate_adapters(value, temporal_window, threshold, max_patterns)
    elif isinstance(node, list):
        for item in node:
            mutate_adapters(item, temporal_window, threshold, max_patterns)


def write_temp_config(base_path: Path, dataset: str, run_id: str, temporal_window: float, threshold: float, max_patterns: int) -> Path:
    config = json.loads(base_path.read_text())
    mutate_adapters(config, temporal_window, threshold, max_patterns)
    config_dir = OUT_DIR / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{dataset}_{run_id}.sonata.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path


def parse_metrics(output: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    patterns = {
        "accuracy": r"Accuracy:\s+([0-9.]+)%",
        "correct": r"Correct:\s+([0-9]+)/([0-9]+)",
        "elapsed_reported_s": r"Elapsed:\s+([0-9.]+)s",
        "initial_accuracy": r"Initial accuracy:\s+([0-9.]+)%",
        "post_correction_accuracy": r"Post-correction accuracy:\s+([0-9.]+)%",
        "hemisphere_agreement": r"Hemisphere agreement:\s+[0-9]+/[0-9]+ \(([0-9.]+)%\)",
        "agreement_accuracy": r"agreement accuracy=([0-9.]+)%",
        "left_stage1_accuracy": r"Stage-1 accuracy by view: left=([0-9.]+)%",
        "right_stage1_accuracy": r"Stage-1 accuracy by view: left=[0-9.]+%, right=([0-9.]+)%",
        "centroid_accuracy": r"Final representation separability: centroid_acc=([0-9.]+)%",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if not match:
            continue
        if key == "correct":
            metrics["correct"] = int(match.group(1))
            metrics["tested"] = int(match.group(2))
        else:
            metrics[key] = float(match.group(1))
    return metrics


def build_command(args: argparse.Namespace, config_path: Path, dataset_info: dict[str, Any], extra_args: list[str]) -> list[str]:
    return [
        str(BINARY),
        "--config",
        str(config_path),
        "--train-images",
        str(first_existing(dataset_info["train_images"])),
        "--train-labels",
        str(first_existing(dataset_info["train_labels"])),
        "--test-images",
        str(first_existing(dataset_info["test_images"])),
        "--test-labels",
        str(first_existing(dataset_info["test_labels"])),
        "--examples-per-class",
        str(args.examples_per_class),
        "--test-limit",
        str(args.test_limit),
        "--seed",
        str(args.seed),
        *extra_args,
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS.keys(), default="mnist")
    parser.add_argument("--examples-per-class", type=int, default=80)
    parser.add_argument("--test-limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temporal-windows", default="100,200,300")
    parser.add_argument("--thresholds", default="0.6,0.7,0.8")
    parser.add_argument("--max-patterns", default="50,100")
    parser.add_argument("--activation-modes", default="binary")
    parser.add_argument("--encoding-strategies", default="rate")
    parser.add_argument("--stdp-modes", default="off")
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--limit-runs", type=int, default=0)
    parser.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    if not BINARY.exists():
        print(f"Missing experiment binary: {BINARY}", file=sys.stderr)
        return 2

    dataset_info = DATASETS[args.dataset]
    temporal_windows = parse_csv(args.temporal_windows, float)
    thresholds = parse_csv(args.thresholds, float)
    max_patterns = parse_csv(args.max_patterns, int)
    activation_modes = parse_csv(args.activation_modes, str)
    encoding_strategies = parse_csv(args.encoding_strategies, str)
    stdp_modes = parse_csv(args.stdp_modes, str)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / f"{args.dataset}_{args.tag}_summary.csv"
    rows: list[dict[str, Any]] = []

    combinations = list(itertools.product(
        temporal_windows,
        thresholds,
        max_patterns,
        activation_modes,
        encoding_strategies,
        stdp_modes))
    if args.limit_runs > 0:
        combinations = combinations[: args.limit_runs]

    for index, (
        window,
        threshold,
        patterns,
        activation_mode,
        encoding_strategy,
        stdp_mode,
    ) in enumerate(combinations, start=1):
        run_id = (
            f"{args.tag}_w{window:g}_t{threshold:g}_p{patterns}_"
            f"{activation_mode}_{encoding_strategy}_{stdp_mode}"
        )
        config_path = write_temp_config(dataset_info["config"], args.dataset, run_id, window, threshold, patterns)
        extra_args: list[str] = [
            "--activation-mode",
            activation_mode,
            "--encoding-strategy",
            encoding_strategy,
        ]
        if stdp_mode == "reward":
            extra_args.extend(["--online-reward-stdp-enabled", "--online-reward-stdp-gain", "0.05"])
        elif stdp_mode == "triplet":
            extra_args.extend(["--online-triplet-stdp-enabled", "--online-triplet-stdp-gain", "0.05"])
        elif stdp_mode != "off":
            raise ValueError(f"Unknown stdp mode: {stdp_mode}")

        cmd = build_command(args, config_path, dataset_info, extra_args)
        log_path = OUT_DIR / f"{args.dataset}_{run_id}.log"
        print(
            f"[{index}/{len(combinations)}] dataset={args.dataset} window={window:g} "
            f"threshold={threshold:g} max_patterns={patterns} activation={activation_mode} "
            f"encoding={encoding_strategy} stdp={stdp_mode}")
        started = time.monotonic()
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"/usr/lib/x86_64-linux-gnu:{env.get('LD_LIBRARY_PATH', '')}"
        try:
            completed = subprocess.run(
                cmd,
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.timeout_seconds if args.timeout_seconds > 0 else None,
                check=False,
            )
            output = completed.stdout
            return_code = completed.returncode
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            return_code = 124
        wall_s = time.monotonic() - started
        log_path.write_text(output)

        row: dict[str, Any] = {
            "dataset": args.dataset,
            "run_id": run_id,
            "temporal_window_ms": window,
            "neuron_threshold": threshold,
            "neuron_max_patterns": patterns,
            "activation_mode": activation_mode,
            "encoding_strategy": encoding_strategy,
            "stdp_mode": stdp_mode,
            "return_code": return_code,
            "wall_s": round(wall_s, 3),
            "log_path": str(log_path),
        }
        row.update(parse_metrics(output))
        rows.append(row)

        fieldnames = sorted({key for item in rows for key in item.keys()})
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        accuracy = row.get("accuracy", row.get("post_correction_accuracy", "NA"))
        correct = f"{row.get('correct', 'NA')}/{row.get('tested', 'NA')}"
        print(f"  -> rc={return_code} accuracy={accuracy} correct={correct} wall={wall_s:.1f}s log={log_path}")

    print(f"Summary: {summary_path}")
    return 0 if all(row["return_code"] == 0 for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
