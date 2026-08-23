#!/usr/bin/env python3
"""Run two Final91 experiments in parallel on separate GPUs, then evaluate External31.

Experiment E200 (default GPU 0)
    Final91 A3, all 91 HUMAN_GOLD, fixed 200-epoch cap.

Experiment A4_APPEARANCE (default GPU 1)
    Final91 A3 plus the locked mild ultrasound appearance augmentation bundle. The
    epoch count defaults to the same CV-derived full-training duration as Final91 A3.

Each worker trains first and then runs the locked raw Student+EMA 50/50 ensemble at
threshold 0.50 over the same 31 external cases one-by-one. External31 is diagnostic;
the script performs no threshold, checkpoint, loss, augmentation, or post-processing search.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_final91_a3_all91.py"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "benchmark_final91_a3_external31.py"

E200_TRAIN_DIR = Path("experiments/final91_a3_all91_e200")
A4_TRAIN_DIR = Path("experiments/final91_a4_appearance_all91")
E200_EVAL_DIR = Path("experiments/external31_final91_a3_e200_locked")
A4_EVAL_DIR = Path("experiments/external31_final91_a4_appearance_locked")
RUN_DIR = Path("experiments/final91_parallel_e200_vs_a4_appearance")
DEFAULT_IMAGE_DIR = Path("/data/v1/compressed/image")
DEFAULT_GT_DIR = Path("/data/v1/compressed/label")


def run_streamed(command, env, log_path: Path, prefix: str):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(f"[{prefix}] {line}", end="", flush=True)
        return int(process.wait())


def worker(name, gpu, train_command, eval_command, log_dir, results):
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        train_code = run_streamed(
            train_command, env, log_dir / f"{name.lower()}_train.log", f"{name}:TRAIN"
        )
        if train_code != 0:
            results[name] = {"status": "TRAIN_FAILED", "exit_code": train_code}
            return
        eval_code = run_streamed(
            eval_command, env, log_dir / f"{name.lower()}_external31.log", f"{name}:EXT31"
        )
        results[name] = {
            "status": "COMPLETE" if eval_code == 0 else "EVALUATION_FAILED",
            "exit_code": eval_code,
        }
    except Exception as exc:
        results[name] = {"status": "EXCEPTION", "error": f"{type(exc).__name__}: {exc}"}


def read_ensemble(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if str(row.get("mode", "")).upper() == "ENSEMBLE"
        ]
    by_id = {str(row["case_id"]): row for row in rows}
    if len(by_id) != 31:
        raise RuntimeError(f"Expected 31 ENSEMBLE rows in {path}, found {len(by_id)}")
    return by_id


def write_comparison(e200_eval_dir: Path, a4_eval_dir: Path, run_dir: Path):
    e200 = read_ensemble(e200_eval_dir / "external31_case_metrics.csv")
    a4 = read_ensemble(a4_eval_dir / "external31_case_metrics.csv")
    if set(e200) != set(a4):
        raise RuntimeError("E200 and A4 External31 IDs differ")
    rows = []
    for case_id in sorted(e200):
        first, second = e200[case_id], a4[case_id]
        rows.append({
            "case_id": case_id,
            "e200_dice": float(first["dice"]),
            "a4_appearance_dice": float(second["dice"]),
            "a4_minus_e200_dice": float(second["dice"]) - float(first["dice"]),
            "e200_precision": float(first["precision"]),
            "a4_appearance_precision": float(second["precision"]),
            "e200_recall": float(first["recall"]),
            "a4_appearance_recall": float(second["recall"]),
            "e200_signed_rve_pct": float(first["signed_rve_pct"]),
            "a4_appearance_signed_rve_pct": float(second["signed_rve_pct"]),
            "e200_hd95_mm": float(first["hd95_mm"]),
            "a4_appearance_hd95_mm": float(second["hd95_mm"]),
        })
    run_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = run_dir / "external31_e200_vs_a4_appearance_case_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    e200_dice = [row["e200_dice"] for row in rows]
    a4_dice = [row["a4_appearance_dice"] for row in rows]
    delta = [row["a4_minus_e200_dice"] for row in rows]
    summary = {
        "version": "final91_e200_vs_a4_appearance_external31_diagnostic_v1",
        "n": len(rows),
        "e200_mean_dice": sum(e200_dice) / len(e200_dice),
        "a4_appearance_mean_dice": sum(a4_dice) / len(a4_dice),
        "a4_minus_e200_mean_dice": sum(delta) / len(delta),
        "a4_improved_cases": sum(value > 1e-6 for value in delta),
        "a4_worsened_cases": sum(value < -1e-6 for value in delta),
        "a4_improved_ge_0p05": sum(value >= 0.05 for value in delta),
        "a4_worsened_le_minus_0p05": sum(value <= -0.05 for value in delta),
        "external31_role": "repeated_frozen_diagnostic_not_model_selection",
    }
    summary_path = run_dir / "external31_e200_vs_a4_appearance_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + "=" * 112)
    print("FINAL91 TWO-EXPERIMENT EXTERNAL31 DIAGNOSTIC")
    print(
        f"Mean Dice E200 -> A4 appearance: {summary['e200_mean_dice']:.4f} -> "
        f"{summary['a4_appearance_mean_dice']:.4f} "
        f"({summary['a4_minus_e200_mean_dice']:+.4f})"
    )
    print(
        f"A4 cases improved={summary['a4_improved_cases']} | "
        f"worsened={summary['a4_worsened_cases']} | "
        f"+>=.05={summary['a4_improved_ge_0p05']} | "
        f"<=-.05={summary['a4_worsened_le_minus_0p05']}"
    )
    print(f"Case comparison: {comparison_path}")
    print(f"Summary:         {summary_path}")
    print("External31 was not used to tune either experiment.")
    print("=" * 112)


def main():
    parser = argparse.ArgumentParser(description="Run Final91 E200 and A4 appearance in parallel")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-e200", type=int, default=0)
    parser.add_argument("--gpu-appearance", type=int, default=1)
    parser.add_argument("--long-epochs", type=int, default=200)
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--gt-dir", default=str(DEFAULT_GT_DIR))
    parser.add_argument("--run-dir", default=str(RUN_DIR))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write and print the two commands without starting either GPU worker",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.gpu_e200 < 0 or args.gpu_appearance < 0:
        parser.error("GPU indices must be non-negative")
    if args.gpu_e200 == args.gpu_appearance:
        parser.error("Parallel experiments require two different GPU indices")
    if args.long_epochs <= 100 or args.long_epochs > 300:
        parser.error("--long-epochs must be between 101 and 300")

    run_dir = Path(args.run_dir)
    log_dir = run_dir / "logs"
    e200_train_dir = E200_TRAIN_DIR
    a4_train_dir = A4_TRAIN_DIR
    e200_eval_dir = E200_EVAL_DIR
    a4_eval_dir = A4_EVAL_DIR

    common_train = [sys.executable, str(TRAIN_SCRIPT), "--config", args.config]
    e200_train = common_train + [
        "--epochs", str(args.long_epochs), "--output-dir", str(e200_train_dir)
    ]
    a4_train = common_train + [
        "--appearance-augmentation", "--output-dir", str(a4_train_dir)
    ]
    if args.overwrite:
        e200_train.append("--overwrite")
        a4_train.append("--overwrite")

    common_eval = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--config", args.config,
        "--image-dir", args.image_dir,
        "--gt-dir", args.gt_dir,
    ]
    e200_eval = common_eval + [
        "--checkpoint", str(e200_train_dir / "checkpoints" / "final_checkpoint.pth"),
        "--training-metadata", str(e200_train_dir / "final_training_metadata.json"),
        "--output-dir", str(e200_eval_dir),
        "--model-label", f"FINAL91 A3 E{args.long_epochs}",
    ]
    a4_eval = common_eval + [
        "--checkpoint", str(a4_train_dir / "checkpoints" / "final_checkpoint.pth"),
        "--training-metadata", str(a4_train_dir / "final_training_metadata.json"),
        "--output-dir", str(a4_eval_dir),
        "--model-label", "FINAL91 A4 APPEARANCE",
    ]

    plan = {
        "version": "final91_parallel_e200_vs_a4_appearance_v1",
        "external31_role": "repeated_frozen_diagnostic_not_model_selection",
        "experiments": {
            "E200": {
                "gpu": args.gpu_e200,
                "training_command": e200_train,
                "evaluation_command": e200_eval,
            },
            "A4_APPEARANCE": {
                "gpu": args.gpu_appearance,
                "training_command": a4_train,
                "evaluation_command": a4_eval,
            },
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "parallel_experiment_plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )

    print("=" * 112)
    print("FINAL91 PARALLEL EXPERIMENTS")
    print(f"GPU {args.gpu_e200}: A3 with {args.long_epochs}-epoch cap")
    print(f"GPU {args.gpu_appearance}: A4 mild ultrasound appearance augmentation")
    print("After training: locked Student+EMA 50/50 @ .50 External31, case-by-case")
    print(f"Logs: {log_dir}")
    print("=" * 112)

    if args.dry_run:
        print("DRY RUN — no training or External31 evaluation was started.")
        print(f"Plan: {run_dir / 'parallel_experiment_plan.json'}")
        for name, item in plan["experiments"].items():
            print(f"{name} train: {' '.join(item['training_command'])}")
            print(f"{name} eval:  {' '.join(item['evaluation_command'])}")
        return

    results = {}
    threads = [
        threading.Thread(
            target=worker,
            args=("E200", args.gpu_e200, e200_train, e200_eval, log_dir, results),
        ),
        threading.Thread(
            target=worker,
            args=(
                "A4_APPEARANCE", args.gpu_appearance, a4_train, a4_eval, log_dir, results
            ),
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    (run_dir / "parallel_experiment_status.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    failed = {name: result for name, result in results.items() if result.get("status") != "COMPLETE"}
    if failed:
        raise RuntimeError(f"One or more parallel experiments failed: {failed}")
    write_comparison(e200_eval_dir, a4_eval_dir, run_dir)


if __name__ == "__main__":
    main()
