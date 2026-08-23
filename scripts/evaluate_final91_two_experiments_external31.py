#!/usr/bin/env python3
"""Evaluate the two completed Final91 experiments on frozen External31 in parallel."""

import argparse
import json
import os
import sys
import threading
from pathlib import Path

from run_final91_two_gpu_experiments import run_streamed, write_comparison


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "benchmark_final91_a3_external31.py"

E200_TRAIN_DIR = Path("experiments/final91_a3_all91_e200")
A4_TRAIN_DIR = Path("experiments/final91_a4_appearance_all91")
E200_EVAL_DIR = Path("experiments/external31_final91_a3_e200_locked")
A4_EVAL_DIR = Path("experiments/external31_final91_a4_appearance_locked")
COMPARISON_DIR = Path("experiments/external31_final91_e200_vs_a4_appearance")
DEFAULT_IMAGE_DIR = Path("/data/v1/compressed/image")
DEFAULT_GT_DIR = Path("/data/v1/compressed/label")
DEFAULT_AUDIT = Path(
    "experiments/round5_supervised_91_a3/final91_live_label_audit.json"
)


def read_training_epoch(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if int(metadata.get("n_total_human_labels", -1)) != 91:
        raise RuntimeError(f"Training metadata is not Final91 all-91: {path}")
    epochs = int(metadata.get("final_training_epochs", -1))
    if epochs <= 0:
        raise RuntimeError(f"Training epoch is missing or invalid in {path}")
    return epochs


def evaluation_worker(name, gpu, command, log_path, results):
    environment = os.environ.copy()
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        exit_code = run_streamed(command, environment, log_path, f"{name}:EXT31")
        results[name] = {
            "status": "COMPLETE" if exit_code == 0 else "EVALUATION_FAILED",
            "exit_code": exit_code,
        }
    except Exception as exc:
        results[name] = {
            "status": "EXCEPTION",
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_command(
    args,
    checkpoint: Path,
    training_metadata: Path,
    output_dir: Path,
    model_label: str,
):
    return [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--config", args.config,
        "--image-dir", args.image_dir,
        "--gt-dir", args.gt_dir,
        "--checkpoint", str(checkpoint),
        "--training-metadata", str(training_metadata),
        "--audit-metadata", args.audit_metadata,
        "--output-dir", str(output_dir),
        "--model-label", model_label,
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Final91 E200 and A4 appearance on External31"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-e200", type=int, default=0)
    parser.add_argument("--gpu-appearance", type=int, default=1)
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--gt-dir", default=str(DEFAULT_GT_DIR))
    parser.add_argument("--audit-metadata", default=str(DEFAULT_AUDIT))
    parser.add_argument(
        "--e200-checkpoint",
        default=str(E200_TRAIN_DIR / "checkpoints" / "final_checkpoint.pth"),
    )
    parser.add_argument(
        "--e200-training-metadata",
        default=str(E200_TRAIN_DIR / "final_training_metadata.json"),
    )
    parser.add_argument(
        "--appearance-checkpoint",
        default=str(A4_TRAIN_DIR / "checkpoints" / "final_checkpoint.pth"),
    )
    parser.add_argument(
        "--appearance-training-metadata",
        default=str(A4_TRAIN_DIR / "final_training_metadata.json"),
    )
    parser.add_argument("--e200-output-dir", default=str(E200_EVAL_DIR))
    parser.add_argument("--appearance-output-dir", default=str(A4_EVAL_DIR))
    parser.add_argument("--comparison-dir", default=str(COMPARISON_DIR))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.gpu_e200 < 0 or args.gpu_appearance < 0:
        parser.error("GPU indices must be non-negative")
    if args.gpu_e200 == args.gpu_appearance:
        parser.error("Parallel evaluation requires two different GPU indices")

    e200_checkpoint = Path(args.e200_checkpoint)
    e200_metadata = Path(args.e200_training_metadata)
    a4_checkpoint = Path(args.appearance_checkpoint)
    a4_metadata = Path(args.appearance_training_metadata)
    e200_output = Path(args.e200_output_dir)
    a4_output = Path(args.appearance_output_dir)
    comparison_dir = Path(args.comparison_dir)

    if args.dry_run:
        e200_epochs = "TRAINED"
        a4_epochs = "TRAINED"
    else:
        for required in [
            e200_checkpoint,
            e200_metadata,
            a4_checkpoint,
            a4_metadata,
            Path(args.audit_metadata),
        ]:
            if not required.exists():
                raise FileNotFoundError(required)
        e200_epochs = str(read_training_epoch(e200_metadata))
        a4_epochs = str(read_training_epoch(a4_metadata))

    e200_command = build_command(
        args,
        e200_checkpoint,
        e200_metadata,
        e200_output,
        f"FINAL91 A3 E{e200_epochs}",
    )
    a4_command = build_command(
        args,
        a4_checkpoint,
        a4_metadata,
        a4_output,
        f"FINAL91 A4 APPEARANCE E{a4_epochs}",
    )

    plan = {
        "version": "final91_two_experiment_external31_evaluation_v1",
        "external31_role": "repeated_frozen_diagnostic_not_model_selection",
        "prediction": "Student+EMA 50/50 @ 0.50; report raw primary and fixed LCC diagnostic",
        "jobs": {
            "E200": {"gpu": args.gpu_e200, "command": e200_command},
            "A4_APPEARANCE": {
                "gpu": args.gpu_appearance,
                "command": a4_command,
            },
        },
    }
    comparison_dir.mkdir(parents=True, exist_ok=True)
    (comparison_dir / "external31_evaluation_plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )

    print("=" * 112)
    print("FINAL91 TWO-EXPERIMENT PARALLEL EXTERNAL31 EVALUATION")
    print(f"GPU {args.gpu_e200}: A3 E{e200_epochs}")
    print(f"GPU {args.gpu_appearance}: A4 appearance E{a4_epochs}")
    print("Each evaluator processes the same 31 cases one-by-one.")
    print("Locked prediction: Student+EMA 50/50 @ .50 | RAW primary + LCC diagnostic")
    print(f"Comparison output: {comparison_dir}")
    print("=" * 112)

    if args.dry_run:
        print("DRY RUN — evaluation was not started.")
        print(f"E200: {' '.join(e200_command)}")
        print(f"A4:   {' '.join(a4_command)}")
        return

    results = {}
    log_dir = comparison_dir / "logs"
    threads = [
        threading.Thread(
            target=evaluation_worker,
            args=(
                "E200",
                args.gpu_e200,
                e200_command,
                log_dir / "e200_external31.log",
                results,
            ),
        ),
        threading.Thread(
            target=evaluation_worker,
            args=(
                "A4_APPEARANCE",
                args.gpu_appearance,
                a4_command,
                log_dir / "a4_appearance_external31.log",
                results,
            ),
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    (comparison_dir / "external31_evaluation_status.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    failed = {
        name: result
        for name, result in results.items()
        if result.get("status") != "COMPLETE"
    }
    if failed:
        raise RuntimeError(f"One or more External31 evaluations failed: {failed}")

    write_comparison(e200_output, a4_output, comparison_dir)


if __name__ == "__main__":
    main()
