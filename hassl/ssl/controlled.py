"""Utilities for leakage-safe controlled SSL experiments.

These helpers deliberately live outside the default HASSL pipeline so experimental SSL
changes cannot alter the established active-learning/training behavior.
"""

from __future__ import annotations

import csv
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import torch
from monai.data import DataLoader, PersistentDataset

from hassl.data.data_engine import get_base_transforms


TASK_HEAD_MARKERS = (
    "output_block",
    "deep_supervision_heads",
)


def strip_suffix(filename: str, suffix: str) -> str:
    if suffix and filename.endswith(suffix):
        return filename[: -len(suffix)]
    return filename


def read_case_ids(path: Path) -> Set[str]:
    """Read case IDs from CSV, JSON, or newline-delimited text.

    CSV accepts case_id/id/volume_id columns. JSON accepts a list of IDs or a dictionary
    containing case_ids/external_case_ids/ids. The function intentionally refuses ambiguous
    dictionary shapes instead of guessing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise RuntimeError(f"Case-ID CSV is empty: {path}")
        fieldnames = rows[0].keys()
        key = next((k for k in ("case_id", "id", "volume_id") if k in fieldnames), None)
        if key is None:
            raise RuntimeError(
                f"Case-ID CSV needs one of case_id/id/volume_id columns: {path}"
            )
        ids = {str(r[key]).strip() for r in rows if str(r.get(key, "")).strip()}
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            values = payload
        elif isinstance(payload, dict):
            values = None
            for key in ("case_ids", "external_case_ids", "ids"):
                if key in payload:
                    values = payload[key]
                    break
            if values is None:
                raise RuntimeError(
                    f"JSON exclusion manifest needs case_ids/external_case_ids/ids: {path}"
                )
        else:
            raise RuntimeError(f"Unsupported JSON exclusion manifest shape: {path}")
        ids = {str(x).strip() for x in values if str(x).strip()}
    else:
        ids = {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

    if not ids:
        raise RuntimeError(f"No case IDs found in {path}")
    return ids


def read_audited_human_ids(audit_path: Path, expected_count: int = 62) -> Set[str]:
    audit_path = Path(audit_path)
    if not audit_path.exists():
        raise FileNotFoundError(audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Human-label audit is not marked passing")
    ids = {str(x) for x in audit.get("all_current_human_label_ids", [])}
    reported = int(audit.get("n_current_valid_human_labels", len(ids)))
    if len(ids) != reported or reported != int(expected_count):
        raise RuntimeError(
            f"Expected exactly {expected_count} audited HUMAN_GOLD cases, "
            f"found ids={len(ids)}, reported={reported}"
        )
    return ids


def discover_unique_images(data_dir: str, image_suffix: str) -> Dict[str, str]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(root)
    by_id: Dict[str, str] = {}
    duplicates: Dict[str, List[str]] = {}
    for path in sorted(root.rglob(f"*{image_suffix}")):
        case_id = strip_suffix(path.name, image_suffix)
        if case_id in by_id:
            duplicates.setdefault(case_id, [by_id[case_id]]).append(str(path))
        else:
            by_id[case_id] = str(path)
    if duplicates:
        preview = "\n".join(
            f"  {case_id}: {paths}" for case_id, paths in list(duplicates.items())[:10]
        )
        raise RuntimeError(
            "Duplicate image case IDs found under data_dir; SSL provenance would be ambiguous:\n"
            + preview
        )
    if not by_id:
        raise RuntimeError(f"No *{image_suffix} images found under {root}")
    return by_id


def build_controlled_ssl_loader(
    config,
    audit_path: Path,
    external_case_manifest: Path,
    output_cache_dir: Path,
    resize_size: int = 128,
    expected_human_count: int = 62,
):
    """Build a fixed-grid SSL loader and return loader + provenance audit metadata.

    The pool contains every unique image under config.data_dir except IDs in the explicit
    external-case manifest. All audited HUMAN_GOLD cases must be present. No HASSL internal
    train/val/test split is used because the downstream Final62 model trains on all 62 labels;
    the independent external31 is the holdout that must remain unseen.
    """
    human_ids = read_audited_human_ids(audit_path, expected_count=expected_human_count)
    external_ids = read_case_ids(external_case_manifest)
    overlap_human_external = sorted(human_ids & external_ids)
    if overlap_human_external:
        raise RuntimeError(
            "HUMAN_GOLD/external ID overlap detected: " + ", ".join(overlap_human_external)
        )

    discovered = discover_unique_images(config.data_dir, config.image_suffix)
    discovered_ids = set(discovered)
    missing_human = sorted(human_ids - discovered_ids)
    if missing_human:
        raise RuntimeError(
            "Audited HUMAN_GOLD images missing from config.data_dir: " + ", ".join(missing_human)
        )

    external_present = sorted(discovered_ids & external_ids)
    included_ids = sorted(discovered_ids - external_ids)
    leakage = sorted(set(included_ids) & external_ids)
    if leakage:
        raise RuntimeError("External images leaked into SSL pool: " + ", ".join(leakage))

    ssl_config = deepcopy(config)
    ssl_config.preprocessing_mode = "resize"
    ssl_config.spatial_size = (int(resize_size),) * 3

    transform = get_base_transforms(
        ssl_config,
        keys=["image"],
        is_training=False,
        apply_strong_aug=False,
    )
    items = [{"image": discovered[case_id], "id": case_id} for case_id in included_ids]

    output_cache_dir = Path(output_cache_dir)
    output_cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = PersistentDataset(data=items, transform=transform, cache_dir=str(output_cache_dir))

    num_workers = int(getattr(ssl_config, "num_workers", 0))
    if os.name == "nt":
        num_workers = 0
    loader = DataLoader(
        dataset,
        batch_size=int(getattr(ssl_config, "batch_size", 1)),
        shuffle=True,
        num_workers=num_workers,
    )

    metadata = {
        "n_discovered_images": len(discovered_ids),
        "n_audited_human_gold": len(human_ids),
        "n_external_ids_manifest": len(external_ids),
        "n_external_images_present_under_data_dir": len(external_present),
        "n_ssl_pool": len(included_ids),
        "n_nonhuman_pool_images": len(set(included_ids) - human_ids),
        "human_gold_ids": sorted(human_ids),
        "external_ids": sorted(external_ids),
        "external_ids_present_under_data_dir": external_present,
        "ssl_case_ids": included_ids,
        "external_overlap_after_filter": leakage,
        "external_overlap_status": "PASS" if not leakage else "FAIL",
        "preprocessing_mode": "resize",
        "spatial_size": [int(resize_size)] * 3,
        "spacing": [float(x) for x in ssl_config.spacing],
    }
    return ssl_config, loader, metadata


def _extract_ssl_state(checkpoint: Path):
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        return payload["model"], payload.get("metadata", {})
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"], payload.get("metadata", {})
    if isinstance(payload, dict) and payload and all(torch.is_tensor(v) for v in payload.values()):
        return payload, {}
    raise RuntimeError(f"Unsupported SSL checkpoint format: {checkpoint}")


def is_task_specific_head(key: str) -> bool:
    return any(marker in key for marker in TASK_HEAD_MARKERS)


def load_ssl_features_into_model(model: torch.nn.Module, checkpoint: Path):
    """Load compatible SSL feature weights while intentionally skipping task output heads."""
    ssl_state, checkpoint_meta = _extract_ssl_state(checkpoint)
    target = model.state_dict()
    transfer = {}
    skipped_task_heads = []
    skipped_shape = []
    skipped_missing = []

    for key, value in ssl_state.items():
        if is_task_specific_head(key):
            skipped_task_heads.append(key)
            continue
        if key not in target:
            skipped_missing.append(key)
            continue
        if tuple(value.shape) != tuple(target[key].shape):
            skipped_shape.append(key)
            continue
        transfer[key] = value

    if not transfer:
        raise RuntimeError("Zero compatible SSL feature tensors were found")

    target.update(transfer)
    model.load_state_dict(target)
    report = {
        "checkpoint": str(checkpoint),
        "transferred_tensor_count": len(transfer),
        "target_tensor_count": len(target),
        "skipped_task_head_count": len(skipped_task_heads),
        "skipped_shape_count": len(skipped_shape),
        "skipped_missing_count": len(skipped_missing),
        "skipped_task_head_keys": sorted(skipped_task_heads),
        "checkpoint_metadata": checkpoint_meta,
    }
    return report


def initialize_trainer_from_ssl(trainer, checkpoint: Path):
    """Transfer SSL features into the student and synchronize the EMA teacher afterward."""
    report = load_ssl_features_into_model(trainer.net_A, checkpoint)
    if getattr(trainer, "mode", None) != "prototype" or not hasattr(trainer, "teacher"):
        raise RuntimeError("Controlled Final62 SSL experiment requires prototype EMA trainer")
    trainer.teacher.load_state_dict(trainer.net_A.state_dict())
    report["ema_teacher_synchronized_after_ssl_load"] = True
    return report
