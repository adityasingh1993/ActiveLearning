"""Pure cohort/provenance rules shared by the two-stage training scripts.

The historical CenterNet OOF artifacts were produced from the frozen original-47
manifest.  Final136 intentionally keeps those detector folds as the internal
validation source while training Stage 2 with every currently available Final136
case except the fold's usable frozen-source validation cases.

This module contains no Torch or MONAI imports so the data contract can be tested
without a training environment.
"""


EXPECTED_FROZEN_SOURCE = 47
EXPECTED_FINAL91 = 91
EXPECTED_FINAL136 = 136
PRIOR_QUARANTINE_ID = (
    "9435b1b67a41b88f6084a3e750fc54d913213ea55f33d165a1f42b9b50dd237c"
)
FINAL136_ALLOWED_MISSING_FROZEN_IDS = frozenset({
    "81a0f3f3fa1e2ad8bd01d3915898298ffff947a83b5749846b228a612356fb2f",
    "96165b4ca29e10f85866ca53ac68e4e44a127fe676e5745e0a94d50602819f0f",
    "a31909b0e87f789c68489e8ebe6a5adfb72c5a19dbe5546cd762ea2f82c7037a",
    "d8269b9a976314fb41fa63187c4b2dc05ee94e2973125df67bb5e4fd75bc7563",
})
FINAL136_TRAINING_SCOPE = (
    "all_136_current_live_labels_with_exact_four_missing_frozen_allowed"
)


def _ids(values):
    return sorted({str(value) for value in values})


def final136_mode(expected_live):
    return int(expected_live) == EXPECTED_FINAL136


def resolve_stage2_cohort(
    audit,
    source_ids,
    current_ids,
    *,
    expected_live,
    expected_trainable=None,
    include_quarantined=False,
    allow_missing_frozen=False,
    split_scope="original46",
):
    """Validate and resolve the Final91 or Final136 Stage-2 CV population."""

    expected_live = int(expected_live)
    if expected_live not in {EXPECTED_FINAL91, EXPECTED_FINAL136}:
        raise RuntimeError("Stage-2 cohort must be exactly Final91 or Final136")

    source_ids = _ids(source_ids)
    current_ids = _ids(current_ids)
    audited_ids = _ids(audit.get("all_current_human_label_ids", []))
    if len(source_ids) != EXPECTED_FROZEN_SOURCE:
        raise RuntimeError("Source manifest is not the frozen original47")
    if PRIOR_QUARANTINE_ID not in source_ids or PRIOR_QUARANTINE_ID not in current_ids:
        raise RuntimeError("Historical 9435... case is absent from source or current cohort")
    if len(current_ids) != expected_live or current_ids != audited_ids:
        raise RuntimeError(
            f"Current dataset does not match the audited Final{expected_live} cohort"
        )
    if audit.get("external31_access") is not False:
        raise RuntimeError("Training audit must record External31 access as false")

    audit_ok = bool(
        audit.get(
            "all_training_labels_passed_audit",
            audit.get("all_visible_labels_passed_audit", False),
        )
    )
    provenance_ok = bool(
        audit.get("selection_provenance_enforced", False)
        or audit.get("training_scope_provenance_enforced", False)
    )
    if not audit_ok or not provenance_ok:
        raise RuntimeError(f"Final{expected_live} audit/provenance is not passing")

    if bool(audit.get("missing_frozen_allowed", False)) != bool(allow_missing_frozen):
        raise RuntimeError(
            "Training --allow-missing-frozen policy differs from the audit metadata"
        )
    missing_frozen_ids = _ids(set(source_ids) - set(current_ids))
    audit_missing_ids = _ids(audit.get("missing_frozen_case_ids", []))
    audit_allowed_missing_ids = _ids(audit.get("allowed_missing_frozen_case_ids", []))
    if allow_missing_frozen:
        approved = sorted(FINAL136_ALLOWED_MISSING_FROZEN_IDS)
        if (
            missing_frozen_ids != approved
            or audit_missing_ids != approved
            or audit_allowed_missing_ids != approved
        ):
            raise RuntimeError(
                "Missing frozen IDs differ from the exact approved Final136 four-case allowlist"
            )
    elif missing_frozen_ids or audit_missing_ids or audit_allowed_missing_ids:
        raise RuntimeError("Frozen source labels are missing without explicit approval")

    excluded_ids = _ids(audit.get("excluded_training_case_ids", []))
    if set(excluded_ids) - set(current_ids):
        raise RuntimeError("Audit excludes IDs outside the current cohort")

    if final136_mode(expected_live):
        if str(split_scope) != "original46":
            raise RuntimeError("Final136 currently requires the frozen original46 OOF scope")
        if not include_quarantined or not allow_missing_frozen:
            raise RuntimeError(
                "Final136 requires --include-quarantined and --allow-missing-frozen"
            )
        if excluded_ids:
            raise RuntimeError("Final136 must train all 136 current cases")
        if audit.get("training_scope") != FINAL136_TRAINING_SCOPE:
            raise RuntimeError("Final136 audit training scope differs from the locked definition")
        if _ids(audit.get("training_case_ids", [])) != current_ids:
            raise RuntimeError("Final136 audit does not select all 136 current cases for training")
        if audit.get("quarantined_case_ids", []):
            raise RuntimeError("Final136 audit unexpectedly quarantines a current case")
        if _ids(audit.get("intentionally_included_prior_quarantine_ids", [])) != [
            PRIOR_QUARANTINE_ID
        ]:
            raise RuntimeError("Final136 audit does not record intentional 9435... inclusion")
    else:
        if include_quarantined or allow_missing_frozen:
            raise RuntimeError(
                "Final91 compatibility mode cannot include quarantine or allow missing frozen IDs"
            )
        if not audit.get("selection_provenance_enforced", False):
            raise RuntimeError("Final91 audit did not enforce selection provenance")

    training_ids = sorted(
        set(current_ids)
        - set(excluded_ids)
        - ({PRIOR_QUARANTINE_ID} if not include_quarantined else set())
    )
    expected_training_count = (
        int(expected_trainable) if expected_trainable is not None else len(training_ids)
    )
    if len(training_ids) != expected_training_count:
        raise RuntimeError(
            f"Expected exactly {expected_training_count} trainable cases, "
            f"found {len(training_ids)}"
        )
    if final136_mode(expected_live) and len(training_ids) != EXPECTED_FINAL136:
        raise RuntimeError("Final136 must contain exactly 136 training cases")
    if not final136_mode(expected_live) and len(training_ids) != EXPECTED_FINAL91 - 1:
        raise RuntimeError("Final91 compatibility mode must contain 90 training cases")

    stage1_quarantined_ids = [PRIOR_QUARANTINE_ID]
    scorable_ids = sorted(
        (set(source_ids) & set(current_ids))
        - set(stage1_quarantined_ids)
        - set(excluded_ids)
    )
    expected_scorable = (
        EXPECTED_FROZEN_SOURCE
        - len(missing_frozen_ids)
        - len(stage1_quarantined_ids)
    )
    if len(scorable_ids) != expected_scorable:
        raise RuntimeError(
            f"Expected {expected_scorable} usable frozen-source OOF cases, "
            f"found {len(scorable_ids)}"
        )
    if not set(scorable_ids).issubset(training_ids):
        raise RuntimeError("Frozen-source scoring population is not part of training cohort")

    extra_ids = sorted(set(current_ids) - set(source_ids))
    train_only_ids = sorted(set(training_ids) - set(scorable_ids))
    return {
        "expected_live": expected_live,
        "training_ids": training_ids,
        "excluded_training_case_ids": excluded_ids,
        "missing_frozen_case_ids": missing_frozen_ids,
        "stage1_quarantined_case_ids": stage1_quarantined_ids,
        "training_quarantined_case_ids": (
            [] if include_quarantined else [PRIOR_QUARANTINE_ID]
        ),
        "intentionally_included_prior_quarantine_ids": (
            [PRIOR_QUARANTINE_ID] if include_quarantined else []
        ),
        "scorable_ids": scorable_ids,
        "extra_ids": extra_ids,
        "train_only_ids": train_only_ids,
    }


def project_frozen_oof_folds(
    manifest_folds,
    training_ids,
    scorable_ids,
    stage1_quarantined_ids,
):
    """Project frozen original47 folds onto the currently available audited cohort."""

    training = set(_ids(training_ids))
    scorable = set(_ids(scorable_ids))
    stage1_quarantine = set(_ids(stage1_quarantined_ids))
    specs = []
    stage1_validation_ids = {}
    held_out = []
    for original in manifest_folds:
        fold = int(original["fold"])
        frozen_val = sorted(set(_ids(original["val_ids"])) - stage1_quarantine)
        val_ids = sorted(set(frozen_val) & scorable)
        train_ids = sorted(training - set(val_ids))
        if not val_ids:
            raise RuntimeError(f"Fold {fold} has no currently available OOF validation cases")
        if set(train_ids) & set(val_ids):
            raise RuntimeError(f"Fold {fold}: train/validation leakage")
        specs.append({"fold": fold, "train_ids": train_ids, "val_ids": val_ids})
        stage1_validation_ids[fold] = frozen_val
        held_out.extend(val_ids)

    fold_ids = sorted(int(spec["fold"]) for spec in specs)
    if fold_ids != list(range(5)):
        raise RuntimeError("Frozen source manifest must contain folds 0..4 exactly once")
    if sorted(held_out) != sorted(scorable) or len(held_out) != len(set(held_out)):
        raise RuntimeError("Projected folds do not cover every usable OOF case exactly once")
    return specs, stage1_validation_ids


def validate_stage2_cv_summary(summary, audit, *, expected_live):
    """Require the correct completed Stage-2 CV summary for an audited cohort."""

    # Preserve the previously locked Final91 original46 result as a valid recipe/epoch
    # source for the established Final136 model. New Final136 robustness runs may instead
    # provide the projected 42-case summary validated below.
    if summary.get("complete_original46_qc", False):
        return

    missing = _ids(audit.get("missing_frozen_case_ids", []))
    if not missing:
        raise RuntimeError("Complete Stage-2 original46-QC OOF comparison is required")

    approved = sorted(FINAL136_ALLOWED_MISSING_FROZEN_IDS)
    if (
        int(expected_live) != EXPECTED_FINAL136
        or not audit.get("missing_frozen_allowed", False)
        or missing != approved
    ):
        raise RuntimeError("Only the audited Final136 missing-four cohort is supported")
    if summary.get("external31_access") is not False:
        raise RuntimeError("Final136 Stage-2 CV summary must record External31 access as false")
    if not summary.get("complete_available_frozen_source_oof", False):
        raise RuntimeError("Complete Stage-2 OOF for all available frozen-source cases is required")
    if _ids(summary.get("missing_frozen_case_ids", [])) != approved:
        raise RuntimeError("Stage-2 CV and Final136 audit missing-frozen provenance differ")
    if int(summary.get("n_live_human_gold", -1)) != EXPECTED_FINAL136:
        raise RuntimeError("Stage-2 CV summary is not from the Final136 cohort")
    if int(summary.get("n_available_frozen_source_scorable", -1)) != 42:
        raise RuntimeError("Final136 Stage-2 CV must score the 42 available original-QC cases")
    if sorted(int(value) for value in summary.get("completed_folds", [])) != list(range(5)):
        raise RuntimeError("Final136 Stage-2 CV does not contain all five folds")
