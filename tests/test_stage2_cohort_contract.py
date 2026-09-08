import unittest

from scripts.stage2_cohort_contract import (
    FINAL136_ALLOWED_MISSING_FROZEN_IDS,
    FINAL136_TRAINING_SCOPE,
    PRIOR_QUARANTINE_ID,
    project_frozen_oof_folds,
    resolve_stage2_cohort,
    validate_stage2_cv_summary,
)


def final136_fixture():
    retained_source = [f"source-{index:02d}" for index in range(42)]
    source_ids = sorted(
        retained_source
        + [PRIOR_QUARANTINE_ID]
        + list(FINAL136_ALLOWED_MISSING_FROZEN_IDS)
    )
    current_ids = sorted(retained_source + [PRIOR_QUARANTINE_ID] + [
        f"extra-{index:02d}" for index in range(93)
    ])
    audit = {
        "all_current_human_label_ids": current_ids,
        "training_case_ids": current_ids,
        "excluded_training_case_ids": [],
        "quarantined_case_ids": [],
        "intentionally_included_prior_quarantine_ids": [PRIOR_QUARANTINE_ID],
        "all_training_labels_passed_audit": True,
        "training_scope_provenance_enforced": True,
        "training_scope": FINAL136_TRAINING_SCOPE,
        "missing_frozen_allowed": True,
        "missing_frozen_case_ids": sorted(FINAL136_ALLOWED_MISSING_FROZEN_IDS),
        "allowed_missing_frozen_case_ids": sorted(FINAL136_ALLOWED_MISSING_FROZEN_IDS),
        "external31_access": False,
    }
    return source_ids, current_ids, audit


class Stage2CohortContractTests(unittest.TestCase):
    def test_final136_uses_all_current_cases_and_scores_available_frozen_cases(self):
        source_ids, current_ids, audit = final136_fixture()
        cohort = resolve_stage2_cohort(
            audit,
            source_ids,
            current_ids,
            expected_live=136,
            expected_trainable=136,
            include_quarantined=True,
            allow_missing_frozen=True,
        )

        self.assertEqual(len(cohort["training_ids"]), 136)
        self.assertEqual(len(cohort["scorable_ids"]), 42)
        self.assertEqual(len(cohort["extra_ids"]), 93)
        self.assertEqual(len(cohort["train_only_ids"]), 94)
        self.assertIn(PRIOR_QUARANTINE_ID, cohort["training_ids"])
        self.assertNotIn(PRIOR_QUARANTINE_ID, cohort["scorable_ids"])
        self.assertEqual(
            cohort["missing_frozen_case_ids"],
            sorted(FINAL136_ALLOWED_MISSING_FROZEN_IDS),
        )

    def test_final136_rejects_missing_frozen_without_explicit_flags(self):
        source_ids, current_ids, audit = final136_fixture()
        with self.assertRaisesRegex(RuntimeError, "policy differs"):
            resolve_stage2_cohort(
                audit,
                source_ids,
                current_ids,
                expected_live=136,
                include_quarantined=True,
                allow_missing_frozen=False,
            )

    def test_projected_folds_train_on_136_minus_each_available_validation_subset(self):
        source_ids, current_ids, audit = final136_fixture()
        cohort = resolve_stage2_cohort(
            audit,
            source_ids,
            current_ids,
            expected_live=136,
            expected_trainable=136,
            include_quarantined=True,
            allow_missing_frozen=True,
        )
        folds = []
        for fold in range(5):
            val_ids = source_ids[fold::5]
            folds.append({
                "fold": fold,
                "val_ids": val_ids,
                "train_ids": sorted(set(source_ids) - set(val_ids)),
            })

        specs, checkpoint_val_ids = project_frozen_oof_folds(
            folds,
            cohort["training_ids"],
            cohort["scorable_ids"],
            cohort["stage1_quarantined_case_ids"],
        )

        self.assertEqual(sum(len(spec["val_ids"]) for spec in specs), 42)
        self.assertEqual(
            sorted({case_id for spec in specs for case_id in spec["val_ids"]}),
            cohort["scorable_ids"],
        )
        for spec in specs:
            self.assertEqual(len(spec["train_ids"]), 136 - len(spec["val_ids"]))
            self.assertIn(PRIOR_QUARANTINE_ID, spec["train_ids"])
            self.assertEqual(
                checkpoint_val_ids[spec["fold"]],
                sorted(set(folds[spec["fold"]]["val_ids"]) - {PRIOR_QUARANTINE_ID}),
            )

    def test_final91_contract_is_unchanged(self):
        retained_source = [f"source-{index:02d}" for index in range(46)]
        source_ids = sorted(retained_source + [PRIOR_QUARANTINE_ID])
        current_ids = sorted(source_ids + [f"extra-{index:02d}" for index in range(44)])
        audit = {
            "all_current_human_label_ids": current_ids,
            "all_visible_labels_passed_audit": True,
            "selection_provenance_enforced": True,
            "external31_access": False,
        }

        cohort = resolve_stage2_cohort(
            audit,
            source_ids,
            current_ids,
            expected_live=91,
        )

        self.assertEqual(len(cohort["training_ids"]), 90)
        self.assertEqual(len(cohort["scorable_ids"]), 46)
        self.assertEqual(len(cohort["extra_ids"]), 44)
        self.assertEqual(cohort["training_quarantined_case_ids"], [PRIOR_QUARANTINE_ID])

    def test_final136_projected_summary_is_accepted(self):
        _, _, audit = final136_fixture()
        summary = {
            "complete_original46_qc": False,
            "complete_available_frozen_source_oof": True,
            "missing_frozen_case_ids": sorted(FINAL136_ALLOWED_MISSING_FROZEN_IDS),
            "n_live_human_gold": 136,
            "n_available_frozen_source_scorable": 42,
            "completed_folds": [0, 1, 2, 3, 4],
            "external31_access": False,
        }
        validate_stage2_cv_summary(summary, audit, expected_live=136)

    def test_legacy_complete_original46_summary_remains_accepted(self):
        _, _, audit = final136_fixture()
        validate_stage2_cv_summary(
            {"complete_original46_qc": True},
            audit,
            expected_live=136,
        )


if __name__ == "__main__":
    unittest.main()
