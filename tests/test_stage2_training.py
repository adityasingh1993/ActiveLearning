import pytest
import torch
from monai.transforms import Compose

import hassl.data.stage2_training as stage2


def base_recipe():
    return {
        "version": "legacy",
        "crop_size": [128, 128, 128],
        "training_crop": {
            "source": "GT bounding box only",
            "margin_per_side_range": [0.40, 0.60],
            "center_jitter_fraction_of_gt_size": 0.10,
        },
        "augmentation_after_crop": {
            "translation_voxels": 4.0,
            "translation_probability": 0.5,
            "lr_flip_probability": 0.5,
        },
    }


def test_legacy_options_do_not_change_recipe():
    recipe = base_recipe()
    options = stage2.Stage2TrainingOptions()
    assert not options.robust_enabled
    assert stage2.add_training_recipe_extensions(recipe, options) == recipe


def test_robust_options_round_trip_through_recipe():
    options = stage2.Stage2TrainingOptions(
        wide_crop_probability=0.35,
        wide_margin_min=0.60,
        wide_margin_max=1.00,
        wide_center_jitter=0.20,
        small_bladder_quantile=1.0 / 3.0,
        small_bladder_weight=2.0,
        mild_appearance_augmentation=True,
    )
    recipe = stage2.add_training_recipe_extensions(base_recipe(), options)
    assert stage2.options_from_recipe(recipe) == options
    assert recipe["training_sampler"]["validation_labels_used"] is False
    assert recipe["training_crop"]["wide_crop_mixture"]["probability"] == 0.35
    assert "mild_ultrasound_appearance" in recipe["augmentation_after_crop"]


def test_wide_crop_is_gt_safe_and_records_occupancy():
    image = torch.zeros((1, 48, 48, 48), dtype=torch.float32)
    label = torch.zeros_like(image)
    label[:, 20:28, 18:30, 22:27] = 1.0
    transform = stage2.SafeMixedGTBoxCropd(stage2.Stage2TrainingOptions(
        output_size=32,
        wide_crop_probability=1.0,
        wide_margin_min=1.0,
        wide_margin_max=1.0,
        wide_center_jitter=0.20,
    ))
    transform.set_random_state(seed=17)
    result = transform({"id": "case", "image": image, "label": label})

    assert tuple(result["image"].shape) == (1, 32, 32, 32)
    assert tuple(result["label"].shape) == (1, 32, 32, 32)
    assert int(result["training_crop_is_wide"].item()) == 1
    assert float(result["training_crop_margin"].item()) == pytest.approx(1.0)
    assert 0.0 < float(result["training_crop_gt_fraction"].item()) < 1.0
    assert int(result["label"].sum().item()) > 0


def test_small_bladder_plan_uses_ordered_training_cases_only():
    case_ids = ["a", "b", "c", "d", "e", "f"]
    volumes = {case_id: float(index) for index, case_id in enumerate(case_ids, start=1)}
    weights, audit = stage2.small_bladder_sampling_plan(
        case_ids, volumes, quantile=1.0 / 3.0, small_weight=2.0
    )

    assert audit["small_training_ids"] == ["a", "b"]
    assert weights == [2.0, 2.0, 1.0, 1.0, 1.0, 1.0]
    assert audit["expected_small_draw_fraction"] == pytest.approx(0.5)
    assert audit["validation_labels_used"] is False


def test_small_bladder_sampler_preserves_epoch_length(monkeypatch):
    items = [
        {"id": case_id, "image": f"{case_id}.mha", "label": f"{case_id}.seg.nrrd"}
        for case_id in ("a", "b", "c", "d", "e", "f")
    ]
    volume_by_label = {
        f"{case_id}.seg.nrrd": float(index)
        for index, case_id in enumerate(("a", "b", "c", "d", "e", "f"), start=1)
    }
    monkeypatch.setattr(
        stage2, "_native_bladder_ml", lambda label, _image: volume_by_label[label]
    )
    options = stage2.Stage2TrainingOptions(small_bladder_weight=2.0)
    sampler, audit = stage2.build_small_bladder_sampler(items, options, seed=123)

    assert sampler.num_samples == len(items)
    assert len(list(iter(sampler))) == len(items)
    assert audit["steps_per_epoch_preserved"] is True
    assert audit["sampler_seed"] == 123


def test_mild_appearance_transforms_are_opt_in():
    legacy = stage2.build_stage2_training_transform(
        Compose([]), stage2.Stage2TrainingOptions(output_size=32)
    )
    robust = stage2.build_stage2_training_transform(
        Compose([]), stage2.Stage2TrainingOptions(
            output_size=32, mild_appearance_augmentation=True
        )
    )
    legacy_names = [type(transform).__name__ for transform in legacy.transforms]
    robust_names = [type(transform).__name__ for transform in robust.transforms]

    assert "RandMultiplicativeSpeckleNoised" not in legacy_names
    assert "RandMultiplicativeSpeckleNoised" in robust_names
    assert "RandAdjustContrastd" in robust_names
    assert "RandScaleIntensityd" in robust_names
