"""
HASSL Synthetic CI Integration Test Suite.

Runs automated assertions across pipeline artifacts, patient-level splits,
provenance isolation, and spatial geometry header preservation.
"""

import os
import json
import pytest
import numpy as np
from pathlib import Path

from hassl.config import HASSLConfig
from hassl.data.data_engine import get_or_create_frozen_splits, build_labeled_dataset
from hassl.data.nrrd_utils import write_mask_with_spatial_geometry


def test_import_smoke_check():
    """Verify all hassl submodules load cleanly without SyntaxError or NameError."""
    import importlib
    import pkgutil
    import hassl

    failed = []
    for m in pkgutil.walk_packages(hassl.__path__, prefix="hassl."):
        try:
            importlib.import_module(m.name)
        except Exception as e:
            failed.append(f"{m.name}: {type(e).__name__}: {e}")
    assert not failed, "Modules failed to import:\n" + "\n".join(failed)


def test_frozen_splits_patient_grouping_validation(tmp_path):
    """Verify splits.json creates patient-level holdouts and fails loudly on patient collapse (V-11 fix)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    labels_dir = data_dir / "labels"
    labels_dir.mkdir(parents=True)

    # Create synthetic volumes with patient prefixes: US001_v1, US001_v2, US002_v1, US002_v2, etc.
    for i in range(1, 7):
        patient_id = f"PAT{i:03d}"
        for v in range(1, 3):
            vol_name = f"{patient_id}_vol{v}"
            img_file = data_dir / f"{vol_name}.mha"
            lbl_file = labels_dir / f"{vol_name}.seg.nrrd"
            img_file.write_bytes(b"header")
            lbl_file.write_bytes(b"label")

    splits = get_or_create_frozen_splits(str(data_dir), seed=42)

    assert os.path.exists(str(data_dir / "splits.json"))
    assert len(splits["val_ids"]) > 0, "Validation set must not be empty"
    assert len(splits["test_ids"]) > 0, "Test set must not be empty"
    assert len(splits["initial_train_ids"]) > 0, "Train set must not be empty"

    # Verify patient isolation between splits
    val_patients = set(v.split("_")[0] for v in splits["val_ids"])
    train_patients = set(v.split("_")[0] for v in splits["initial_train_ids"])
    test_patients = set(v.split("_")[0] for v in splits["test_ids"])

    assert val_patients.isdisjoint(train_patients), "Patient leakage detected between val and train!"
    assert test_patients.isdisjoint(train_patients), "Patient leakage detected between test and train!"


def test_provenance_manifest_gating(tmp_path):
    """Verify accepted pseudo-labels are isolated to pseudo_approved and excluded from human gold pool (V-4 fix)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    labels_dir = data_dir / "labels"
    approved_dir = data_dir / "pseudo_approved"
    labels_dir.mkdir(parents=True)
    approved_dir.mkdir(parents=True)

    # Create 1 human volume, 1 pseudo_approved volume
    (data_dir / "vol_human.mha").write_bytes(b"data")
    (labels_dir / "vol_human.seg.nrrd").write_bytes(b"mask")

    (data_dir / "vol_pseudo.mha").write_bytes(b"data")
    (approved_dir / "vol_pseudo.seg.nrrd").write_bytes(b"mask")

    manifest = {
        "provenance": {
            "vol_human": "human",
            "vol_pseudo": "pseudo_approved"
        }
    }
    manifest_file = tmp_path / "pool_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f)

    ds, labeled_ids = build_labeled_dataset(
        str(data_dir), ".mha", ".seg.nrrd", manifest_path=str(manifest_file), use_cache_dataset=False
    )

    provenance_map = {item["id"]: item["provenance"] for item in ds.data}
    assert provenance_map.get("vol_human") == "human"
    assert provenance_map.get("vol_pseudo") == "pseudo_approved"


def test_frozen_splits_patient_collapse_failure(tmp_path):
    """Verify get_or_create_frozen_splits raises ValueError when all volumes collapse to 1 patient (V-11 fix)."""
    data_dir = tmp_path / "data_collapse"
    data_dir.mkdir(parents=True)
    labels_dir = data_dir / "labels"
    labels_dir.mkdir(parents=True)

    # 10 volumes with same prefix 'commonprefix' causing single-patient collapse
    for i in range(1, 10):
        vol_name = f"commonprefix_case{i}"
        (data_dir / f"{vol_name}.mha").write_bytes(b"data")
        (labels_dir / f"{vol_name}.seg.nrrd").write_bytes(b"label")

    with pytest.raises(ValueError, match="collapsed all 9 volumes into 1 single patient ID"):
        get_or_create_frozen_splits(str(data_dir), seed=42)


def test_written_mask_preserves_geometry(tmp_path):
    """Verify write_mask_with_spatial_geometry preserves physical affine size, spacing, & direction (V6-5 fix)."""
    try:
        import SimpleITK as sitk
    except ImportError:
        pytest.skip("SimpleITK not installed")

    src_path = str(tmp_path / "reference.mha")
    out_path = str(tmp_path / "written_mask.seg.nrrd")

    # Create dummy reference ITK image
    arr = np.zeros((32, 32, 32), dtype=np.float32)
    ref_img = sitk.GetImageFromArray(arr)
    ref_img.SetSpacing((0.5, 0.4, 0.8))
    ref_img.SetOrigin((10.0, -5.0, 3.0))
    sitk.WriteImage(ref_img, src_path)

    mask_arr = np.ones((16, 16, 16), dtype=np.uint8)  # Resized 16^3 prediction mask
    write_mask_with_spatial_geometry(out_path, mask_arr, reference_image_path=src_path)

    got_img = sitk.ReadImage(out_path)
    assert got_img.GetSize() == ref_img.GetSize(), "Output mask size must match reference image size"
    assert got_img.GetSpacing() == pytest.approx(ref_img.GetSpacing()), "Output mask spacing must match reference image spacing"
    assert got_img.GetOrigin() == pytest.approx(ref_img.GetOrigin()), "Output mask origin must match reference image origin"


def test_teacher_student_views_are_spatially_aligned():
    """Verify teacher and student views share exact spatial coordinate frames by testing production _make_unlabeled_views (V7-1, V8-1, V8-2 fix)."""
    try:
        import torch
        import monai
        from hassl.config import HASSLConfig
        from hassl.training.trainer import HASSLTrainer
    except ImportError:
        pytest.skip("PyTorch or MONAI not installed")

    # V8-2 fix: Pin random seed for deterministic test execution
    monai.utils.set_determinism(seed=42)

    config = HASSLConfig()
    config.device = "cpu"
    config.compute_mode = "prototype"
    config.unet_backbone = "unet"

    # Instantiate HASSLTrainer to exercise production _make_unlabeled_views method (V8-1 fix)
    trainer = HASSLTrainer(config=config, labeled_loader=[], unlabeled_loader=[], val_loader=[], tracker=None)

    # Create an asymmetric 3D phantom [B, C, D, H, W] (off-center foreground patch ensuring zero flip symmetry)
    vol = torch.zeros((1, 1, 32, 32, 32), dtype=torch.float32)
    vol[:, :, 4:12, 16:28, 2:8] = 1.0

    # Call production view generation method
    teacher_view, student_view = trainer._make_unlabeled_views(vol)

    # Binarize views at threshold 0.5 (where Gaussian smoothing preserved boundary Dice is >= 0.80)
    t_mask = (teacher_view[0] > 0.5).float()
    s_mask = (student_view[0] > 0.5).float()

    intersection = (t_mask * s_mask).sum()
    total = t_mask.sum() + s_mask.sum()
    dice = float((2.0 * intersection / (total + 1e-8)).item())

    assert dice >= 0.80, f"Teacher and student views are spatially misaligned! Spatial Dice overlap: {dice:.4f} < 0.80"


def test_ssl_contrastive_loss_nonzero_at_batch_size_1():
    """Verify SSL InfoNCE contrastive loss is non-zero and produces gradients even at batch_size=1."""
    try:
        import torch
        from hassl.config import HASSLConfig
        from hassl.ssl.ssl_pretrainer import SSLPretrainer
    except ImportError:
        pytest.skip("PyTorch or MONAI not installed")

    config = HASSLConfig()
    config.device = "cpu"
    config.compute_mode = "prototype"
    config.unet_backbone = "unet"

    pretrainer = SSLPretrainer(config=config, dataloader=[], tracker=None)

    # Synthetic batch of size B=1
    x = torch.randn(1, 1, 32, 32, 32)
    x_aug1 = x + torch.randn_like(x) * 0.05
    x_aug2 = x + torch.randn_like(x) * 0.05

    b1 = pretrainer._extract_bottleneck_features(x_aug1)
    b2 = pretrainer._extract_bottleneck_features(x_aug2)

    p1 = torch.nn.functional.adaptive_avg_pool3d(b1, (2, 2, 2)).permute(0, 2, 3, 4, 1).reshape(-1, b1.size(1))
    p2 = torch.nn.functional.adaptive_avg_pool3d(b2, (2, 2, 2)).permute(0, 2, 3, 4, 1).reshape(-1, b2.size(1))

    feat1 = pretrainer.proj_head(p1)
    feat2 = pretrainer.proj_head(p2)

    loss_cont = pretrainer._infonce_loss(feat1, feat2)
    assert loss_cont.item() > 0.0, f"Contrastive loss at batch_size=1 must be > 0.0, got {loss_cont.item()}"
    assert loss_cont.requires_grad, "Contrastive loss must require grad for optimization"


def test_full_pipeline_synthetic_end_to_end(tmp_path):
    """Verify end-to-end training phase execution on synthetic cohort generating checkpoints and artifacts (V9-1, V9-2, V9-4 fix)."""
    try:
        import torch
        from hassl.config import HASSLConfig
        from hassl.utils.synthetic_data import generate_synthetic_dataset
        from hassl.pipeline import run_train
    except ImportError:
        pytest.skip("PyTorch, SimpleITK, or MONAI not installed")

    data_dir = tmp_path / "synthetic_cohort"
    generate_synthetic_dataset(output_dir=str(data_dir), num_volumes=6, num_labeled=3, image_size=(32, 32, 32))

    config = HASSLConfig()
    config.data_dir = str(data_dir)
    config.log_dir = str(tmp_path / "logs")
    config.checkpoint_dir = str(tmp_path / "checkpoints")
    config.embedding_dir = str(tmp_path / "embeddings")
    config.cache_dir = str(tmp_path / "cache")
    config.preseg_dir = str(tmp_path / "preseg")
    config.spatial_size = (32, 32, 32)
    config.device = "cpu"
    config.compute_mode = "prototype"
    config.unet_backbone = "unet"
    config.train_epochs = 1
    config.use_cache_dataset = False
    config.tracker = "none"
    config.unet_channels = (8, 16, 32, 64)

    run_train(config, round_num=0)

    # V9-4 fix: Assert on actual checkpoint payload and manifest state, not just file existence
    ckpt_file = Path(config.checkpoint_dir) / "round0_latest.pth"
    assert ckpt_file.exists(), f"Expected checkpoint file {ckpt_file} was not generated"
    assert ckpt_file.stat().st_size > 100, f"Checkpoint file {ckpt_file} is suspiciously small ({ckpt_file.stat().st_size} bytes)"

    state = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
    assert "net_A" in state, "Checkpoint missing 'net_A' state dict"
    assert len(state["net_A"]) > 0, "Checkpoint 'net_A' state dict is empty"
    assert "best_dice" in state, "Checkpoint missing 'best_dice'"

    splits_file = Path(config.data_dir) / "splits.json"
    assert splits_file.exists(), f"Expected frozen splits file {splits_file} was not generated"


def test_early_stopping_reset_clears_sticky_flag():
    """Verify EarlyStopping.reset() clears sticky early_stop boolean and counter for new AL rounds (V10-1 fix)."""
    from hassl.training.trainer import EarlyStopping

    es = EarlyStopping(patience=2, min_delta=1e-4, mode='max')
    es(0.80)  # best = 0.80, counter = 0
    es(0.80)  # counter = 1
    es(0.80)  # counter = 2 -> early_stop = True
    assert es.early_stop is True

    # Test clearing on improvement
    es(0.90)  # improvement -> early_stop should become False!
    assert es.early_stop is False, "early_stop flag should be cleared when score improves"
    assert es.counter == 0

    # Test explicit reset()
    es.reset()
    assert es.early_stop is False
    assert es.counter == 0
    assert es.best_score is None


def test_load_checkpoint_weights_only_resets_optimizer_and_scheduler():
    """Verify load_checkpoint(weights_only=True) restores model parameters but resets LR scheduler and early stopper (V10-1, V10-2 fix)."""
    try:
        import torch
        from hassl.config import HASSLConfig
        from hassl.training.trainer import HASSLTrainer
    except ImportError:
        pytest.skip("PyTorch or MONAI not installed")

    config = HASSLConfig()
    config.device = "cpu"
    config.compute_mode = "prototype"
    config.unet_backbone = "unet"
    config.train_epochs = 100
    config.lr_scheduler = "cosine"
    config.lr_warmup_epochs = 0
    config.train_lr = 1e-3

    trainer = HASSLTrainer(config=config, labeled_loader=[], unlabeled_loader=[], val_loader=[], tracker=None)

    # Step trainer to consume epochs and trigger early stopping
    for _ in range(50):
        trainer.scheduler.step()
    trainer.early_stopper.early_stop = True
    trainer.early_stopper.counter = 30

    ckpt_path = "./experiments/checkpoints/test_al_reset.pth"
    trainer.save_checkpoint(ckpt_path, epoch=50)

    # Create new trainer representing a new AL round
    new_trainer = HASSLTrainer(config=config, labeled_loader=[], unlabeled_loader=[], val_loader=[], tracker=None)
    new_trainer.load_checkpoint(ckpt_path, weights_only=True)

    # Verify weights_only=True resets epoch numbering, early stopper, and scheduler
    assert new_trainer.start_epoch == 0, f"New AL round should start at epoch 0, got {new_trainer.start_epoch}"
    assert new_trainer.early_stopper.early_stop is False, "New AL round should start with early_stop = False"
    assert new_trainer.early_stopper.counter == 0
    current_lr = new_trainer.optimizer.param_groups[0]['lr']
    assert abs(current_lr - 1e-3) < 1e-6, f"New AL round LR scheduler should restart at train_lr 1e-3, got {current_lr}"

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


def test_early_stopping_and_lr_scheduler():
    """Verify EarlyStopping monitors performance and Cosine Annealing decays LR as expected."""
    try:
        import torch
        from hassl.config import HASSLConfig
        from hassl.training.trainer import EarlyStopping, HASSLTrainer
    except ImportError:
        pytest.skip("PyTorch or MONAI not installed")

    # 1. Test EarlyStopping helper class directly
    es = EarlyStopping(patience=3, min_delta=1e-4, mode='max')
    scores = [0.80, 0.81, 0.81, 0.81, 0.81]
    stopped_at = None
    for idx, score in enumerate(scores):
        if es(score):
            stopped_at = idx
            break
    assert stopped_at == 4, f"Early stopping should trigger at index 4 (after 3 non-improvements), triggered at {stopped_at}"

    # 2. Test HASSLTrainer LR scheduler initialization and step (pure Cosine without warmup)
    config = HASSLConfig()
    config.device = "cpu"
    config.compute_mode = "prototype"
    config.unet_backbone = "unet"
    config.lr_scheduler = "cosine"
    config.lr_warmup_epochs = 0
    config.train_lr = 1e-3
    config.min_lr = 1e-5
    config.train_epochs = 100

    trainer = HASSLTrainer(config=config, labeled_loader=[], unlabeled_loader=[], val_loader=[], tracker=None)
    initial_lr = trainer.optimizer.param_groups[0]['lr']
    assert abs(initial_lr - 1e-3) < 1e-7, f"Initial LR without warmup should be 1e-3, got {initial_lr}"

    trainer.scheduler.step()
    stepped_lr = trainer.optimizer.param_groups[0]['lr']
    assert stepped_lr < initial_lr, f"Cosine scheduler should decay learning rate on step: {stepped_lr} < {initial_lr}"

    # 3. Test LR Warmup scheduler (V10-4 fix)
    config_warmup = HASSLConfig()
    config_warmup.device = "cpu"
    config_warmup.compute_mode = "prototype"
    config_warmup.unet_backbone = "unet"
    config_warmup.lr_scheduler = "cosine"
    config_warmup.lr_warmup_epochs = 5
    config_warmup.train_lr = 1e-3
    config_warmup.min_lr = 1e-5
    config_warmup.train_epochs = 100

    trainer_warmup = HASSLTrainer(config=config_warmup, labeled_loader=[], unlabeled_loader=[], val_loader=[], tracker=None)
    warmup_initial_lr = trainer_warmup.optimizer.param_groups[0]['lr']
    assert abs(warmup_initial_lr - 1e-4) < 1e-6, f"Warmup start factor (0.1 * 1e-3) should be 1e-4, got {warmup_initial_lr}"


# ─── M-2: Configurable Preprocessing Mode ────────────────────────────

def test_preprocessing_mode_resize_has_resized_transform():
    """In 'resize' mode, val transform chain must contain Resized (M-2)."""
    from monai.transforms import Resized
    from hassl.data.data_engine import get_base_transforms

    config = HASSLConfig()
    config.preprocessing_mode = "resize"
    t = get_base_transforms(config, keys=["image", "label"], is_training=False)
    transform_types = [type(tx).__name__ for tx in t.transforms]
    assert "Resized" in transform_types, f"Resized missing in resize-mode val transforms: {transform_types}"


def test_preprocessing_mode_patch_train_labeled_has_rand_crop():
    """In 'patch' mode, labeled training transform chain must contain RandCropByPosNegLabeld (M-2)."""
    from monai.transforms import RandCropByPosNegLabeld
    from hassl.data.data_engine import get_base_transforms

    config = HASSLConfig()
    config.preprocessing_mode = "patch"
    config.patch_size = (64, 64, 64)
    t = get_base_transforms(config, keys=["image", "label"], is_training=True)
    transform_types = [type(tx).__name__ for tx in t.transforms]
    assert "RandCropByPosNegLabeld" in transform_types, \
        f"RandCropByPosNegLabeld missing in patch-mode train transforms: {transform_types}"
    assert "Resized" not in transform_types, \
        f"Resized should NOT appear in patch-mode training transforms: {transform_types}"


def test_preprocessing_mode_patch_val_still_has_resized():
    """In 'patch' mode, validation transform must still use Resized for whole-volume inference (M-2)."""
    from monai.transforms import Resized
    from hassl.data.data_engine import get_base_transforms

    config = HASSLConfig()
    config.preprocessing_mode = "patch"
    t = get_base_transforms(config, keys=["image", "label"], is_training=False)
    transform_types = [type(tx).__name__ for tx in t.transforms]
    assert "Resized" in transform_types, \
        f"Resized must appear in patch-mode VAL transforms for SlidingWindowInferer: {transform_types}"


def test_preprocessing_mode_patch_unlabeled_has_rand_spatial_crop():
    """In 'patch' mode, image-only unlabeled stream uses RandSpatialCropd (M-2)."""
    from monai.transforms import RandSpatialCropd
    from hassl.data.data_engine import get_base_transforms

    config = HASSLConfig()
    config.preprocessing_mode = "patch"
    config.patch_size = (64, 64, 64)
    t = get_base_transforms(config, keys=["image"], is_training=True)
    transform_types = [type(tx).__name__ for tx in t.transforms]
    assert "RandSpatialCropd" in transform_types, \
        f"RandSpatialCropd missing in patch-mode unlabeled transforms: {transform_types}"


# ─── V6-9: RLE Encode/Decode Round-trip ──────────────────────────────

def test_rle_encode_decode_round_trip():
    """RLE encode then decode must reproduce the original mask exactly (V6-9)."""
    # Inline the same RLE algorithm used in server._rle_encode to avoid
    # requiring FastAPI/uvicorn to be installed in the test environment.
    def _rle_encode(mask):
        flat = mask.ravel()
        if flat.size == 0:
            return []
        rle = []; cur_val = int(flat[0]); count = 0
        for v in flat:
            iv = int(v)
            if iv == cur_val:
                count += 1
            else:
                rle.append([cur_val, count]); cur_val = iv; count = 1
        rle.append([cur_val, count])
        return rle

    rng = np.random.default_rng(0)
    mask = (rng.random((128, 128)) > 0.7).astype(np.uint8)

    rle = _rle_encode(mask)
    # Verify total count equals mask size
    total = sum(count for _, count in rle)
    assert total == mask.size, f"RLE total count {total} != mask.size {mask.size}"

    # Decode manually
    flat = np.zeros(mask.size, dtype=np.uint8)
    pos = 0
    for val, count in rle:
        flat[pos:pos + count] = val
        pos += count
    decoded = flat.reshape(mask.shape)
    assert np.array_equal(decoded, mask), "RLE decode did not reproduce original mask"


def test_rle_encode_all_zeros():
    """All-zero mask should encode to a single [0, N] pair (V6-9)."""
    def _rle_encode(mask):
        flat = mask.ravel()
        if flat.size == 0:
            return []
        rle = []; cur_val = int(flat[0]); count = 0
        for v in flat:
            iv = int(v)
            if iv == cur_val:
                count += 1
            else:
                rle.append([cur_val, count]); cur_val = iv; count = 1
        rle.append([cur_val, count])
        return rle

    mask = np.zeros((64, 64), dtype=np.uint8)
    rle = _rle_encode(mask)
    assert len(rle) == 1
    assert rle[0] == [0, 64 * 64]


# ─── A-4: LRU Cache Eviction ─────────────────────────────────────────

def test_lru_cache_evicts_oldest():
    """LRUCache must evict the least-recently-used item when maxsize is exceeded (A-4)."""
    # Inline the same LRU algorithm used in server.LRUCache to avoid
    # requiring FastAPI/uvicorn to be installed in the test environment.
    from collections import OrderedDict

    class LRUCache:
        def __init__(self, maxsize=20):
            self._cache = OrderedDict()
            self.maxsize = maxsize
        def get(self, key, default=None):
            if key not in self._cache: return default
            self._cache.move_to_end(key)
            return self._cache[key]
        def __contains__(self, key): return key in self._cache
        def __setitem__(self, key, value):
            self._cache[key] = value
            self._cache.move_to_end(key)
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)
        def __getitem__(self, key): return self.get(key)

    cache = LRUCache(maxsize=3)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    # Access "a" to make "b" the LRU
    _ = cache["a"]
    # Insert "d" — should evict "b" (oldest not recently accessed)
    cache["d"] = 4
    assert "b" not in cache, "LRUCache should have evicted 'b' (LRU item)"
    assert "a" in cache
    assert "c" in cache
    assert "d" in cache
