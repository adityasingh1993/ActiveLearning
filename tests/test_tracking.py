import pytest
from hassl.tracking import ExperimentTracker

def test_none_backend(tmp_path):
    tracker = ExperimentTracker(backend='none', log_dir=str(tmp_path))
    # Should not crash
    tracker.log_metric("loss", 0.5, step=1)
    tracker.log_metrics({"acc": 0.9, "val_loss": 0.4}, step=1)

def test_log_metrics(tmp_path):
    tracker = ExperimentTracker(backend='none', log_dir=str(tmp_path))
    tracker.log_metrics({"a": 1, "b": 2}, step=0)
    # Just verifying it doesn't crash

def test_context_manager(tmp_path):
    with ExperimentTracker(backend='none', log_dir=str(tmp_path)) as tracker:
        tracker.log_metric("test", 1.0)
    # Exited context cleanly
