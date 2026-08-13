import pytest
from hassl.tracking import ExperimentTracker


def test_none_backend():
    tracker = ExperimentTracker(backend='none')
    # Should not crash
    tracker.log_metrics({"loss": 0.5}, step=1)
    tracker.log_metrics({"acc": 0.9, "val_loss": 0.4}, step=1)


def test_log_metrics():
    tracker = ExperimentTracker(backend='none')
    tracker.log_metrics({"a": 1, "b": 2}, step=0)
    # Just verifying it doesn't crash


def test_context_manager():
    with ExperimentTracker(backend='none') as tracker:
        tracker.log_metrics({"test": 1.0}, step=0)
    # Exited context cleanly
