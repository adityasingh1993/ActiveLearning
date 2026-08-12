"""
HASSL Experiment Tracking Module.

Provides a unified abstraction layer for experiment tracking,
supporting both WandB and MLflow backends with graceful fallback.

Usage:
    tracker = ExperimentTracker(backend="wandb", project="hassl", run_name="round0")
    tracker.log_config(config.to_dict())
    tracker.log_metrics({"dice": 0.85, "loss": 0.23}, step=10)
    tracker.log_artifact("model.pth", name="best_model")
    tracker.finish()

    # Or as context manager:
    with ExperimentTracker(backend="mlflow", project="hassl") as tracker:
        tracker.log_metrics({"loss": 0.1}, step=0)
"""

import logging
from typing import Any, Dict, Optional, Union

import numpy as np


logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Unified experiment tracking interface for WandB and MLflow.

    Gracefully falls back to console logging if the chosen backend
    is not installed.

    Args:
        backend: Tracking backend - "wandb", "mlflow", or "none".
        project: Project/experiment group name.
        run_name: Optional name for this specific run.
        config: Optional config dict to log at initialization.
    """

    def __init__(
        self,
        backend: str = "wandb",
        project: str = "hassl",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tracking_uri: Optional[str] = None,
    ):
        self.backend = backend
        self.project = project
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self._run = None

        if self.backend == "wandb":
            try:
                import wandb
                self._run = wandb.init(
                    project=project,
                    name=run_name,
                    config=config,
                    reinit=True,
                )
            except ImportError:
                logger.warning("wandb not installed. Falling back to console logging.")
                self.backend = "none"
            except Exception as e:
                logger.warning(f"wandb init failed: {e}. Falling back to console logging.")
                self.backend = "none"

        elif self.backend == "mlflow":
            try:
                import mlflow
                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(project)
                self._run = mlflow.start_run(run_name=run_name)
                if config:
                    # MLflow params must be strings and have length limits
                    flat_config = {k: str(v)[:250] for k, v in config.items()}
                    mlflow.log_params(flat_config)
            except ImportError:
                logger.warning("mlflow not installed. Falling back to console logging.")
                self.backend = "none"
            except Exception as e:
                logger.warning(f"mlflow init failed: {e}. Falling back to console logging.")
                self.backend = "none"

        elif self.backend != "none":
            logger.warning(f"Unknown tracker backend '{backend}'. Using console logging.")
            self.backend = "none"

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.finish()

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log scalar metrics.

        Args:
            metrics: Dictionary of metric name → value.
            step: Training step or epoch number.
        """
        if self.backend == "wandb":
            import wandb
            wandb.log(metrics, step=step)
        elif self.backend == "mlflow":
            import mlflow
            mlflow.log_metrics(metrics, step=step)
        else:
            logger.info(f"[Step {step}] {metrics}")

    def log_config(self, config: Union[Dict, Any]) -> None:
        """Log configuration parameters.

        Args:
            config: Configuration dict or object with __dict__.
        """
        if isinstance(config, dict):
            config_dict = config
        elif hasattr(config, "__dict__"):
            config_dict = config.__dict__
        elif hasattr(config, "to_dict"):
            config_dict = config.to_dict()
        else:
            config_dict = {"config": str(config)}

        if self.backend == "wandb":
            import wandb
            wandb.config.update(config_dict, allow_val_change=True)
        elif self.backend == "mlflow":
            import mlflow
            flat = {k: str(v)[:250] for k, v in config_dict.items()}
            mlflow.log_params(flat)
        else:
            logger.info(f"[Config] {config_dict}")

    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        """Log a file artifact (checkpoint, image, etc.).

        Args:
            path: Path to the artifact file.
            name: Optional display name for the artifact.
        """
        if self.backend == "wandb":
            import wandb
            wandb.save(path)
        elif self.backend == "mlflow":
            import mlflow
            mlflow.log_artifact(path)
        else:
            logger.info(f"[Artifact] {name or path}: {path}")

    def log_image(
        self,
        image: np.ndarray,
        name: str,
        step: int,
        caption: Optional[str] = None,
    ) -> None:
        """Log an image array.

        Args:
            image: Image as numpy array (H, W, C) or (H, W).
            name: Name/key for the image.
            step: Training step.
            caption: Optional caption for the image.
        """
        if self.backend == "wandb":
            import wandb
            wandb.log({name: wandb.Image(image, caption=caption)}, step=step)
        elif self.backend == "mlflow":
            import mlflow
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
                ax.set_title(name)
                ax.axis("off")
                mlflow.log_figure(fig, f"{name}_step{step}.png")
                plt.close(fig)
            except Exception:
                logger.info(f"[Image] {name} at step {step} (shape: {image.shape})")
        else:
            logger.info(f"[Image] {name} at step {step} (shape: {image.shape})")

    def log_table(self, data: Dict[str, list], name: str) -> None:
        """Log tabular data.

        Args:
            data: Dictionary of column_name → list of values.
            name: Name/key for the table.
        """
        if self.backend == "wandb":
            import wandb
            columns = list(data.keys())
            rows = list(zip(*data.values()))
            table = wandb.Table(columns=columns, data=rows)
            wandb.log({name: table})
        elif self.backend == "mlflow":
            logger.info(f"[Table] {name}: {data}")
        else:
            logger.info(f"[Table] {name}: {data}")

    def finish(self) -> None:
        """Finalize the tracking run."""
        if self.backend == "wandb":
            import wandb
            wandb.finish()
        elif self.backend == "mlflow":
            import mlflow
            mlflow.end_run()
