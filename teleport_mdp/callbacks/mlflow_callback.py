import math
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter

from utils.experiment_logger import MlflowLogger


def _finite_scalars(key_values: dict[str, Any]) -> dict[str, float]:
    """Keep only the finite numeric diagnostics, as floats.

    Args:
        key_values: The SB3 logger's current key/value diagnostics.

    Returns:
        A mapping of metric name to float, dropping strings, bools, and non-finite
        values that MLflow cannot store as metrics.
    """
    metrics: dict[str, float] = {}
    for key, value in key_values.items():
        # bool is an int subclass; exclude it and any non-scalar diagnostics.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        scalar = float(value)
        if math.isnan(scalar) or math.isinf(scalar):
            continue
        metrics[key] = scalar
    return metrics


class _MlflowOutputFormat(KVWriter):
    """SB3 output format that forwards each `logger.dump` through an `MlflowLogger`.

    This is the *hook*: Stable-Baselines3 calls :meth:`write` once per
    `logger.dump` (i.e. once per rollout, not per env step) with the complete set of
    diagnostics it is about to emit.

    Args:
        mlflow_logger: The project MLflow logger that owns the MLflow connection.
    """

    def __init__(self, mlflow_logger: MlflowLogger) -> None:
        self._mlflow_logger = mlflow_logger

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...]],
        step: int = 0,
    ) -> None:
        """Forward the finite numeric diagnostics of one iteration to the logger.

        Args:
            key_values: The SB3 logger's current key/value diagnostics.
            key_excluded: Per-key output-format exclusions (unused; nothing excludes
                the MLflow path).
            step: The training step (SB3 passes `num_timesteps`).
        """
        metrics = _finite_scalars(key_values)
        if metrics:
            # `log_metrics` is run-guarded by MlflowLogger: a no-op without a run.
            self._mlflow_logger.log_metrics(metrics, step=step)

    def close(self) -> None:
        """Release resources (none held)."""
        return


class MlflowCallback(BaseCallback):
    """Stream SB3 training metrics to MLflow live, through the `MlflowLogger`.

    Args:
        mlflow_logger: The active project MLflow logger to route metrics through.
        verbose: SB3 verbosity level.
    """

    def __init__(self, mlflow_logger: MlflowLogger, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._mlflow_logger = mlflow_logger
        self._output_format: _MlflowOutputFormat | None = None

    def _on_training_start(self) -> None:
        """Attach the MLflow output format to the model's logger (idempotent)."""
        if self._output_format is None:
            self._output_format = _MlflowOutputFormat(self._mlflow_logger)
            self.logger.output_formats.append(self._output_format)

    def _on_step(self) -> bool:
        """Continue training; metric streaming happens on `logger.dump`.

        Returns:
            `True` so training always continues.
        """
        return True

    def _on_training_end(self) -> None:
        """Detach and close the MLflow output format."""
        if self._output_format is not None:
            if self._output_format in self.logger.output_formats:
                self.logger.output_formats.remove(self._output_format)
            self._output_format.close()
            self._output_format = None
