from typing import Any, cast

import pytest

from teleport_mdp.models import EnvConfig, ExperimentConfig
from teleport_mdp.trainer import Trainer
from utils.configs import MlflowLoggerConfig
from utils.experiment_logger import MlflowLogger


class _StubLogger:
    """No-op context manager capturing streamed metrics."""

    def __init__(self, config: MlflowLoggerConfig) -> None:
        self.config = config
        self.params: dict[str, Any] = {}
        self.metrics: list[tuple[dict[str, float], int]] = []
        self.artifacts: list[str] = []

    def __enter__(self) -> "_StubLogger":
        """Enter context."""
        return self

    def __exit__(self, *exc_info: object) -> bool:
        """Exit context."""
        return False

    def log_params(self, params: dict[str, Any]) -> None:
        """Record params."""
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        """Record metrics."""
        self.metrics.append((dict(metrics), step))

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Record artifact path."""
        self.artifacts.append(local_path)


def _stub_factory() -> tuple[list[_StubLogger], Any]:
    created: list[_StubLogger] = []

    def factory(config: MlflowLoggerConfig) -> MlflowLogger:
        stub = _StubLogger(config)
        created.append(stub)
        return cast(MlflowLogger, stub)

    return created, factory


def _mlflow_config() -> MlflowLoggerConfig:
    return MlflowLoggerConfig(
        project_name="test",
        experiment_name="test",
        tracking_uri="http://localhost:5002",  # type: ignore[arg-type]
    )


def _curriculum_config(**overrides: Any) -> ExperimentConfig:
    base: dict[str, Any] = {
        "name": "exp-curriculum",
        "algorithm": {"kind": "ppo", "n_steps": 16, "batch_size": 16, "n_epochs": 1},
        "env": EnvConfig(map_name="4x4", is_slippery=False),
        "gamma": 0.99,
        "total_timesteps": 64,
        "seed": 0,
        "teleport": {"tau_0": 0.5, "distribution": "uniform_nonterminal"},
    }
    base.update(overrides)
    return ExperimentConfig(**base)


def _tau_values(logger: _StubLogger) -> list[float]:
    return [m["teleport/tau"] for m, _ in logger.metrics if "teleport/tau" in m]


def test_static_curriculum_drives_tau_to_zero():
    """A static run logs teleport diagnostics and anneals tau to 0 within budget."""
    created, factory = _stub_factory()
    cfg = _curriculum_config(curriculum={"kind": "static"})  # n_updates = 64 // 16 = 4
    results = Trainer(
        cfg, mlflow_config=_mlflow_config(), logger_factory=factory, progress=False
    ).run()

    assert len(results) == 1
    assert 0.0 <= results[0].mean_return <= 1.0

    (logger,) = created
    taus = _tau_values(logger)
    # The logged teleport/tau series is the post-update rate, so it reaches 0.
    assert taus, "teleport/tau was never logged"
    assert taus[-1] == pytest.approx(0.0)
    logged_keys = {key for metrics, _ in logger.metrics for key in metrics}
    assert "teleport/d_inf" in logged_keys


def test_dynamic_curriculum_reduces_tau():
    """A dynamic run with a generous budget lowers tau below its initial value."""
    created, factory = _stub_factory()
    cfg = _curriculum_config(curriculum={"kind": "dynamic", "eps": 1.0, "eps_tau_max": 0.1})
    Trainer(cfg, mlflow_config=_mlflow_config(), logger_factory=factory, progress=False).run()

    (logger,) = created
    taus = _tau_values(logger)
    assert taus, "teleport/tau was never logged"
    assert min(taus) < 0.5
    assert all(value >= 0.0 for value in taus)
