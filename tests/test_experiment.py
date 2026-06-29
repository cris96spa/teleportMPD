from typing import Any, cast

import pytest

from teleport_mdp.enums import TeleportDistribution
from teleport_mdp.models import EnvConfig, ExperimentConfig
from teleport_mdp.trainer import RunResult, Trainer
from utils.configs import MlflowLoggerConfig
from utils.experiment_logger import MlflowLogger


class _StubLogger:
    """No-op context manager capturing all MlflowLogger calls."""

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
    """Return (created_list, factory) for injecting into Trainer."""
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


def _smoke_config(**overrides: Any) -> ExperimentConfig:
    base: dict[str, Any] = {
        "name": "exp-smoke",
        "algorithm": {"kind": "ppo", "n_steps": 16, "batch_size": 16, "n_epochs": 1},
        "env": EnvConfig(map_name="4x4", is_slippery=False),
        "gamma": 0.99,
        "total_timesteps": 48,
        "seed": 0,
    }
    base.update(overrides)
    return ExperimentConfig(**base)


def test_streams_metrics_and_evaluates():
    """Training streams live metrics and logs a final eval return."""
    created, factory = _stub_factory()
    results = Trainer(
        _smoke_config(), mlflow_config=_mlflow_config(), logger_factory=factory, progress=False
    ).run()

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, RunResult)
    assert result.seed == 0
    assert 0.0 <= result.mean_return <= 1.0

    (logger,) = created
    logged_keys = {key for metrics, _ in logger.metrics for key in metrics}
    assert "rollout/ep_rew_mean" in logged_keys
    assert "eval/mean_return" in logged_keys
    assert logger.params["name"] == "exp-smoke"
    assert logger.params["algorithm.kind"] == "ppo"
    assert logger.artifacts


def test_distinct_seeds_and_run_names():
    """n_runs produces one logged run per consecutive seed with distinct names."""
    created, factory = _stub_factory()
    cfg = _smoke_config(total_timesteps=16, n_runs=3, seed=5)
    results = Trainer(
        cfg, mlflow_config=_mlflow_config(), logger_factory=factory, progress=False
    ).run()

    assert [r.seed for r in results] == [5, 6, 7]
    run_names = [logger.config.run_name for logger in created]
    assert run_names == ["exp-smoke_seed_5", "exp-smoke_seed_6", "exp-smoke_seed_7"]
    assert all(logger.config.experiment_name == "exp-smoke" for logger in created)


def test_reproducible():
    """Same seed yields identical evaluation returns."""
    _, factory_a = _stub_factory()
    _, factory_b = _stub_factory()
    first = Trainer(
        _smoke_config(), mlflow_config=_mlflow_config(), logger_factory=factory_a, progress=False
    ).run()
    second = Trainer(
        _smoke_config(), mlflow_config=_mlflow_config(), logger_factory=factory_b, progress=False
    ).run()

    assert first[0].mean_return == pytest.approx(second[0].mean_return)
    assert first[0].std_return == pytest.approx(second[0].std_return)


def test_rejects_unported_curriculum():
    """Teleport curriculum raises NotImplementedError until TeleportPPO lands."""
    _, factory = _stub_factory()
    cfg = _smoke_config(
        teleport={"tau_0": 0.5, "distribution": TeleportDistribution.UNIFORM_NONTERMINAL},
        curriculum={"kind": "static"},
    )
    with pytest.raises(NotImplementedError):
        Trainer(cfg, mlflow_config=_mlflow_config(), logger_factory=factory, progress=False).run()
