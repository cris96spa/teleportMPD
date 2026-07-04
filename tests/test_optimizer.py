from pathlib import Path
from typing import Any, cast

import pytest

from teleport_mdp.callbacks import OptunaPruningCallback
from teleport_mdp.models import EnvConfig, ExperimentConfig
from teleport_mdp.optimizer import HyperparameterOptimizer
from utils.configs import MlflowLoggerConfig
from utils.experiment_logger import MlflowLogger


class _StubLogger:
    """No-op context manager standing in for the MlflowLogger during a study."""

    def __init__(self, config: MlflowLoggerConfig) -> None:
        self.config = config

    def __enter__(self) -> "_StubLogger":
        """Enter context."""
        return self

    def __exit__(self, *exc_info: object) -> bool:
        """Exit context."""
        return False

    def log_params(self, params: dict[str, Any]) -> None:
        """Ignore params."""

    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        """Ignore metrics."""

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Ignore artifacts."""


def _stub_factory() -> Any:
    """A logger factory that hands the Trainer a no-op logger (no MLflow server)."""

    def factory(config: MlflowLoggerConfig) -> MlflowLogger:
        return cast(MlflowLogger, _StubLogger(config))

    return factory


def _mlflow_config() -> MlflowLoggerConfig:
    return MlflowLoggerConfig(
        project_name="test",
        experiment_name="test",
        tracking_uri="http://localhost:5002",  # type: ignore[arg-type]
    )


def _study_config(**overrides: Any) -> ExperimentConfig:
    base: dict[str, Any] = {
        "name": "hpo-smoke",
        "algorithm": {"kind": "ppo", "n_steps": 16, "batch_size": 16, "n_epochs": 1},
        "env": EnvConfig(map_name="4x4", is_slippery=False),
        "gamma": 0.99,
        "total_timesteps": 32,
        "seed": 0,
        "optuna": {
            "enabled": True,
            "n_trials": 2,
            "direction": "maximize",
            "study_name": "hpo-smoke",
            "param_space": [
                {
                    "name": "algorithm.learning_rate",
                    "kind": "float",
                    "low": 1.0e-4,
                    "high": 1.0e-2,
                    "log": True,
                }
            ],
        },
    }
    base.update(overrides)
    return ExperimentConfig(**base)


def _optimizer(cfg: ExperimentConfig) -> HyperparameterOptimizer:
    return HyperparameterOptimizer(
        cfg, mlflow_config=_mlflow_config(), logger_factory=_stub_factory()
    )


def test_optimize_writes_a_runnable_best_config(tmp_path: Path):
    """A finished study writes a valid, optuna-free config with the tuned field in range."""
    best_path = tmp_path / "best_config.yaml"
    study = _optimizer(_study_config()).optimize(best_config_path=best_path)

    assert len(study.trials) == 2
    best = ExperimentConfig.from_yaml(best_path)
    assert best.optuna is None
    assert 1.0e-4 <= best.algorithm.learning_rate <= 1.0e-2  # type: ignore[union-attr]


def test_study_is_seed_reproducible(tmp_path: Path):
    """A seeded sampler over deterministic training reproduces the best trial."""
    first = _optimizer(_study_config()).optimize(best_config_path=tmp_path / "a.yaml")
    second = _optimizer(_study_config()).optimize(best_config_path=tmp_path / "b.yaml")

    assert first.best_params == second.best_params
    assert first.best_value == pytest.approx(second.best_value)


def test_build_config_applies_dotted_overrides():
    """Dotted keys land on the right nested field and the optuna block is dropped."""
    cfg = _optimizer(_study_config()).build_config(
        {"algorithm.learning_rate": 1.0e-3, "teleport.tau_0": 0.0}, name="trial-x"
    )
    assert cfg.algorithm.learning_rate == pytest.approx(1.0e-3)  # type: ignore[union-attr]
    assert cfg.name == "trial-x"
    assert cfg.optuna is None


def test_requires_an_enabled_optuna_block():
    """Without an enabled optuna block the optimizer refuses to build."""
    with pytest.raises(ValueError, match="optuna"):
        _optimizer(_study_config(optuna=None))
    with pytest.raises(ValueError, match="optuna"):
        _optimizer(_study_config(optuna={"enabled": False}))


class _FakeTrial:
    """Minimal Optuna trial capturing reports and forcing a prune verdict."""

    def __init__(self, *, prune: bool) -> None:
        self._prune = prune
        self.reports: list[tuple[float, int]] = []

    def report(self, value: float, step: int) -> None:
        """Record an intermediate value."""
        self.reports.append((value, step))

    def should_prune(self) -> bool:
        """Return the pre-set prune verdict."""
        return self._prune


class _FakeModel:
    """Stand-in exposing only the recent-episode buffer the callback reads."""

    def __init__(self, ep_info_buffer: list[dict[str, float]]) -> None:
        self.ep_info_buffer = ep_info_buffer


def _drive_callback(callback: OptunaPruningCallback, model: _FakeModel, timestep: int) -> bool:
    """Invoke one `_on_step` with the model and timestep the callback reads."""
    callback.model = cast(Any, model)
    callback.num_timesteps = timestep
    return callback._on_step()


def test_pruning_callback_reports_running_return():
    """The callback reports the mean buffered return at ordinal report steps."""
    trial = _FakeTrial(prune=False)
    callback = OptunaPruningCallback(cast(Any, trial), report_freq=100)

    keep_going = _drive_callback(callback, _FakeModel([{"r": 1.0}, {"r": 0.0}]), timestep=100)

    assert keep_going is True
    assert trial.reports == [(0.5, 0)]
    assert callback.pruned is False


def test_pruning_callback_flags_prune_without_raising():
    """A prune verdict sets `pruned` and stops the run instead of raising."""
    trial = _FakeTrial(prune=True)
    callback = OptunaPruningCallback(cast(Any, trial), report_freq=100)

    keep_going = _drive_callback(callback, _FakeModel([{"r": 0.0}]), timestep=100)

    assert keep_going is False
    assert callback.pruned is True


def test_pruning_callback_waits_for_report_freq():
    """No report is made before `report_freq` timesteps elapse."""
    trial = _FakeTrial(prune=False)
    callback = OptunaPruningCallback(cast(Any, trial), report_freq=100)

    keep_going = _drive_callback(callback, _FakeModel([{"r": 1.0}]), timestep=50)

    assert keep_going is True
    assert trial.reports == []
