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


def test_eval_freq_streams_intermediate_evaluations():
    """With eval_freq set, real-MDP evals are streamed mid-training, not just at the end."""
    created, factory = _stub_factory()
    cfg = ExperimentConfig(
        name="exp-eval",
        algorithm={"kind": "ppo", "n_steps": 16, "batch_size": 16, "n_epochs": 1},
        env=EnvConfig(map_name="4x4", is_slippery=False),
        gamma=0.99,
        total_timesteps=64,  # four 16-step rollouts
        eval_freq=16,  # evaluate every rollout -> evals before the final step
        seed=0,
    )
    Trainer(cfg, mlflow_config=_mlflow_config(), logger_factory=factory, progress=False).run()

    (logger,) = created
    eval_steps = [step for metrics, step in logger.metrics if "eval/mean_reward" in metrics]
    assert eval_steps, "eval/mean_reward was never streamed during training"
    assert any(step < cfg.total_timesteps for step in eval_steps), "no mid-training evaluation"


def test_no_eval_freq_runs_only_final_evaluation():
    """Without eval_freq (the default), no intermediate eval/mean_reward is streamed."""
    created, factory = _stub_factory()
    cfg = _curriculum_config(curriculum={"kind": "static"})  # eval_freq defaults to None
    Trainer(cfg, mlflow_config=_mlflow_config(), logger_factory=factory, progress=False).run()

    (logger,) = created
    assert not any("eval/mean_reward" in metrics for metrics, _ in logger.metrics)


def test_dynamic_curriculum_frozen_budget_warns(caplog):
    """A budget too small for the gamma/(1-gamma) scale freezes tau and warns once."""
    _, factory = _stub_factory()
    # eps far below gamma/(1-gamma)*D_inf, so eps_tau <= 0 every update and tau never moves.
    cfg = _curriculum_config(
        curriculum={"kind": "dynamic", "eps": 1e-9, "eps_tau_max": 1e-9},
        total_timesteps=192,  # 12 updates at n_steps=16, past the default patience of 10
    )
    with caplog.at_level("WARNING", logger="teleport_mdp.agents.teleport_ppo"):
        Trainer(cfg, mlflow_config=_mlflow_config(), logger_factory=factory, progress=False).run()

    frozen_warnings = [r for r in caplog.records if "has not annealed" in r.getMessage()]
    assert len(frozen_warnings) == 1, "expected exactly one frozen-curriculum warning"
