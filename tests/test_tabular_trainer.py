import pytest

from teleport_mdp.models import (
    CurriculumConfig,
    EnvConfig,
    ExperimentConfig,
    PPOConfig,
    QLearningConfig,
    TeleportConfig,
    TMPIConfig,
)
from teleport_mdp.tabular.trainer import TabularTrainer


class _StubLogger:
    """Context-managed logger that records params and metrics without MLflow."""

    def __init__(self, _config) -> None:
        self.params: dict = {}
        self.metrics: list[tuple[dict, int]] = []

    def __enter__(self) -> "_StubLogger":
        return self

    def __exit__(self, *_exc) -> bool:
        return False

    def log_params(self, params: dict) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict, step: int) -> None:
        self.metrics.append((metrics, step))


class _StubMlflowConfig:
    """Minimal stand-in for MlflowLoggerConfig supporting `model_copy`."""

    def model_copy(self, update: dict) -> "_StubMlflowConfig":
        return self


def _make_trainer(cfg: ExperimentConfig) -> tuple[TabularTrainer, list[_StubLogger]]:
    """Build a trainer whose logger factory records every created stub logger."""
    created: list[_StubLogger] = []

    def factory(config) -> _StubLogger:
        logger = _StubLogger(config)
        created.append(logger)
        return logger

    trainer = TabularTrainer(cfg, mlflow_config=_StubMlflowConfig(), logger_factory=factory)
    return trainer, created


def _q_learning_cfg(**overrides) -> ExperimentConfig:
    """A small 4x4 tabular Q-learning experiment config."""
    base = dict(
        name="test_q",
        gamma=0.99,
        seed=0,
        algorithm=QLearningConfig(alpha=0.6, eps=0.4, episodes=6000, status_step=1000),
        env=EnvConfig(map_name="4x4", is_slippery=False),
        teleport=TeleportConfig(tau_0=0.0),
        curriculum=CurriculumConfig(kind="none"),
    )
    base.update(overrides)
    return ExperimentConfig(**base)


def test_q_learning_run_solves_frozenlake():
    """A no-curriculum Q-learning run learns the optimal 4x4 policy (J approx gamma^5)."""
    trainer, loggers = _make_trainer(_q_learning_cfg())

    results = trainer.run()

    assert [r.seed for r in results] == [0]
    # Optimal deterministic path to the 4x4 goal is 5 steps: J = gamma^5.
    assert results[0].performance == pytest.approx(0.99**5, abs=1e-6)
    # The config was logged as flattened params, and eval/performance was recorded.
    assert loggers[0].params["algorithm.kind"] == "q_learning"
    assert any("eval/performance" in m for m, _ in loggers[0].metrics)


def test_static_curriculum_q_learning_runs_and_solves():
    """The static curriculum path builds a scheduler and still solves the task."""
    cfg = _q_learning_cfg(
        name="test_st_q",
        algorithm=QLearningConfig(alpha=0.6, eps=0.4, episodes=12000, status_step=500),
        teleport=TeleportConfig(tau_0=0.35),
        curriculum=CurriculumConfig(kind="static"),
    )
    trainer, _ = _make_trainer(cfg)

    results = trainer.run()

    assert results[0].performance == pytest.approx(0.99**5, abs=1e-6)


def test_tmpi_run_reports_annealed_performance():
    """A TMPI run drives tau to 0 and reports a positive real-MDP performance."""
    cfg = ExperimentConfig(
        name="test_tmpi",
        gamma=0.9,
        seed=0,
        algorithm=TMPIConfig(threshold=1e-9, max_iterations=2000, temperature=0.1),
        env=EnvConfig(desc=["SFFG"], is_slippery=False),
        teleport=TeleportConfig(tau_0=0.4),
        curriculum=CurriculumConfig(kind="none"),
    )
    trainer, loggers = _make_trainer(cfg)

    results = trainer.run()

    assert 0.0 < results[0].performance <= 1.0
    assert any("tmpi/bound" in m for m, _ in loggers[0].metrics)


def test_multiple_seeds_produce_one_result_each():
    """`n_runs` seeded repeats yield consecutive seeds and one logger per run."""
    trainer, loggers = _make_trainer(_q_learning_cfg(seed=5, n_runs=3))

    results = trainer.run()

    assert [r.seed for r in results] == [5, 6, 7]
    assert len(loggers) == 3


def test_rejects_non_tabular_algorithm():
    """Pointing the tabular trainer at a PPO config is rejected at train time."""
    cfg = ExperimentConfig(
        name="test_ppo",
        algorithm=PPOConfig(),
        env=EnvConfig(map_name="4x4", is_slippery=False),
        teleport=TeleportConfig(tau_0=0.0),
        curriculum=CurriculumConfig(kind="none"),
    )
    trainer, _ = _make_trainer(cfg)

    with pytest.raises(ValueError, match="does not handle"):
        trainer.run()
