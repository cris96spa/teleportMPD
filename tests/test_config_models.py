from pathlib import Path

import pytest
from pydantic import ValidationError

from teleport_mdp.enums import Algorithm, Curriculum, TeleportDistribution
from teleport_mdp.models import (
    CurriculumConfig,
    ExperimentConfig,
    PPOConfig,
    TeleportConfig,
)

EXPERIMENTS_DIR = Path("configs/commands")
EXAMPLE_CONFIGS = [
    "frozen_lake_ppo.yaml",
    "frozen_lake_st_ppo.yaml",
    "frozen_lake_dt_ppo.yaml",
]


# region example configs load
@pytest.mark.parametrize("filename", EXAMPLE_CONFIGS)
def test_example_configs_validate(filename: str):
    """Every shipped example experiment config validates."""
    cfg = ExperimentConfig.from_yaml(EXPERIMENTS_DIR / filename)
    assert isinstance(cfg.algorithm, PPOConfig)
    assert cfg.algorithm.kind == Algorithm.PPO


def test_roundtrip_is_stable(tmp_path: Path):
    """`from_yaml` -> `to_yaml` -> `from_yaml` reproduces the same config."""
    cfg = ExperimentConfig.from_yaml(EXPERIMENTS_DIR / "frozen_lake_dt_ppo.yaml")
    out = tmp_path / "roundtrip.yaml"
    cfg.to_yaml(out)
    reloaded = ExperimentConfig.from_yaml(out)
    assert reloaded == cfg


# endregion


# region validators
def test_tau_out_of_range_rejected():
    """tau_0 must live in [0, 1)."""
    with pytest.raises(ValidationError):
        TeleportConfig(tau_0=1.2)
    with pytest.raises(ValidationError):
        TeleportConfig(tau_0=1.0)


def test_dynamic_requires_budgets():
    """A dynamic curriculum without its budgets is rejected."""
    with pytest.raises(ValidationError):
        CurriculumConfig(kind=Curriculum.DYNAMIC)


def test_dynamic_fields_forbidden_for_non_dynamic():
    """Dynamic-only budgets cannot be set on a non-dynamic curriculum."""
    with pytest.raises(ValidationError):
        CurriculumConfig(kind=Curriculum.STATIC, eps=0.1, eps_tau_max=0.05)


def test_custom_distribution_requires_path():
    """A custom teleport distribution must provide its xi source."""
    with pytest.raises(ValidationError):
        TeleportConfig(distribution=TeleportDistribution.CUSTOM)


def test_algorithm_discriminator_selects_variant():
    """The `kind` tag selects the matching algorithm config variant."""
    cfg = ExperimentConfig(name="x", algorithm={"kind": "q_learning", "alpha": 0.5})
    assert cfg.algorithm.kind == Algorithm.Q_LEARNING
    assert cfg.algorithm.alpha == pytest.approx(0.5)


def test_unknown_algorithm_kind_rejected():
    """An algorithm block with an unknown/absent `kind` tag is rejected."""
    with pytest.raises(ValidationError):
        ExperimentConfig(name="x", algorithm={"kind": "not_an_algorithm"})
    with pytest.raises(ValidationError):
        ExperimentConfig(name="x", algorithm={"n_steps": 64})


def test_ppo_block_rejects_q_learning_fields():
    """A PPO algorithm block cannot carry foreign (Q-learning) hyperparameters."""
    with pytest.raises(ValidationError):
        ExperimentConfig(name="x", algorithm={"kind": "ppo", "alpha": 0.5})


def test_curriculum_requires_positive_tau():
    """A teleport curriculum with tau_0 == 0 is rejected."""
    with pytest.raises(ValidationError):
        ExperimentConfig(
            name="x",
            algorithm={"kind": "q_learning"},
            curriculum={"kind": "static"},
            teleport={"tau_0": 0.0},
        )


def test_tmpi_rejects_external_curriculum():
    """TMPI schedules its own rate, so an external curriculum is rejected."""
    with pytest.raises(ValidationError):
        ExperimentConfig(
            name="x",
            algorithm={"kind": "tmpi"},
            curriculum={"kind": "static"},
            teleport={"tau_0": 0.5},
        )


def test_extra_fields_forbidden():
    """Unknown keys are rejected (typo protection)."""
    with pytest.raises(ValidationError):
        TeleportConfig(taoo=0.5)  # type: ignore[call-arg]


# endregion
