from collections.abc import Mapping
from logging import getLogger
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from pydantic import ValidationError

from teleport_mdp.callbacks import OptunaPruningCallback
from teleport_mdp.models import ExperimentConfig, OptunaConfig, PPOConfig
from teleport_mdp.trainer import LoggerFactory, Trainer
from utils.configs import MlflowLoggerConfig
from utils.experiment_logger import MlflowLogger

logger = getLogger(__name__)

#: Report cadence (in timesteps) for algorithms without a natural rollout length.
DEFAULT_REPORT_FREQ = 2048
#: File the best trial's config is written to when no destination is given.
DEFAULT_BEST_CONFIG_PATH = Path("best_config.yaml")


class HyperparameterOptimizer:
    """Optuna study over an experiment's `optuna.param_space`.

    Each trial samples the configured hyperparameters, bakes them into a copy of
    the base :class:`ExperimentConfig`, and runs the standard :class:`Trainer` —
    the same train + real-MDP (`tau=0`) evaluation used by `run_experiment`. The
    objective is the trial's mean evaluation return, averaged over the config's
    `n_runs` seeds so a single lucky seed cannot win the study. Trials stream
    intermediate returns to a :class:`~optuna.pruners.MedianPruner`, and every
    trial's fully-resolved config is logged by the Trainer (flattened MLflow params
    plus a YAML artifact), so any trial is reproducible from its record. Trials are
    grouped by run-name convention (`<study>_trial_<n>_seed_<s>`) rather than
    MLflow parent/child nesting, which the single-run `MlflowLogger` does not model.

    Args:
        cfg: The base experiment config; its `optuna` block defines the study.
        mlflow_config: MLflow logger config; loaded from default YAML when omitted.
        logger_factory: Builds the context-managed MlflowLogger per run.
        progress: Whether each trial's training shows a tqdm progress bar.

    Raises:
        ValueError: If `cfg.optuna` is missing or disabled.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        mlflow_config: MlflowLoggerConfig | None = None,
        logger_factory: LoggerFactory = MlflowLogger,
        *,
        progress: bool = False,
    ) -> None:
        if cfg.optuna is None or not cfg.optuna.enabled:
            raise ValueError(
                "HyperparameterOptimizer requires an enabled `optuna` block in the config."
            )
        self._cfg = cfg
        self._optuna: OptunaConfig = cfg.optuna
        self._mlflow_config = mlflow_config or MlflowLoggerConfig.from_yaml()
        self._logger_factory = logger_factory
        self._progress = progress
        self._study_name = self._optuna.study_name or cfg.name

    def optimize(self, best_config_path: Path | None = None) -> optuna.Study:
        """Run the study and persist the best trial's config to YAML.

        Args:
            best_config_path: Where to write the best config; defaults to
                `best_config.yaml` in the working directory.

        Returns:
            The completed Optuna study.
        """
        study = optuna.create_study(
            study_name=self._study_name,
            direction=self._optuna.direction.value,
            sampler=TPESampler(seed=self._cfg.seed),
            pruner=MedianPruner(),
            load_if_exists=self._optuna.load_if_exists,
        )
        study.optimize(
            self._objective,
            n_trials=self._optuna.n_trials,
            timeout=self._optuna.timeout,
        )

        destination = best_config_path or DEFAULT_BEST_CONFIG_PATH
        self.build_config(study.best_params).to_yaml(destination)
        logger.info(
            "Best trial %d: objective=%.4f -> wrote %s",
            study.best_trial.number,
            study.best_value,
            destination,
        )
        return study

    def build_config(
        self, overrides: Mapping[str, Any], *, name: str | None = None
    ) -> ExperimentConfig:
        """Bake dotted-key hyperparameter overrides into a runnable config.

        Overrides are dotted paths into the base config (e.g.
        `algorithm.learning_rate`, `teleport.tau_0`). The result drops the `optuna`
        block so it runs as a plain experiment, and is re-validated so an invalid
        sampled combination surfaces as a `ValidationError`.

        Args:
            overrides: Dotted-key hyperparameter values to apply.
            name: Experiment name for the copy; keeps the base name when omitted.

        Returns:
            The validated, optuna-free experiment config.
        """
        data = self._cfg.model_dump(mode="python")
        for dotted_key, value in overrides.items():
            _assign_nested(data, dotted_key, value)
        data["optuna"] = None
        if name is not None:
            data["name"] = name
        return ExperimentConfig.model_validate(data)

    def _objective(self, trial: optuna.Trial) -> float:
        """Sample, train, and return a trial's mean evaluation return.

        Args:
            trial: The Optuna trial supplying the sampled hyperparameters.

        Returns:
            The mean evaluation return across the config's seeds.

        Raises:
            optuna.TrialPruned: If the sampled params are invalid or the pruner
                stops the trial before it finishes.
        """
        overrides = self._sample(trial)
        try:
            trial_cfg = self.build_config(
                overrides, name=f"{self._study_name}_trial_{trial.number:03d}"
            )
        except ValidationError as error:
            logger.warning("Pruning trial %d with invalid params %s.", trial.number, overrides)
            raise optuna.TrialPruned() from error

        pruning_callback = OptunaPruningCallback(trial, report_freq=_report_freq(trial_cfg))
        results = Trainer(
            trial_cfg,
            mlflow_config=self._mlflow_config,
            logger_factory=self._logger_factory,
            progress=self._progress,
            extra_callbacks=[pruning_callback],
        ).run()

        if pruning_callback.pruned:
            raise optuna.TrialPruned()
        return float(np.mean([result.mean_return for result in results]))

    def _sample(self, trial: optuna.Trial) -> dict[str, Any]:
        """Draw one value per parameter in the search space."""
        overrides: dict[str, Any] = {}
        for parameter in self._optuna.param_space:
            overrides.update(parameter.get_optuna_suggestion(trial))
        return overrides


def _report_freq(cfg: ExperimentConfig) -> int:
    """Rollout-length report cadence for PPO, else a fixed default."""
    if isinstance(cfg.algorithm, PPOConfig):
        return cfg.algorithm.n_steps
    return DEFAULT_REPORT_FREQ


def _assign_nested(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set `data[a][b]... = value` for a dotted key, walking existing dicts.

    Args:
        data: The mutable nested mapping (a dumped config) to update in place.
        dotted_key: A dotted path such as `algorithm.learning_rate`.
        value: The value to assign at the leaf.
    """
    *branches, leaf = dotted_key.split(".")
    cursor = data
    for branch in branches:
        cursor = cursor[branch]
    cursor[leaf] = value
