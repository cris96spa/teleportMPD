import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, NamedTuple

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed

from teleport_mdp.callbacks import MlflowCallback, ProgressCallback
from teleport_mdp.environments.factory import make_vec_env
from teleport_mdp.models import ExperimentConfig
from teleport_mdp.registries import build_agent
from utils.configs import MlflowLoggerConfig
from utils.experiment_logger import MlflowLogger

DEFAULT_EVAL_EPISODES = 20
EVAL_SEED_OFFSET = 10_000
LoggerFactory = Callable[[MlflowLoggerConfig], MlflowLogger]


class RunResult(NamedTuple):
    """Summary of a single seeded training run."""

    seed: int
    mean_return: float
    std_return: float
    n_eval_episodes: int


class Trainer:
    """Orchestrates training, evaluation, and MLflow logging for an experiment.

    Args:
        cfg: The experiment configuration.
        mlflow_config: MLflow logger config; loaded from default YAML when omitted.
        logger_factory: Builds the context-managed MlflowLogger per run.
        progress: Whether to show a tqdm progress bar during training.
        extra_callbacks: SB3 callbacks appended to every seeded run's callback list
            (e.g. an Optuna pruning hook). The same instances are reused across
            seeds, so any per-run state they hold must reset in `_on_training_start`.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        mlflow_config: MlflowLoggerConfig | None = None,
        logger_factory: LoggerFactory = MlflowLogger,
        *,
        progress: bool = True,
        extra_callbacks: Sequence[BaseCallback] = (),
    ) -> None:
        self._cfg = cfg
        self._mlflow_config = mlflow_config or MlflowLoggerConfig.from_yaml()
        self._logger_factory = logger_factory
        self._progress = progress
        self._extra_callbacks = list(extra_callbacks)

    def run(self) -> list[RunResult]:
        """Execute all seeded runs and return their evaluation results."""
        seeds = self._resolve_seeds()
        n_runs = len(seeds)
        results: list[RunResult] = []
        for idx, seed in enumerate(seeds, 1):
            run_config = self._mlflow_config.model_copy(
                update={
                    "experiment_name": self._cfg.name,
                    "run_name": f"{self._cfg.name}_seed_{seed}",
                }
            )
            with self._logger_factory(run_config) as mlflow_logger:
                results.append(self._run_single(seed, mlflow_logger, run_idx=idx, n_runs=n_runs))
        return results

    def _run_single(
        self, seed: int, mlflow_logger: MlflowLogger, *, run_idx: int, n_runs: int
    ) -> RunResult:
        """Train and evaluate one seeded run."""
        set_random_seed(seed)
        train_env = make_vec_env(self._cfg, n_envs=self._cfg.n_envs, seed=seed)
        try:
            self._log_config(mlflow_logger)
            agent = build_agent(self._cfg, train_env, seed)
            callbacks = self._build_callbacks(
                mlflow_logger, seed=seed, run_idx=run_idx, n_runs=n_runs
            )
            agent.learn(total_timesteps=self._cfg.total_timesteps, callback=callbacks)
            mean_return, std_return = self._evaluate(agent, seed)
        finally:
            train_env.close()

        mlflow_logger.log_metrics(
            {"eval/mean_return": mean_return, "eval/std_return": std_return},
            step=self._cfg.total_timesteps,
        )
        return RunResult(seed, mean_return, std_return, DEFAULT_EVAL_EPISODES)

    def _build_callbacks(
        self, mlflow_logger: MlflowLogger, *, seed: int, run_idx: int, n_runs: int
    ) -> CallbackList:
        """Compose MlflowCallback (always) + ProgressCallback (if enabled) + extras."""
        callbacks: list[BaseCallback] = [MlflowCallback(mlflow_logger)]
        if self._progress:
            desc = (
                f"Run {run_idx}/{n_runs} (seed={seed})" if n_runs > 1 else f"Training (seed={seed})"
            )
            callbacks.append(ProgressCallback(desc=desc))
        callbacks.extend(self._extra_callbacks)
        return CallbackList(callbacks)

    def _evaluate(self, model: BaseAlgorithm, seed: int) -> tuple[float, float]:
        """Evaluate policy on the real MDP (τ forced to 0)."""
        eval_cfg = self._cfg.model_copy(
            update={"teleport": self._cfg.teleport.model_copy(update={"tau_0": 0.0})}
        )
        eval_env = make_vec_env(eval_cfg, n_envs=1, seed=seed + EVAL_SEED_OFFSET)
        try:
            mean_return, std_return = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=DEFAULT_EVAL_EPISODES,
                deterministic=True,
            )
        finally:
            eval_env.close()
        return float(mean_return), float(std_return)

    def _resolve_seeds(self) -> list[int]:
        """Derive `n_runs` consecutive seeds from the base seed."""
        base = 0 if self._cfg.seed is None else self._cfg.seed
        return [base + offset for offset in range(self._cfg.n_runs)]

    def _log_config(self, mlflow_logger: MlflowLogger) -> None:
        """Log config as flattened MLflow params + resolved-YAML artifact."""
        mlflow_logger.log_params(self._flatten_params(self._cfg.model_dump(mode="json")))
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "experiment_config.yaml"
            self._cfg.to_yaml(config_path)
            mlflow_logger.log_artifact(str(config_path))

    @staticmethod
    def _flatten_params(values: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
        """Recursively flatten nested dict to dotted keys for MLflow."""
        flat: dict[str, Any] = {}
        for key, value in values.items():
            dotted = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                flat.update(Trainer._flatten_params(value, dotted))
            else:
                flat[dotted] = value
        return flat
