from collections.abc import Callable
from typing import Any, NamedTuple

from teleport_mdp.curriculum.scheduler import (
    DynamicTeleportScheduler,
    StaticTeleportScheduler,
    TeleportScheduler,
)
from teleport_mdp.enums import Algorithm, Curriculum
from teleport_mdp.environments.factory import build_xi, make_env, wrap_tmdp
from teleport_mdp.models import ExperimentConfig, QLearningConfig, TMPIConfig
from teleport_mdp.tabular import (
    TMPI,
    QLearner,
    compute_value_function,
    dense_model,
    get_policy,
)
from teleport_mdp.tabular.model_functions import FloatArray
from teleport_mdp.wrappers.tmdp import TMDP
from utils.configs import MlflowLoggerConfig
from utils.experiment_logger import MlflowLogger

LoggerFactory = Callable[[MlflowLoggerConfig], MlflowLogger]


class TabularRunResult(NamedTuple):
    """Summary of a single seeded tabular run."""

    seed: int
    performance: float


class TabularTrainer:
    """Runs a tabular experiment (Q-learning or TMPI) across seeds with MLflow logging.

    Every run is evaluated the same exact, model-based way: the final policy's performance
    `J = mu . V` on the real MDP (teleport rate forced to 0).

    Args:
        cfg: The experiment configuration; its `algorithm.kind` selects the learner.
        mlflow_config: MLflow logger config; loaded from default YAML when omitted.
        logger_factory: Builds the context-managed :class:`MlflowLogger` per run.
        progress: Whether to show a tqdm progress bar during training.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        mlflow_config: MlflowLoggerConfig | None = None,
        logger_factory: LoggerFactory = MlflowLogger,
        *,
        progress: bool = True,
    ) -> None:
        self._cfg = cfg
        self._mlflow_config = mlflow_config or MlflowLoggerConfig.from_yaml()
        self._logger_factory = logger_factory
        self._progress = progress

    def run(self) -> list[TabularRunResult]:
        """Execute all seeded runs and return their evaluation results."""
        seeds = self._resolve_seeds()
        n_runs = len(seeds)
        results: list[TabularRunResult] = []
        for idx, seed in enumerate(seeds, 1):
            run_config = self._mlflow_config.model_copy(
                update={
                    "experiment_name": self._cfg.name,
                    "run_name": f"{self._cfg.name}_seed_{seed}",
                }
            )
            desc = f"Run {idx}/{n_runs} (seed={seed})" if n_runs > 1 else f"seed={seed}"
            with self._logger_factory(run_config) as mlflow_logger:
                results.append(self._run_single(seed, mlflow_logger, desc))
        return results

    def _run_single(self, seed: int, mlflow_logger: MlflowLogger, desc: str) -> TabularRunResult:
        """Train and exactly evaluate one seeded run."""
        self._log_config(mlflow_logger)
        env = self._build_env()
        try:
            pi = self._train(env, seed, mlflow_logger, desc)
            performance = self._evaluate(env, pi)
        finally:
            env.close()

        mlflow_logger.log_metrics({"eval/performance": performance}, step=0)
        return TabularRunResult(seed, performance)

    def _train(self, env: TMDP, seed: int, mlflow_logger: MlflowLogger, desc: str) -> FloatArray:
        """Dispatch to the configured tabular learner and return the final policy.

        Args:
            env: The tabular teleport MDP to learn on.
            seed: Seed for the Q-learner's action RNG (ignored by TMPI, which is
                deterministic).
            mlflow_logger: Metric sink passed through to the learner.
            desc: Progress-bar label for this run.

        Returns:
            The final policy `[nS, nA]` (greedy for Q-learning, the mixed policy
            for TMPI).

        Raises:
            ValueError: If the configured algorithm is not a tabular one
                (Q-learning or TMPI).
        """
        algorithm = self._cfg.algorithm
        if isinstance(algorithm, QLearningConfig):
            scheduler = self._build_scheduler(algorithm)
            result = QLearner(algorithm, self._cfg.gamma, seed=seed).train(
                env, scheduler=scheduler, logger=mlflow_logger, progress=self._progress, desc=desc
            )
            return get_policy(result.q, deterministic=True)
        if isinstance(algorithm, TMPIConfig):
            return (
                TMPI(algorithm, self._cfg.gamma)
                .optimize(env, logger=mlflow_logger, progress=self._progress, desc=desc)
                .pi
            )
        raise ValueError(
            f"TabularTrainer does not handle algorithm '{self._cfg.algorithm.kind.value}'; "
            f"use it only for {Algorithm.Q_LEARNING.value} or {Algorithm.TMPI.value}."
        )

    def _evaluate(self, env: TMDP, pi: FloatArray) -> float:
        """Exact performance `J = mu . V` of `pi` on the real MDP (teleport rate 0)."""
        p, reward, mu, _ = dense_model(env)
        v = compute_value_function(p, reward, pi, self._cfg.gamma)
        return float(mu @ v)

    def _build_env(self) -> TMDP:
        """Build the tabular teleport MDP (raw Discrete states, no SB3 wrappers)."""
        base = make_env(self._cfg.env)
        xi = build_xi(base, self._cfg.teleport)
        return wrap_tmdp(base, xi, self._cfg.teleport.tau_0)

    def _build_scheduler(self, algorithm: QLearningConfig) -> TeleportScheduler | None:
        """Build the Q-learning teleport-rate scheduler from the curriculum config.

        Unlike the PPO scheduler registry, the static budget here is the number of
        status steps (`episodes // status_step`), matching when the Q-learner ticks
        the curriculum.
        """
        cfg = self._cfg
        if cfg.curriculum.kind == Curriculum.STATIC:
            n_updates = max(1, algorithm.episodes // algorithm.status_step)
            return StaticTeleportScheduler(cfg.gamma, cfg.teleport.tau_0, n_updates)
        if cfg.curriculum.kind == Curriculum.DYNAMIC:
            assert cfg.curriculum.eps is not None
            assert cfg.curriculum.eps_tau_max is not None
            return DynamicTeleportScheduler(
                cfg.gamma, eps=cfg.curriculum.eps, eps_tau_max=cfg.curriculum.eps_tau_max
            )
        return None

    def _resolve_seeds(self) -> list[int]:
        """Derive `n_runs` consecutive seeds from the base seed."""
        base = 0 if self._cfg.seed is None else self._cfg.seed
        return [base + offset for offset in range(self._cfg.n_runs)]

    def _log_config(self, mlflow_logger: MlflowLogger) -> None:
        """Log the config as flattened, dotted MLflow params."""
        mlflow_logger.log_params(_flatten_params(self._cfg.model_dump(mode="json")))


def _flatten_params(values: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
    """Recursively flatten a nested dict to dotted keys for MLflow params."""
    flat: dict[str, Any] = {}
    for key, value in values.items():
        dotted = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            flat.update(_flatten_params(value, dotted))
        else:
            flat[dotted] = value
    return flat
