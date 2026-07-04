from logging import getLogger

from dotenv import load_dotenv

from teleport_mdp.models import ExperimentConfig
from teleport_mdp.optimizer import HyperparameterOptimizer
from utils.commands import command
from utils.configs import MlflowLoggerConfig
from utils.logger import setup_logger

logger = getLogger(__name__)


@command
def main(config: ExperimentConfig) -> None:
    """Run an Optuna hyperparameter study from the CLI.

    Requires an enabled `optuna` block in the config (search space, trial count,
    direction). Each trial is trained and evaluated through the standard PPO
    pipeline; the best trial's config is written to `best_config.yaml`.

    Args:
        config: The validated experiment configuration loaded from `--config`.
    """
    load_dotenv()
    setup_logger()
    logger.info("Starting Optuna study for experiment %r...", config.name)

    mlflow_logger_config = MlflowLoggerConfig.from_yaml()
    study = HyperparameterOptimizer(config, mlflow_config=mlflow_logger_config).optimize()

    logger.info(
        "Study %r finished: best objective %.4f from trial %d.",
        study.study_name,
        study.best_value,
        study.best_trial.number,
    )


if __name__ == "__main__":
    main()
