from logging import getLogger

from dotenv import load_dotenv

from teleport_mdp.models import ExperimentConfig
from teleport_mdp.trainer import RunResult, Trainer
from utils.commands import command
from utils.configs import MlflowLoggerConfig
from utils.logger import setup_logger

logger = getLogger(__name__)


@command
def main(config: ExperimentConfig) -> None:
    """Run a configured experiment end-to-end from the CLI.

    Loads the MLflow logger config, trains and evaluates every seeded run, and
    reports the final evaluation return for each.

    Args:
        config: The validated experiment configuration loaded from `--config`.
    """
    load_dotenv()
    setup_logger()
    logger.info("Starting training pipeline for experiment %r...", config.name)

    mlflow_logger_config = MlflowLoggerConfig.from_yaml()
    trainer = Trainer(config, mlflow_config=mlflow_logger_config)
    results: list[RunResult] = trainer.run()

    for result in results:
        logger.info(
            "seed=%d -> eval return %.4f +/- %.4f over %d episodes",
            result.seed,
            result.mean_return,
            result.std_return,
            result.n_eval_episodes,
        )


if __name__ == "__main__":
    main()
