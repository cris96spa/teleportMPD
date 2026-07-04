from logging import getLogger

from dotenv import load_dotenv

from teleport_mdp.models import ExperimentConfig
from teleport_mdp.tabular.trainer import TabularRunResult, TabularTrainer
from utils.commands import command
from utils.configs import MlflowLoggerConfig
from utils.logger import setup_logger

logger = getLogger(__name__)


@command
def main(config: ExperimentConfig) -> None:
    """Run a tabular experiment (Q-learning or TMPI) end-to-end from the CLI.

    Loads the MLflow logger config, trains and exactly evaluates every seeded run,
    and reports each run's final performance `J` on the real MDP.

    Args:
        config: The validated experiment configuration loaded from `--config`.
    """
    load_dotenv()
    setup_logger()
    logger.info("Starting tabular pipeline for experiment %r...", config.name)

    mlflow_logger_config = MlflowLoggerConfig.from_yaml()
    trainer = TabularTrainer(config, mlflow_config=mlflow_logger_config)
    results: list[TabularRunResult] = trainer.run()

    for result in results:
        logger.info("seed=%d -> eval performance %.4f", result.seed, result.performance)


if __name__ == "__main__":
    main()
