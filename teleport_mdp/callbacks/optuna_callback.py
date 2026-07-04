import numpy as np
import optuna
from stable_baselines3.common.callbacks import BaseCallback


class OptunaPruningCallback(BaseCallback):
    """Report the running episode-return mean to an Optuna trial and flag pruning.

    Every `report_freq` timesteps the callback reports the mean return over the
    agent's recent-episode buffer to `trial` and asks Optuna whether the trial
    should be pruned. It never raises: a prune verdict only sets `pruned` and stops
    the current run by returning `False` from `_on_step`, so the surrounding
    :class:`~teleport_mdp.trainer.Trainer` unwinds cleanly (env closed, MLflow run
    finished); the optimizer inspects `pruned` afterwards and raises
    `optuna.TrialPruned` at the trial boundary. Report steps are ordinal (the k-th
    report), so trials stay comparable even though each seeded run restarts its own
    timestep count.

    Args:
        trial: The Optuna trial to report intermediate values to.
        report_freq: Minimum number of timesteps between two reports.
        verbose: SB3 verbosity level.
    """

    def __init__(self, trial: optuna.Trial, report_freq: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._trial = trial
        self._report_freq = report_freq
        self._report_step = 0
        self._last_report_timestep = 0
        self.pruned = False

    def _on_training_start(self) -> None:
        """Anchor the report cadence to this run's starting timestep."""
        self._last_report_timestep = self.num_timesteps

    def _on_step(self) -> bool:
        """Report the running return every `report_freq` steps; stop once pruned.

        Returns:
            `False` after a prune verdict (halting this run), else `True`.
        """
        if self.pruned:
            return False
        if self.num_timesteps - self._last_report_timestep < self._report_freq:
            return True
        self._last_report_timestep = self.num_timesteps

        recent_episodes = self.model.ep_info_buffer
        if not recent_episodes:
            return True
        mean_return = float(np.mean([episode["r"] for episode in recent_episodes]))
        self._trial.report(mean_return, self._report_step)
        self._report_step += 1
        if self._trial.should_prune():
            self.pruned = True
            return False
        return True
