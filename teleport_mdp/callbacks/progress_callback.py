from typing import Any

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter
from tqdm.auto import tqdm

DISPLAY_KEYS: dict[str, tuple[str, str]] = {
    "rollout/ep_rew_mean": ("rew", ".4f"),
    "rollout/ep_len_mean": ("len", ".1f"),
    "time/fps": ("fps", ".0f"),
}


class ProgressOutputFormat(KVWriter):
    """SB3 output format that forwards each `logger.dump` to a tqdm bar."""

    def __init__(
        self,
        pbar: "tqdm[Any]",
        display_keys: dict[str, tuple[str, str]] = DISPLAY_KEYS,
    ) -> None:
        self._pbar = pbar
        self._display_keys = display_keys

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...]],
        step: int = 0,
    ) -> None:
        """Update tqdm postfix with latest numeric diagnostics."""
        postfix: dict[str, str] = {}
        for key, (label, fmt) in self._display_keys.items():
            val = key_values.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                postfix[label] = format(float(val), fmt)
        if postfix:
            self._pbar.set_postfix(postfix, refresh=False)

    def close(self) -> None:
        """No-op; resources released by the owning callback."""


class ProgressCallback(BaseCallback):
    """tqdm progress bar with live metric display for SB3 training.

    Args:
        desc: Bar label (e.g. `"Run 2/3 (seed=6)"`).
        verbose: SB3 verbosity level.
    """

    def __init__(self, desc: str = "Training", verbose: int = 0) -> None:
        super().__init__(verbose)
        self._desc = desc
        self._pbar: "tqdm[Any] | None" = None
        self._output_format: ProgressOutputFormat | None = None

    def _on_training_start(self) -> None:
        """Create the bar and attach metric forwarding to the SB3 logger."""
        remaining = self.locals["total_timesteps"] - self.model.num_timesteps
        self._pbar = tqdm(total=remaining, unit="step", desc=self._desc, dynamic_ncols=True)
        self._output_format = ProgressOutputFormat(self._pbar)
        self.logger.output_formats.append(self._output_format)

    def _on_step(self) -> bool:
        """Advance bar by num_envs."""
        if self._pbar is not None:
            self._pbar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self) -> None:
        """Detach output format and close the bar."""
        if self._output_format is not None:
            if self._output_format in self.logger.output_formats:
                self.logger.output_formats.remove(self._output_format)
            self._output_format.close()
            self._output_format = None
        if self._pbar is not None:
            self._pbar.refresh()
            self._pbar.close()
            self._pbar = None
