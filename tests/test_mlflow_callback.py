from typing import Any, cast

from teleport_mdp.callbacks.mlflow_callback import (
    MlflowCallback,
    _finite_scalars,
    _MlflowOutputFormat,
)
from teleport_mdp.environments.factory import make_vec_env
from teleport_mdp.models import EnvConfig, ExperimentConfig
from utils.experiment_logger import MlflowLogger


class _RecordingLogger:
    """Minimal stand-in for `MlflowLogger`, capturing forwarded metrics.

    It records `log_metrics` calls so tests can assert what the callback funnels
    through the logger without needing a live MLflow backend.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, float], int]] = []

    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        """Record a forwarded metrics payload.

        Args:
            metrics: The metric name/value mapping.
            step: The training step.
        """
        self.calls.append((dict(metrics), step))


def _as_logger(stub: _RecordingLogger) -> MlflowLogger:
    """Cast the duck-typed stub to the `MlflowLogger` type the callback expects.

    Args:
        stub: The recording stand-in.

    Returns:
        The same object, typed as `MlflowLogger`.
    """
    return cast(MlflowLogger, stub)


# region filtering / forwarding


def test_finite_scalars_drops_non_metric_values():
    """Strings, bools and non-finite numbers are not valid MLflow metrics."""
    payload: dict[str, Any] = {
        "rollout/ep_rew_mean": 0.25,
        "time/fps": 1234,
        "train/loss": float("nan"),
        "train/inf": float("inf"),
        "label": "skip-me",
        "flag": True,
    }
    assert _finite_scalars(payload) == {"rollout/ep_rew_mean": 0.25, "time/fps": 1234.0}


def test_output_format_forwards_through_logger():
    """The output format funnels finite metrics through the injected logger."""
    stub = _RecordingLogger()
    fmt = _MlflowOutputFormat(_as_logger(stub))
    fmt.write({"rollout/ep_rew_mean": 1.0, "label": "x"}, {}, step=7)
    assert stub.calls == [({"rollout/ep_rew_mean": 1.0}, 7)]


def test_output_format_skips_empty_payload():
    """With nothing numeric to log, the logger is not called at all."""
    stub = _RecordingLogger()
    fmt = _MlflowOutputFormat(_as_logger(stub))
    fmt.write({"label": "x", "flag": True}, {}, step=1)
    assert stub.calls == []


# endregion

# region SB3 integration


def test_callback_streams_ppo_metrics_through_logger():
    """A short PPO run streams per-rollout metrics through the logger and cleans up."""
    from stable_baselines3 import PPO

    cfg = ExperimentConfig(
        name="callback-smoke",
        algorithm={"kind": "ppo", "n_steps": 16, "batch_size": 16, "n_epochs": 1},
        env=EnvConfig(map_name="4x4", is_slippery=False),
        seed=0,
    )
    stub = _RecordingLogger()
    vec_env = make_vec_env(cfg, n_envs=1, max_episode_steps=50)
    try:
        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=cfg.algorithm.n_steps,
            batch_size=cfg.algorithm.batch_size,
            n_epochs=cfg.algorithm.n_epochs,
            gamma=cfg.gamma,
            device="cpu",
        )
        model.learn(total_timesteps=48, callback=MlflowCallback(_as_logger(stub)))
        # The output format is detached once training ends.
        assert all(not isinstance(f, _MlflowOutputFormat) for f in model.logger.output_formats)
    finally:
        vec_env.close()

    assert stub.calls, "Expected metrics to be streamed through the logger."
    logged_keys = {key for metrics, _ in stub.calls for key in metrics}
    assert "rollout/ep_rew_mean" in logged_keys


# endregion
