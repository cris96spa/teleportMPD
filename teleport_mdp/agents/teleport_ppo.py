"""PPO with a teleport-rate curriculum (thesis Algorithms 2 & 3).

``TeleportPPO`` ties together the two math-bearing pieces:

- :class:`~teleport_mdp.agents.teleport_rollout_buffer.TeleportRolloutBuffer` truncates
  GAE on teleport transitions, and
- a :class:`~teleport_mdp.curriculum.scheduler.TeleportScheduler` lowers ``tau`` toward 0
  over training.

Each PPO update it measures the policy shift ``D_inf = max_s ||pi'(.|s) - pi(.|s)||_1``
on the rollout observations (before vs. after ``train()``), asks the scheduler for the
next ``tau``, pushes it to every environment, and records the teleport diagnostics to
the SB3 logger (which the :class:`~teleport_mdp.callbacks.MlflowCallback` streams to
MLflow). SB3's ``train()`` is wrapped, never forked, to keep upgrades easy.
"""

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import obs_as_tensor

from teleport_mdp.agents.teleport_rollout_buffer import TeleportRolloutBuffer
from teleport_mdp.curriculum.scheduler import TeleportScheduler


class TeleportPPO(PPO):
    """PPO subclass that runs a teleport-rate curriculum on a teleport MDP.

    Args:
        *args: Positional arguments forwarded to :class:`stable_baselines3.PPO`.
        scheduler: The teleport-rate scheduler (static or dynamic) driving the curriculum.
        curriculum_log_interval: Record teleport diagnostics every this many updates.
        convergence_threshold: If set, stop early once ``tau == 0`` and the policy shift
            ``D_inf`` falls below this threshold.
        **kwargs: Keyword arguments forwarded to :class:`stable_baselines3.PPO`. The
            ``rollout_buffer_class`` defaults to :class:`TeleportRolloutBuffer`.
    """

    def __init__(
        self,
        *args: object,
        scheduler: TeleportScheduler,
        curriculum_log_interval: int = 1,
        convergence_threshold: float | None = None,
        **kwargs: object,
    ) -> None:
        kwargs.setdefault("rollout_buffer_class", TeleportRolloutBuffer)
        super().__init__(*args, **kwargs)
        self._scheduler = scheduler
        self._curriculum_log_interval = curriculum_log_interval
        self._convergence_threshold = convergence_threshold

    @property
    def tau(self) -> float:
        """Current teleport rate, read from the first training environment."""
        return float(self.env.get_attr("teleport_probability")[0])

    def _set_tau(self, tau: float) -> None:
        """Push a new teleport rate to every training environment."""
        self.env.env_method("update_tau", tau)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "TeleportPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "TeleportPPO":
        """Train with PPO while annealing the teleport rate each update.

        Mirrors SB3's ``OnPolicyAlgorithm.learn`` loop, but wraps ``train()`` to measure
        the per-update policy shift and applies the scheduler before logging, so the
        teleport diagnostics are dumped together with the rollout metrics.

        Args:
            total_timesteps: Total environment steps to train for.
            callback: SB3 callback(s) to run during training.
            log_interval: Dump SB3 logs every this many updates.
            tb_log_name: Run name for the SB3 logger.
            reset_num_timesteps: Whether to reset the timestep counter.
            progress_bar: Whether SB3 should display its own progress bar.

        Returns:
            This trained agent.
        """
        iteration = 0
        total_timesteps, callback = self._setup_learn(
            total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar
        )
        callback.on_training_start(locals(), globals())
        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(
                self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps
            )
            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            policy_shift = self._train_with_policy_shift()
            self._update_teleport_rate(policy_shift, iteration)

            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self.dump_logs(iteration)

            if self._has_converged(policy_shift):
                break

        callback.on_training_end()
        return self

    def _train_with_policy_shift(self) -> float:
        """Run one PPO update and return the policy shift it induced.

        Snapshots the action distribution on the rollout observations before and after
        ``train()`` and returns ``D_inf = max_s ||pi'(.|s) - pi(.|s)||_1`` (legacy
        ``calculate_d_inf_distance``), kept on the same observation batch for consistency.

        Returns:
            The infinity-norm policy shift over the rollout observations.
        """
        observations = self.rollout_buffer.observations
        flat = observations.reshape((-1, *observations.shape[2:]))
        obs_tensor = obs_as_tensor(flat, self.device)

        with th.no_grad():
            old_probs = self.policy.get_distribution(obs_tensor).distribution.probs
        self.train()
        with th.no_grad():
            new_probs = self.policy.get_distribution(obs_tensor).distribution.probs

        l1_per_state = th.norm(old_probs - new_probs, p=1, dim=-1)
        return float(th.max(l1_per_state).item())

    def _update_teleport_rate(self, policy_shift: float, iteration: int) -> None:
        """Apply the scheduler, push the new ``tau`` to the envs, and log diagnostics.

        Args:
            policy_shift: The update's policy shift ``D_inf``.
            iteration: The current update index (for the log interval).
        """
        tau = self.tau
        tau_prime = self._scheduler.next_tau(tau, policy_shift=policy_shift)
        if tau_prime != tau:
            self._set_tau(tau_prime)

        if self._curriculum_log_interval and iteration % self._curriculum_log_interval == 0:
            self.logger.record("teleport/tau", tau)
            self.logger.record("teleport/tau_prime", tau_prime)
            self.logger.record("teleport/d_inf", policy_shift)

    def _has_converged(self, policy_shift: float) -> bool:
        """Whether to stop early: ``tau`` annealed to 0 and the policy has settled."""
        if self._convergence_threshold is None:
            return False
        return self.tau <= 0.0 and policy_shift < self._convergence_threshold

    def collect_rollouts(
        self,
        env: object,
        callback: BaseCallback,
        rollout_buffer: TeleportRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Collect a rollout, forwarding ``infos`` so the buffer can flag teleports.

        This is SB3 2.9.0's ``OnPolicyAlgorithm.collect_rollouts`` with one change: the
        per-step ``infos`` are passed to :meth:`TeleportRolloutBuffer.add` so it can mark
        teleport transitions. SB3's timeout/bootstrap handling is left intact.

        Args:
            env: The vectorized training environment.
            callback: The SB3 callback to drive during collection.
            rollout_buffer: The teleport rollout buffer to fill.
            n_rollout_steps: Number of steps to collect per environment.

        Returns:
            ``True`` if the rollout completed, ``False`` if a callback stopped it.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        assert isinstance(rollout_buffer, TeleportRolloutBuffer), (
            "TeleportPPO requires a TeleportRolloutBuffer."
        )
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high
                    )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function (SB3 GitHub issue #633).
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                infos=infos,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())
        callback.on_rollout_end()

        return True
