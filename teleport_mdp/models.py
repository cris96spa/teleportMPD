from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeAlias

import optuna
from pydantic import BaseModel, ConfigDict, Field, model_validator

from teleport_mdp.constants import DEFAULT_MAP_NAME
from teleport_mdp.enums import (
    Algorithm,
    Curriculum,
    MetricType,
    OptimizationDirection,
    ParamType,
    TeleportDistribution,
)
from utils.configs import YamlBaseModel


class OptunaParamModel(BaseModel):
    """Parameter space definition for Optuna."""

    name: str = Field(description="Name of the hyperparameter.")
    kind: ParamType = Field(description="Type of the hyperparameter.")


class OptunaCategoricalParam(OptunaParamModel):
    """Categorical parameter space definition for Optuna."""

    choices: list[int | float | str | bool] = Field(
        description="List of choices for 'categorical' type."
    )
    kind: Literal[ParamType.CATEGORICAL] = Field(
        default=ParamType.CATEGORICAL, description="Type of the hyperparameter."
    )

    def get_optuna_suggestion(
        self, trial: optuna.Trial
    ) -> dict[str, int | float | str | bool | None]:
        """Get the Optuna suggestion for this categorical parameter."""
        suggestion = trial.suggest_categorical(self.name, self.choices)
        return {self.name: suggestion}


class OptunaIntParam(OptunaParamModel):
    """Integer parameter space definition for Optuna."""

    kind: Literal[ParamType.INT] = Field(
        default=ParamType.INT, description="Type of the hyperparameter."
    )
    low: int = Field(description="Lower bound of the integer parameter.")
    high: int = Field(description="Upper bound of the integer parameter.")
    step: int = Field(description="Step size for the integer parameter.", default=1)
    log: bool = Field(
        description="Whether to sample the integer parameter on a logarithmic scale.",
        default=False,
    )

    def get_optuna_suggestion(self, trial: optuna.Trial) -> dict[str, int]:
        """Get the Optuna suggestion for this integer parameter."""
        suggestion = trial.suggest_int(self.name, self.low, self.high, step=self.step, log=self.log)
        return {self.name: suggestion}

    @model_validator(mode="before")
    @classmethod
    def validate_step_and_log(cls, values):
        step = values.get("step")
        log = values.get("log")
        if step != 1 and log:
            raise ValueError(
                "Step must be 1 when used with logarithmic sampling, "
                f"got step={step} and log={log}."
            )
        return values


class OptunaFloatParam(OptunaParamModel):
    """Float parameter space definition for Optuna."""

    kind: Literal[ParamType.FLOAT] = Field(
        default=ParamType.FLOAT, description="Type of the hyperparameter."
    )
    low: float = Field(description="Lower bound of the float parameter.")
    high: float = Field(description="Upper bound of the float parameter.")
    step: float | None = Field(description="Step size for the float parameter.", default=None)
    log: bool = Field(
        description="Whether to sample the float parameter on a logarithmic scale.",
        default=False,
    )

    def get_optuna_suggestion(self, trial: optuna.Trial) -> dict[str, float]:
        """Get the Optuna suggestion for this float parameter."""
        suggestion = trial.suggest_float(
            self.name, self.low, self.high, step=self.step, log=self.log
        )
        return {self.name: suggestion}

    @model_validator(mode="before")
    @classmethod
    def validate_step_and_log(cls, values):
        step = values.get("step")
        log = values.get("log")
        if step is not None and log:
            raise ValueError(
                "Step must be None when used with logarithmic sampling, "
                f"got step={step} and log={log}."
            )
        return values


OptunaParam: TypeAlias = Annotated[
    OptunaCategoricalParam | OptunaIntParam | OptunaFloatParam,
    Field(discriminator="kind"),
]


class OptunaConfig(BaseModel):
    """Configuration for Optuna hyperparameter tuning."""

    enabled: bool = Field(
        description="Whether to enable Optuna hyperparameter tuning.", default=False
    )
    n_trials: int = Field(description="Number of optimization trials.", default=20)
    timeout: int | None = Field(description="Time limit in seconds for optimization.", default=None)
    direction: OptimizationDirection = Field(
        description="Direction of optimization (maximize or minimize).",
        default=OptimizationDirection.MAXIMIZE,
    )
    optimization_metric: MetricType | str = Field(
        description="Metric to optimize (e.g., 'accuracy', 'f1').",
        default=MetricType.ACCURACY,
    )
    param_space: list[OptunaParam] = Field(
        description="Parameter search space for Optuna.", default_factory=list
    )
    study_name: str | None = Field(description="Name of the Optuna study.", default=None)
    load_if_exists: bool = Field(
        description="Whether to load existing study if it exists.", default=False
    )


class EnvConfig(BaseModel):
    """Configuration for the (teleport) FrozenLake environment."""

    model_config = ConfigDict(extra="forbid")

    map_name: str | None = Field(
        default=None,
        description="Built-in map name (e.g. '4x4', '8x8'). Ignored when `desc` is set; when "
        "null (and no `desc`), a `size`x`size` random map is generated from `size`/`p`/`seed`.",
    )
    desc: list[str] | None = Field(
        default=None,
        description="Explicit map description; overrides `map_name` when provided.",
    )
    is_slippery: bool = Field(
        default=False, description="Whether moves are stochastic (slippery lake)."
    )
    size: int = Field(default=8, ge=2, description="Side length of a randomly generated map.")
    p: float = Field(
        default=0.8,
        gt=0.0,
        le=1.0,
        description="Probability a tile is frozen when generating a random map.",
    )
    seed: int | None = Field(default=None, description="Seed for random map generation.")
    render_mode: str | None = Field(
        default=None, description="Gymnasium render mode ('ansi', 'rgb_array', 'human')."
    )
    n_bins: int = Field(
        default=0,
        ge=0,
        description="Manhattan reward-shaping bins; 0 disables shaping.",
    )

    @model_validator(mode="after")
    def _require_map_source(self) -> Self:
        """Require an explicit map source so no map is picked silently.

        Valid sources: an explicit `map_name`, an explicit `desc`, or an explicit
        random-map parameter (`size`/`p`/`seed`, which builds a random map). An env
        block that sets none of these is rejected rather than defaulting quietly.

        Returns:
            The validated config instance.

        Raises:
            ValueError: If neither `map_name`, `desc`, nor a random-map parameter
                is set.
        """
        if self.map_name is not None or self.desc is not None:
            return self
        if {"size", "p", "seed"} & self.model_fields_set:
            return self
        raise ValueError(
            "No map source specified: set `map_name`, `desc`, or a random-map "
            "parameter (`size`/`p`/`seed`)."
        )


class TeleportConfig(BaseModel):
    """Configuration for the teleport mechanism (initial rate and distribution xi)."""

    model_config = ConfigDict(extra="forbid")

    tau_0: float = Field(
        default=0.0, ge=0.0, lt=1.0, description="Initial teleport rate tau, in [0, 1)."
    )
    distribution: TeleportDistribution = Field(
        default=TeleportDistribution.UNIFORM_NONTERMINAL,
        description="How the teleport distribution xi over states is built.",
    )
    custom_xi_path: Path | None = Field(
        default=None,
        description="Path to a custom xi distribution; required when distribution is 'custom'.",
    )

    @model_validator(mode="after")
    def _check_custom_xi(self) -> Self:
        """Ensure a custom distribution provides its xi source.

        Returns:
            The validated config instance.

        Raises:
            ValueError: If distribution is 'custom' but no `custom_xi_path` is set.
        """
        if self.distribution == TeleportDistribution.CUSTOM and self.custom_xi_path is None:
            raise ValueError("custom_xi_path must be set when distribution == 'custom'.")
        return self


class CurriculumConfig(BaseModel):
    """Configuration for the teleport-rate curriculum (thesis Algorithms 2 & 3).

    The static curriculum derives its per-update budget from the run length at
    runtime (task 07), so it needs no extra fields here. The dynamic curriculum
    requires the total state-visit shift budget `eps` and the per-update cap
    `eps_tau_max` (legacy `eps_shift`/`max_eps_model`).
    """

    model_config = ConfigDict(extra="forbid")

    kind: Curriculum = Field(
        default=Curriculum.NONE, description="Teleport-rate curriculum strategy."
    )
    eps: float | None = Field(
        default=None,
        gt=0.0,
        description="[dynamic] Total allowed gamma-discounted state-visit shift budget.",
    )
    eps_tau_max: float | None = Field(
        default=None,
        gt=0.0,
        description="[dynamic] Maximum model shift applied in a single update.",
    )

    @model_validator(mode="after")
    def _check_kind(self) -> Self:
        """Enforce that dynamic-only fields are present iff the curriculum is dynamic.

        Returns:
            The validated config instance.

        Raises:
            ValueError: If the dynamic curriculum is missing its budgets, or if
                dynamic-only fields are set for a non-dynamic curriculum.
        """
        dynamic_fields = {"eps": self.eps, "eps_tau_max": self.eps_tau_max}
        if self.kind == Curriculum.DYNAMIC:
            missing = sorted(k for k, v in dynamic_fields.items() if v is None)
            if missing:
                raise ValueError(f"Dynamic curriculum requires {missing} to be set.")
        else:
            extra = sorted(k for k, v in dynamic_fields.items() if v is not None)
            if extra:
                raise ValueError(
                    f"{extra} are only valid for the dynamic curriculum, not '{self.kind.value}'."
                )
        return self


class PPOConfig(BaseModel):
    """Stable-Baselines3 PPO hyperparameters (excluding `gamma`, held at experiment level)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[Algorithm.PPO] = Field(
        default=Algorithm.PPO, description="Algorithm discriminator."
    )

    # Rollout / minibatch geometry
    n_steps: int = Field(default=2048, gt=0, description="Rollout length per update.")
    batch_size: int = Field(default=64, gt=0, description="Minibatch size.")
    n_epochs: int = Field(default=10, gt=0, description="Optimization epochs per update.")

    # Advantage estimation
    gae_lambda: float = Field(
        default=0.95, ge=0.0, le=1.0, description="GAE lambda trade-off factor."
    )
    normalize_advantage: bool = Field(
        default=True, description="Normalize advantages in each minibatch."
    )

    # Core hyperparameters
    learning_rate: float = Field(default=3e-4, gt=0.0, description="Optimizer learning rate.")
    clip_range: float = Field(default=0.2, gt=0.0, description="PPO policy clip range.")
    clip_range_vf: float | None = Field(
        default=None, gt=0.0, description="Value-function clip range; None disables VF clipping."
    )

    # Loss coefficients
    ent_coef: float = Field(default=0.0, ge=0.0, description="Entropy bonus coefficient.")
    vf_coef: float = Field(default=0.5, ge=0.0, description="Value-function loss coefficient.")

    # Optimization
    max_grad_norm: float = Field(default=0.5, gt=0.0, description="Gradient clipping norm.")
    target_kl: float | None = Field(
        default=None, gt=0.0, description="KL threshold for early stopping; None disables."
    )

    # Generalized State-Dependent Exploration
    use_sde: bool = Field(default=False, description="Use gSDE instead of action noise.")
    sde_sample_freq: int = Field(
        default=-1, description="Noise matrix resample frequency (-1 = only at rollout start)."
    )

    # Logging
    stats_window_size: int = Field(
        default=100, gt=0, description="Episode window size for rollout statistics."
    )

    # Policy network
    policy_kwargs: dict[str, Any] | None = Field(
        default=None, description="Extra keyword arguments forwarded to the SB3 policy."
    )

    @model_validator(mode="after")
    def _check_batch_size_with_normalization(self) -> Self:
        """SB3 requires batch_size > 1 when normalize_advantage is enabled.

        Returns:
            The validated config instance.

        Raises:
            ValueError: If batch_size <= 1 and normalize_advantage is True.
        """
        if self.normalize_advantage and self.batch_size <= 1:
            raise ValueError("batch_size must be > 1 when normalize_advantage is True.")
        return self


class QLearningConfig(BaseModel):
    """Tabular Q-learning hyperparameters (excluding `gamma`)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[Algorithm.Q_LEARNING] = Field(
        default=Algorithm.Q_LEARNING, description="Algorithm discriminator."
    )
    alpha: float = Field(default=1.0, gt=0.0, description="Initial learning rate.")
    eps: float = Field(default=0.0, ge=0.0, description="Initial epsilon-greedy exploration.")
    episodes: int = Field(default=5000, gt=0, description="Number of training episodes.")
    status_step: int = Field(default=5000, gt=0, description="Interval between Q snapshots.")


class TMPIConfig(BaseModel):
    """Teleport Model Policy Iteration hyperparameters (tabular; see task 12)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[Algorithm.TMPI] = Field(
        default=Algorithm.TMPI, description="Algorithm discriminator."
    )
    threshold: float = Field(
        default=1e-6, gt=0.0, description="Convergence threshold on the value function."
    )
    max_iterations: int = Field(
        default=1000, gt=0, description="Maximum number of policy-iteration steps."
    )
    temperature: float = Field(default=1.0, gt=0.0, description="Softmax policy temperature.")


#: Discriminated union of the algorithm-specific configs. The `kind` tag selects the
#: variant, so the algorithm and its hyperparameters are one inseparable source of truth
#: (you cannot pick an algorithm without its matching block, nor leave a stray one behind).
AlgorithmConfig: TypeAlias = Annotated[
    PPOConfig | QLearningConfig | TMPIConfig,
    Field(discriminator="kind"),
]


class ExperimentConfig(YamlBaseModel):
    """Top-level, YAML-loadable configuration for a single experiment."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Experiment name (used to group MLflow runs).")
    algorithm: AlgorithmConfig = Field(
        description="Algorithm and its hyperparameters, selected by the `kind` tag "
        "(e.g. `kind: ppo`)."
    )
    gamma: float = Field(
        default=0.99,
        gt=0.0,
        le=1.0,
        description="Discount factor — single source of truth for the agent and the "
        "teleport schedule, to avoid drift between them.",
    )
    total_timesteps: int = Field(default=100_000, gt=0, description="Total training timesteps.")
    seed: int | None = Field(default=None, description="Base RNG seed for reproducibility.")
    n_runs: int = Field(default=1, gt=0, description="Number of seeded repeats for CI bands.")
    n_envs: int = Field(
        default=1, gt=0, description="Number of parallel training environments in the VecEnv."
    )
    env: EnvConfig = Field(default_factory=lambda: EnvConfig(map_name=DEFAULT_MAP_NAME))
    teleport: TeleportConfig = Field(default_factory=TeleportConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
    optuna: OptunaConfig | None = Field(
        default=None, description="Optional Optuna HPO configuration (see task 14)."
    )

    @model_validator(mode="after")
    def _check_curriculum(self) -> Self:
        """Validate the curriculum against the teleport rate and algorithm.

        Returns:
            The validated config instance.

        Raises:
            ValueError: If a teleport curriculum is requested without a positive
                initial rate, or alongside TMPI (which schedules its own rate).
        """
        if self.curriculum.kind != Curriculum.NONE:
            if self.teleport.tau_0 <= 0.0:
                raise ValueError("A teleport curriculum requires teleport.tau_0 > 0.")
            if self.algorithm.kind == Algorithm.TMPI:
                raise ValueError(
                    "TMPI schedules its own teleport rate; set curriculum.kind to 'none'."
                )
        return self
