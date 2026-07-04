# Running experiments

Everything is driven from YAML: an experiment is one `ExperimentConfig` file, launched
through a CLI command. There is no code to edit to run a new variant — you write a config
and point a command at it. This page is a self-contained guide to launching runs.

## Prerequisites

```bash
make dev          # create the venv and install everything (once)
make mlflow-up    # start the local MLflow tracking server (UI on http://localhost:5002)
```

Every run streams metrics to MLflow **live during training**, so the tracking server must
be up first. Its address lives in `configs/mlflow_logger.yaml` (`tracking_uri`). Stop it
later with `make mlflow-down`.

## The three entry points

All commands share the form `uv run python -m teleport_mdp <command> --config <path>`:

| Command | Track | Algorithms |
| --- | --- | --- |
| `run_experiment` | SB3 PPO | vanilla PPO, Static-Teleport, Dynamic-Teleport |
| `run_tabular` | numpy (exact) | Q-learning, TMPI |
| `run_optuna` | Optuna HPO | a study over any PPO config |

Ready-to-run configs live in `configs/commands/`. To reproduce the baselines:

```bash
# Vanilla PPO baseline on 8x8 FrozenLake
uv run python -m teleport_mdp run_experiment --config configs/commands/frozen_lake_ppo.yaml

# Teleport curricula (PPO): Static-Teleport and Dynamic-Teleport
uv run python -m teleport_mdp run_experiment --config configs/commands/frozen_lake_st_ppo.yaml
uv run python -m teleport_mdp run_experiment --config configs/commands/frozen_lake_dt_ppo.yaml

# Tabular track: exact Q-learning and TMPI
uv run python -m teleport_mdp run_tabular --config configs/commands/frozen_lake_q_learning.yaml
uv run python -m teleport_mdp run_tabular --config configs/commands/frozen_lake_tmpi.yaml
```

Each run trains `n_runs` seeded repeats and prints the final evaluation return per seed;
the PPO track always evaluates on the **real** MDP (teleport rate forced to `0`).

## Anatomy of a config

A minimal, complete PPO experiment is just a handful of blocks:

```yaml
name: my_first_experiment   # groups the runs in MLflow
gamma: 0.99                 # discount, shared by the agent and the teleport schedule
total_timesteps: 200000
seed: 42
n_runs: 1                  # seeded repeats -> confidence bands
n_envs: 1                  # parallel training environments

algorithm:                 # a `kind`-discriminated union: ppo | q_learning | tmpi
  kind: ppo
  n_steps: 2048
  batch_size: 64
  learning_rate: 0.0003
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.0

env:                       # pick exactly one map source: map_name | desc | size
  map_name: "8x8"          # built-in "4x4"/"8x8", or `desc: ["SFFG", ...]`, or `size: 16`
  is_slippery: false
  n_bins: 0                # Manhattan reward shaping; 0 = original sparse reward

teleport:
  tau_0: 0.0               # initial teleport rate in [0, 1); 0 = plain MDP (no teleport)
  distribution: uniform_nonterminal   # uniform | uniform_nonterminal | custom

curriculum:
  kind: none               # none | static | dynamic (dynamic also needs eps/eps_tau_max)
```

Configs are validated on load with `extra="forbid"`, so a misspelled key fails fast rather
than being silently ignored. The fully-resolved config is logged to MLflow as both flattened
params and a YAML artifact, so every run is reproducible from its own record.

Turning the baseline into a **teleport curriculum** is a two-line change: set a positive
`teleport.tau_0` and a `curriculum.kind`. Static-Teleport (`static`) anneals `τ` linearly to
`0` over the run; Dynamic-Teleport (`dynamic`) lowers `τ` in proportion to the policy shift and
additionally needs a budget:

```yaml
teleport:
  tau_0: 0.9
curriculum:
  kind: dynamic
  eps: 1.0            # total state-visit shift budget
  eps_tau_max: 0.05   # cap on a single teleport-rate step
```

To make a **vanilla vs. curriculum** comparison fair on the sparse map, give both agents the
same dense signal by setting `env.n_bins` (the thesis used `3` or `10`); `0` keeps the
original sparse reward.

## Hyperparameter optimization (Optuna)

`run_optuna` turns any PPO config into a search by adding an `optuna` block. Each trial samples
the search space, bakes the values into a copy of the config, and trains + evaluates it through
the exact same pipeline as `run_experiment`. A `MedianPruner` stops weak trials early from their
intermediate returns, and the best trial's config is written to `best_config.yaml`.

```bash
uv run python -m teleport_mdp run_optuna --config configs/commands/frozen_lake_ppo_optuna.yaml
```

The search space lives entirely in YAML. Each parameter's `name` is a **dotted path** into the
config, so any field is tunable:

```yaml
optuna:
  enabled: true
  n_trials: 20
  direction: maximize          # objective = mean evaluation return
  study_name: frozen_lake_ppo_hpo
  param_space:
    - name: algorithm.learning_rate   # dotted path -> config.algorithm.learning_rate
      kind: float
      low: 1.0e-5
      high: 1.0e-2
      log: true                # sample log-uniformly
    - name: algorithm.n_steps
      kind: categorical
      choices: [512, 1024, 2048]
    - name: teleport.tau_0      # curricula are tunable too
      kind: float
      low: 0.0
      high: 0.95
```

The objective is averaged over the config's `n_runs` seeds, so a lucky seed cannot win the
study. When it finishes, run the winner directly — `best_config.yaml` is a normal experiment
config (the `optuna` block is stripped):

```bash
uv run python -m teleport_mdp run_experiment --config best_config.yaml
```

## Writing your own

1. Copy the closest config in `configs/commands/` and edit the blocks above.
2. Launch it with the matching command (`run_experiment` for PPO, `run_tabular` for
   Q-learning/TMPI).
3. Watch it live in the MLflow UI at http://localhost:5002 — runs are grouped by `name`, one
   per seed, showing streamed metrics (`rollout/ep_rew_mean`, `eval/mean_return`), the flattened
   params, and the resolved config artifact.
