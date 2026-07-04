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
the PPO track always evaluates on the **real** MDP (teleport rate forced to `0`). Set
`eval_freq` to also evaluate *during* training every `eval_freq` timesteps — each such
evaluation streams `eval/mean_reward` to MLflow, giving a learning curve instead of a single
end-of-run point (the final `eval/mean_return` summary is logged regardless).

## Anatomy of a config

A minimal, complete PPO experiment is just a handful of blocks:

```yaml
name: my_first_experiment   # groups the runs in MLflow
gamma: 0.99                 # discount, shared by the agent and the teleport schedule
total_timesteps: 200000
eval_freq: 20000           # optional: real-MDP eval every N steps during training (omit = final eval only)
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
  eps: 80.0           # total state-visit shift budget (mind the scale, see below)
  eps_tau_max: 4.0    # cap on a single teleport-rate step
```

Mind the **scale of `eps`**: each update the policy shift consumes `eps_pi = γ/(1−γ)·D_inf` of
the budget, and only the remainder lowers `τ`. That `γ/(1−γ)` factor is `99` at `γ=0.99`, so a
realistic per-update shift `D_inf ≈ 0.4` already spends `≈40` — `eps` must be `O(10–100)`, not
`O(1)`, or `τ` never anneals at all. Training logs a warning if it detects `τ` still pinned at
its initial value after several updates.

To make a **vanilla vs. curriculum** comparison fair on the sparse map, give both agents the
same dense signal by setting `env.n_bins` (the thesis used `3` or `10`); `0` keeps the
original sparse reward.

## Sizing `n_steps` and `n_envs`

Each PPO update collects a batch of **`n_steps × n_envs`** transitions (`n_steps` per env, all
`n_envs` stepping in lockstep), then reuses it for `n_epochs` passes in `batch_size` minibatches.
Three things follow from that product:

- **`batch_size` should divide `n_steps × n_envs`.** Otherwise the last minibatch is a ragged
  remainder and SB3 warns. The configs use `2048 × 4 = 8192` transitions per update, cleanly
  split by `batch_size: 64`.
- **The number of updates is `total_timesteps / (n_steps × n_envs)`**, not `total_timesteps`.
  Raising either `n_steps` or `n_envs` gives *fewer, larger* updates for the same budget — and
  the **static curriculum anneals `τ` over exactly that many updates** (the dynamic one likewise
  ticks once per update), so more envs ⇒ coarser `τ` steps. If you scale `n_envs` up and want the
  same curriculum granularity, scale `total_timesteps` up to match.
- **`n_envs` buys throughput and sample diversity, not extra gradient steps.** More parallel envs
  fill the batch faster in wall-clock and decorrelate it, but each update still consumes
  `n_steps × n_envs` of the budget. `n_steps` trades update frequency against on-policy batch
  size and GAE quality — longer rollouts give steadier advantages but a staler within-batch
  policy.

Rule of thumb for these FrozenLake tasks: keep `n_steps × n_envs` in the low thousands, pick
`batch_size` as a divisor of it, and set `n_envs` near your core count — then remember the update
count (hence curriculum granularity) moves *inversely* with it. Cadences expressed in timesteps —
`total_timesteps` and `eval_freq` — are unaffected by `n_envs`; the trainer converts `eval_freq`
into vectorized steps internally.

## Monitoring a run

A PPO run reports on two channels while it trains:

- **Terminal progress bar** (`tqdm`): live `rew` (mean episode reward), `len` (mean episode
  length) and `fps`, plus **`tau`** (current teleport rate) and **`d_inf`** (the per-update
  policy shift) whenever a curriculum is active — so you can watch `τ` anneal in real time.
- **MLflow** (live, at http://localhost:5002): `rollout/ep_rew_mean`, the curriculum series
  `teleport/tau` and `teleport/d_inf`, the final `eval/mean_return`, and — when `eval_freq` is
  set — a periodic **`eval/mean_reward`** learning curve evaluated on the real MDP.

Two guardrails surface silent misconfigurations at runtime:

- A **dynamic curriculum that never anneals** (its `eps` is too small for the `γ/(1−γ)` scale,
  so the policy shift consumes the whole budget every update) logs a one-off warning naming the
  observed `D_inf`, the implied `eps_pi`, and the configured `eps`.
- `eval_freq` is interpreted in **timesteps**: with `n_envs > 1` the vectorized rollout advances
  all envs in lockstep, so the trainer divides `eval_freq` by `n_envs` internally to keep the
  cadence in timesteps rather than in (fewer) vectorized steps.

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
   per seed, showing streamed metrics (`rollout/ep_rew_mean`, `eval/mean_reward` when `eval_freq`
   is set, `eval/mean_return`), the flattened params, and the resolved config artifact.

## Reproducing the thesis experiments

`configs/thesis/` holds the six methods of the thesis FrozenLake comparison, all on **one
shared task** so their curves line up: a fixed 30×30 random map (map seed `2999`), not
slippery, sparse reward, with initial `τ = 0.2` for every curriculum method.

| Method | File | Command |
| --- | --- | --- |
| PPO (vanilla) | `frozen_lake_ppo.yaml` | `run_experiment` |
| Static-Teleport PPO | `frozen_lake_st_ppo.yaml` | `run_experiment` |
| Dynamic-Teleport PPO | `frozen_lake_dt_ppo.yaml` | `run_experiment` |
| Q-learning (vanilla) | `frozen_lake_q_learning.yaml` | `run_tabular` |
| Static-Teleport Q-learning | `frozen_lake_st_q_learning.yaml` | `run_tabular` |
| TMPI | `frozen_lake_tmpi.yaml` | `run_tabular` |

```bash
uv run python -m teleport_mdp run_experiment --config configs/thesis/frozen_lake_st_ppo.yaml
uv run python -m teleport_mdp run_tabular    --config configs/thesis/frozen_lake_tmpi.yaml
```

The **experimental design** (map, sparse reward, teleport distribution, γ, initial τ) is
faithful to the thesis; the per-algorithm hyperparameters were re-tuned for the ported
implementations (SB3 PPO, exact TMPI, the numpy `QLearner`), since the legacy hand-rolled
`pol_lr`/`model_lr`/`temp`/`episodes` do not transfer. A few deliberate choices:

- The thesis swept `env.n_bins` over `{0, 1, 3, 7, 10, 15}`; these ship at `0` (sparse) — raise
  it for the dense-reward variants.
- TMPI keeps the thesis `gamma: 0.99` and schedules its own `τ` through the exact Teleport
  Bound, which is conservative: it improves the policy at the current `τ` and only lowers `τ`
  once that stops paying off. So `τ` annealing is map-dependent — it can hold at `τ₀` on small
  maps while the policy converges and anneals on larger ones like this 30×30. (The thesis's own
  `τ` curriculum came from its *sampled* CurrMPI, which anneals more readily.)
- `n_runs: 1` for a quick launch; the thesis averaged **10** seeds — raise `n_runs` to match.
