# teleport_mdp

Clean, production-grade re-implementation of the MSc thesis *"Curriculum Reinforcement
Learning through Teleportation: The Teleport MDP"*. It replaces the legacy research code
in `../TMDP` (notebooks, untyped) with a typed, tested, config-driven package.

A Teleport MDP adds a teleportation mechanism to a standard MDP: with probability `τ`
(teleport rate) the agent is relocated to a state drawn from `ξ`, giving
`P_τ(s'|s,a) = (1-τ)P(s'|s,a) + τξ(s')`. A curriculum anneals `τ` from a high initial
value to 0, so training starts with wide exploration and converges to the original
problem. Implemented algorithms: TMPI (exact, tabular model-based policy iteration),
Static-Teleport and Dynamic-Teleport curricula over PPO, and tabular Q-learning.

## Layout

- `teleport_mdp/` — the package.
  - `models.py` — every Pydantic config (`ExperimentConfig` is the YAML entry point).
  - `enums.py`, `constants.py` — algorithm/curriculum tags, map data, tolerances.
  - `environments/` — the (teleport) FrozenLake env, gym registration, and the factory
    that builds an env + `ξ` from an `EnvConfig`.
  - `wrappers/tmdp.py` — `TMDP`, the Gym wrapper implementing the teleport coin flip.
  - `curriculum/scheduler.py` — `TeleportScheduler` and its Static/Dynamic subclasses.
  - `agents/` — `TeleportPPO` and `TeleportRolloutBuffer` (GAE/return truncation on
    teleport), subclassing Stable-Baselines3 rather than reimplementing PPO.
  - `callbacks/` — SB3 callbacks that stream metrics to MLflow during training.
  - `tabular/` — numpy-only, SB3-independent track: `model_functions.py`, `q_learning.py`
    (`QLearner`), `tmpi.py` (exact TMPI), `bound.py` (Teleport Bound), `trainer.py`
    (`TabularTrainer`, dispatches on `algorithm.kind`).
  - `commands/` — CLI entry points (`run_experiment` for SB3 PPO, `run_tabular` for the
    tabular track), wired up in `__main__.py`.
  - `registries/` — string/enum → factory lookups (e.g. curriculum kind → scheduler).
- `utils/` — infra shared across the repo: `configs.py` (`YamlBaseModel`/`YamlBaseSettings`),
  `experiment_logger.py` (MLflow wrapper), not teleport-MDP-specific.
- `configs/` — YAML experiment definitions, one file per variant.
- `tests/` — one `test_<module>.py` per source module; math-bearing modules are checked
  against hand-computed or closed-form values, not just smoke-tested.
- `tasks/` — lean checklist task files tracking the port from `../TMDP`; `ROADMAP.md` is
  the index, `tasks/done/` holds completed ones.

## Style

This is the part to get right — code is read far more often than it's written, and a
reader should never have to run the code to know what it does.

**Naming.** Names carry the meaning; a long, explicit name beats a short, vague one plus
a comment explaining it. `disc_visit_distribution` over `dv`, `max_steps_per_episode` over
`max_steps`, `_require_map_source` over `_validate`. A name should tell a reader what a
function does and, for non-obvious returns, what it returns — without needing the body.

**Docstrings.** Google style (`ruff` pydocstyle `convention = "google"`, enforced by
`pydoclint`/flake8-DOC). One-line summary for anything trivial; `Args`/`Returns`/`Raises`
sections only when they add information the signature doesn't already give (skip
`Returns` on an obvious getter, skip documenting a self-explanatory `config: FooConfig`
arg). Class docstrings explain the *contract* — invariants, what a subclass must
implement, how instances relate to each other — not a restatement of the class name.
Inline code uses single backticks (`` `x` ``), never RST double backticks (``x``);
double-backtick only where it's actually necessary (e.g. quoting a literal that itself
contains a backtick).

**No file/module-level docstrings.** Never add a top-of-file docstring. A file is
identified by its path and its contents; a preamble restating "this module does X" is
dead weight that drifts from the code. This is stricter than the `D10` ruff ignore (which
merely doesn't *require* docstrings) — module docstrings are actively disallowed here.


**Comments.** Default to none. Add one only when the code cannot explain itself: a
non-obvious invariant, a subtle math step, a deliberate deviation from the "natural"
implementation. Never write boilerplate ("# imports", "# init", "# return result") or
comments that restate the line below them. Never encode chat history, task numbers, or
"why we changed this from X" — that belongs in the commit message, not the source; see
[[feedback-lean-docstrings]] in memory.

**Math-bearing code** (teleport kernels, GAE/return truncation, τ schedulers, TMPI) must
cite the thesis result it implements (e.g. "thesis Theorem 5.2") in its docstring, and
ship a test against a hand-computed or closed-form value — not just a shape/smoke test.

## Conventions

- Config: Pydantic `BaseModel` loaded from YAML via `utils.configs.YamlBaseModel`
  (`from_yaml`, optional `DEFAULT_CONFIG_PATH`); `NamedTuple` for lightweight value
  bundles (e.g. `QLearningResult`). No ad-hoc dicts for structured data.
  `model_config = ConfigDict(extra="forbid")` on every config model.
  Algorithm-specific configs are a `kind`-discriminated union (`AlgorithmConfig`) so an
  algorithm and its hyperparameters are one inseparable source of truth.
- No notebooks, ever — scripts and typed functions only.
  Extend Stable-Baselines3 rather than re-implement RL from scratch; new behavior is a
  subclass/wrapper/callback (`TeleportPPO`, `TeleportRolloutBuffer`, MLflow callbacks),
  never a fork of SB3 internals.
- Experiments stream to MLflow live during training via callbacks, not a final dump.

## Commands

- `make dev` — create the venv, install all dependency groups, install pre-commit hooks.
- `make format` / `make format-check` — `ruff format`.
- `make lint` — `ruff check`; `make lint-doc` — `flake8`/pydoclint docstring checks.
- `make test` — `pytest` with coverage and doctest-modules.
- `uv run python -m teleport_mdp run_experiment --config <yaml>` — SB3 PPO track.
- `uv run python -m teleport_mdp run_tabular --config <yaml>` — tabular track.
- `make mlflow-up` / `make mlflow-down` — local MLflow tracking server (UI on :5002).

## Tasks

Work in progress is tracked in `tasks/` as numbered, lean checklist files (goal +
checklist + acceptance criteria), indexed by `tasks/ROADMAP.md`; finished ones move to
`tasks/done/`. Keep new task files just as lean — no narrative, no restated context.
