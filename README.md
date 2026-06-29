[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue?style=flat-square&logo=github)](https://cris96spa.github.io/teleport-mdp/)

# TMDP: Teleport Markov Decision Process

![Teleport MDP demo](https://raw.githubusercontent.com/cris96spa/teleportMPD/main/docs/src/teleport_demo.gif)

## Introduction

Deep Reinforcement Learning (DRL) has revolutionized complex decision-making tasks, but still faces challenges in environments with sparse rewards, high-dimensional spaces, and long-term credit assignment issues. This project introduces the Teleport Markov Decision Processes (TMDPs) framework, which enhances the exploration capabilities of RL agents through a teleportation mechanism, contributing to more effective curriculum learning.

## The Teleport MDP Framework

A Teleport MDP extends the traditional Markov Decision Process (MDP) by adding a teleportation mechanism. It allows an agent to be relocated to any state during an episode, controlled by:

- Teleport rate (τ): Determines the frequency of teleportation
- State teleport probability distribution (ξ): Dictates the possible states for teleportation

### The Curriculum

TMDPs start with a high teleport rate for wide exploration, gradually reducing it to increase task complexity and converge towards the original problem formulation.

### Mathematical Formulation

A TMDP is defined by the tuple M=⟨S,A,P,R,γ,μ,τ,ξ⟩, where:

- S: State space
- A: Action space
- P(s′∣s,a): Transition probability model
- R(s,a): Reward function
- γ: Discount factor
- μ: Initial state distribution
- τ: Teleport rate
- ξ: Teleport probability distribution

The transition model in TMDP is defined as:

Pτ(s′∣s,a)=(1−τ)P(s′∣s,a)+τξ(s′)

## Practical Algorithms

We developed several algorithms integrating teleport-based curricula:

1. Teleport Model Policy Iteration (TMPI)
2. Static Teleport (S-T)
3. Dynamic Teleport (D-T)

## Experimental Evaluation

We conducted experiments using two RL environments:

1. Frozen Lake
2. River Swim

Results demonstrated that TMDP-based algorithms consistently outperformed their vanilla counterparts in both environments.

## Conclusion

The Teleport MDP framework offers a flexible and effective approach to curriculum design in reinforcement learning, reducing reliance on domain-specific expertise and improving learning efficiency.

## Co-Authors

This research was conducted in collaboration with:

- Prof. Marcello Restelli
- Dr. Alberto Maria Metelli
- Dr. Luca Sabbioni

## References

1. Andrychowicz, M., et al. (2017). Hindsight experience replay.
2. Florensa, C., et al. (2017). Reverse curriculum generation for reinforcement learning.
3. Kakade, S. M., & Langford, J. (2002). Approximately optimal approximate reinforcement learning.
4. Metelli, A. M., et al. (2018). Configurable Markov decision processes.
5. Schulman, J., et al. (2017). Proximal policy optimization algorithms.
6. Bengio, Y., et al. (2009). Curriculum learning.

## Usage

The template is based on [UV](https://docs.astral.sh/) as package manager and [Make](https://www.gnu.org/software/make/) as command runner. You need to have both installed in your system to use this template.

Once you have those, you can run

```bash
make dev
```

to create a virtual environment and install all the dependencies, including the development ones, and set up pre-commit hooks. If instead you want to install only production dependencies, you can run

```bash
make install
```

You can see all available targets with:

```bash
make help
```

### Formatting, Linting and Testing

You can configure Ruff by editing the `[tool.ruff]` section in `pyproject.toml`.

Format your code:

```bash
make format
```

Run linters:

```bash
make lint
```

Check formatting without modifying files:

```bash
make format-check
```

### Executing

The code is a simple hello world example, which just requires a number as input. It will output the sum of the provided number with a random number.
You can run the code with:

```bash
uv run python main.py --number 5
```

### Docker

The template includes a multi-stage Dockerfile, which produces an image with the code and the dependencies installed. You can build the image with:

```bash
docker build -t teleport-mdp .
```

### Documentation

Build and serve the documentation locally:

```bash
make doc
```

### Github Actions

The template includes two Github Actions workflows.

The first one runs tests and linters on every push on the main and dev branches. You can find the workflow file in `.github/workflows/main-list-test.yml`.

The second one is triggered on every tag push and can also be triggered manually. It builds the distribution and uploads it to PyPI. You can find the workflow file in `.github/workflows/publish.yaml`.

## Configuration

The template separates configuration into two kinds, each with its own base class in `utils/configs.py`:

- **Process settings** — `YamlBaseSettings`, layered over the environment so environment variables can override the YAML file. Best for singular, per-process settings such as the global log level.
- **Instance configs** — `YamlBaseModel`, plain data models loaded explicitly from a file. The same class can be loaded many times from different files, with no shared environment state between instances.

### Default path with per-instance override

Instance configs are loaded through `from_yaml`. A config class may set a `DEFAULT_CONFIG_PATH`, which is used whenever no path is given — so the common case takes no arguments, while any case that needs a different file simply passes one:

```python
from utils.configs import MlflowLoggerConfig

config = MlflowLoggerConfig.from_yaml()                      # default file
config = MlflowLoggerConfig.from_yaml("configs/other.yaml")  # this instance only
```

Classes without a `DEFAULT_CONFIG_PATH` require an explicit path.

## Greetings
A big thank you to [Giovanni Giacometti](https://github.com/GiovanniGiacometti) for creating this template and sharing it with the community. This template is a fork of his original work, which can be found at [giovannigiacometti/python-repository-template](https://github.com/GiovanniGiacometti/teleport-mdp).
