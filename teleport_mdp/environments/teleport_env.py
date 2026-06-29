from abc import ABC, abstractmethod
from typing import Any

from gymnasium import Env


class TeleportEnv(ABC, Env):
    """Abstract class for teleport environments. A teleport environment is a gymnasium environment that has the following methods:
    - `is_terminal(self, state: Any) -> bool`: Returns whether the given state is terminal.
    - `set_state(self, state: Any) -> None`: Sets the environment state.
    """

    @abstractmethod
    def is_terminal(self, state: Any) -> bool:
        """Returns whether the given state is terminal."""
        raise NotImplementedError

    @abstractmethod
    def teleport(self, teleport_distribution: Any) -> Any:
        """Teleport the agent to a random state."""
        raise NotImplementedError
