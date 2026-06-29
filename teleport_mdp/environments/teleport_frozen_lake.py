from contextlib import closing
from io import StringIO
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces, utils
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled
from overrides import override

from teleport_mdp.constants import (
    ACTION_NAMES,
    DOWN,
    LEFT,
    MAPS,
    RIGHT,
    TERMINAL_TILES,
    UP,
)
from teleport_mdp.utils.frozen_lake import generate_random_map

if TYPE_CHECKING:
    from numpy.typing import NDArray

from teleport_mdp.environments.teleport_env import TeleportEnv


class Transition(NamedTuple):
    """A single environment transition."""

    probability: float
    next_state: int
    reward: float
    terminated: bool


# region Frozen Lake
class TeleportFrozenLakeEnv(TeleportEnv):
    """Frozen lake environment: cross a frozen lake from start to goal.

    Crossing the lake means walking from start to goal without falling into any
    holes. The player may not always move in the intended direction due to the
    slippery nature of the frozen lake. With the teleportation distribution, the
    player may teleport to a random state with a given probability.

    ## Description
    The game starts with the player at location [0,0] of the frozen lake grid world with the
    goal located at far extent of the world e.g. [3,3] for the 4x4 environment.

    Holes in the ice are distributed in set locations when using a pre-determined map
    or in random locations when a random map is generated.

    The player makes moves until they reach the goal or fall in a hole.

    The lake is slippery (unless disabled) so the player may move perpendicular
    to the intended direction sometimes (see <a href="#is_slippy">`is_slippery`</a>).

    Randomly generated worlds will always have a path to the goal.

    Elf and stool from [https://franuka.itch.io/rpg-snow-tileset](https://franuka.itch.io/rpg-snow-tileset).
    All other assets by Mel Tillery [http://www.cyaneus.com/](http://www.cyaneus.com/).

    ## Action Space
    The action shape is `(1,)` in the range `{0, 3}` indicating
    which direction to move the player.

    - 0: Move left
    - 1: Move down
    - 2: Move right
    - 3: Move up

    ## Observation Space
    The observation is a value representing the player's current position as
    current_row * ncols + current_col (where both the row and col start at 0).

    For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15.
    The number of possible observations is dependent on the size of the map.

    The observation is returned as an `int()`.

    ## Starting State
    The episode starts with the player in state `[0]` (location [0, 0]).

    ## Rewards

    Reward schedule:
    - Reach goal: +1
    - Reach hole: 0
    - Reach frozen: 0

    ## Episode End
    The episode ends if the following happens:

    - Termination:
        1. The player moves into a hole.
        2. The player reaches the goal at `max(nrow) * max(ncol) - 1`
           (location `[max(nrow)-1, max(ncol)-1]`).

    - Truncation (when using the time_limit wrapper):
        1. The length of the episode is 100 for the 4x4 environment, 200 for
           the FrozenLake8x8-v1 environment.

    ## Information

    `step()` and `reset()` return a dict with the following keys:
    - p - transition probability for the state.

    See <a href="#is_slippy">`is_slippery`</a> for transition probability information.


    ## Arguments

    ```python
    import gymnasium as gym

    gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True)
    ```

    `desc=None`: Used to specify maps non-preloaded maps.

    Specify a custom map.
    ```
        desc=["SFFF", "FHFH", "FFFH", "HFFG"].
    ```
    The tile letters denote:
    - "S" for Start tile
    - "G" for Goal tile
    - "F" for frozen tile
    - "H" for a tile with a hole

    A random generated map can be specified by calling the function `generate_random_map`.
    ```
    from gymnasium.envs.toy_text.frozen_lake import generate_random_map

    gym.make("FrozenLake-v1", desc=generate_random_map(size=8))
    ```

    `map_name="4x4"`: ID to use any of the preloaded maps.
    ```
        "4x4":[
            "SFFF",
            "FHFH",
            "FFFH",
            "HFFG"
            ]

        "8x8": [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG",
        ]
    ```

    If `desc=None` then `map_name` will be used. If both `desc` and `map_name` are
    `None` a random 8x8 map with 80% of locations frozen will be generated.

    <a id="is_slippy"></a>`is_slippery=True`: If true the player will move in the
    intended direction with probability of 1/3 else will move in either
    perpendicular direction with equal probability of 1/3 in both directions.

    For example, if action is left and is_slippery is True, then:
    - P(move left)=1/3
    - P(move up)=1/3
    - P(move down)=1/3


    ## Version History
    * v1: Bug fixes to rewards
    * v0: Initial version release

    """

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        desc: list[str] | None = None,
        map_name: str = "4x4",
        is_slippery: bool = True,
    ):
        # check if the map is valid
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        # convert the map to a numpy array
        self.desc: NDArray = np.asarray(desc, dtype="c")

        # get the number of rows and columns
        self.nrow, self.ncol = self.desc.shape

        # define the reward range
        self.reward_range = (0, 1)

        # the number of actions is fixed
        n_actions = 4

        # the number of states is the number of rows times the number of columns
        n_states = self.nrow * self.ncol

        # the initial state distribution is fixed and assume
        # the player always starts at the start location 0
        self.initial_state_distrib = np.array(self.desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        # build the probability transition matrix
        self.P = self._build_transition_matrix(n_states, n_actions, is_slippery)

        # define the observation and action space
        self.observation_space = spaces.Discrete(n_states)
        self.action_space = spaces.Discrete(n_actions)

        # define the render mode
        self.render_mode = render_mode

        # visualization related variables
        self.window_size = (min(64 * self.ncol, 512), min(64 * self.nrow, 512))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None

    def _to_s(self, row: int, col: int) -> int:
        """Convert a row and column to a state number.

        Args:
            row: the row number.
            col: the column number.

        Returns:
            The state number corresponding to `(row, col)`.
        """
        return row * self.ncol + col

    def _inc(self, row: int, col: int, action: int) -> tuple[int, int]:
        """Move within the grid according to an action, clamping at the borders.

        Args:
            row: the row number.
            col: the column number.
            action: the action to take.

        Returns:
            The new `(row, col)` position after applying the action.
        """
        if action == LEFT:
            col = max(col - 1, 0)
        elif action == DOWN:
            row = min(row + 1, self.nrow - 1)
        elif action == RIGHT:
            col = min(col + 1, self.ncol - 1)
        elif action == UP:
            row = max(row - 1, 0)
        return (row, col)

    def _update_probability_matrix(
        self, row: int, col: int, action: int
    ) -> tuple[int, float, bool]:
        """Compute the transition produced by taking an action in a cell.

        Args:
            row: the row number.
            col: the column number.
            action: the action to take.

        Returns:
            The new state, the obtained reward, and whether the new state is
            terminal.
        """
        new_row, new_col = self._inc(row, col, action)
        new_state = self._to_s(new_row, new_col)
        new_letter = self.desc[new_row, new_col]
        terminated = bytes(new_letter) in TERMINAL_TILES
        reward = float(new_letter == b"G")
        return new_state, reward, terminated

    def _build_transition_matrix(
        self, n_states: int, n_actions: int, is_slippery: bool
    ) -> dict[int, dict[int, list[Transition]]]:
        """Build the transition probability matrix of the environment.

        Args:
            n_states: the number of states in the environment.
            n_actions: the number of available actions.
            is_slippery: whether moves are stochastic (slippery lake).

        Returns:
            A mapping from each state to a mapping from each action to the list
            of possible transitions.
        """
        transition_matrix: dict[int, dict[int, list[Transition]]] = {
            s: {a: [] for a in range(n_actions)} for s in range(n_states)
        }
        for row in range(self.nrow):
            for col in range(self.ncol):
                s = self._to_s(row, col)
                for a in range(n_actions):
                    transitions = transition_matrix[s][a]
                    letter = self.desc[row, col]
                    if letter in TERMINAL_TILES:
                        transitions.append(Transition(1.0, s, 0.0, True))
                    elif is_slippery:
                        for b in [(a - 1) % 4, a, (a + 1) % 4]:
                            transitions.append(
                                Transition(
                                    1.0 / 3.0,
                                    *self._update_probability_matrix(row, col, b),
                                )
                            )
                    else:
                        transitions.append(
                            Transition(1.0, *self._update_probability_matrix(row, col, a))
                        )
        return transition_matrix

    # endregion
    # region Gym API
    @override
    def step(self, action: int) -> tuple[int, SupportsFloat, bool, bool, dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: the action to take.

        Returns:
            A tuple `(s_prime, r, terminated, truncated, info)` with the next
            state, reward, termination flag, truncation flag, and an info
            dictionary holding the transition probability.
        """
        transitions = self.P[int(self.s)][action]
        i = categorical_sample([t.probability for t in transitions], self.np_random)
        probability, s, r, terminated = transitions[i]
        self.s: int = s
        self.lastaction = action

        if self.render_mode == "human":
            self.render()
        return int(s), r, terminated, False, {"prob": probability}

    @override
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: the seed to use for random number generation.
            options: additional options to pass to the environment.

        Returns:
            A tuple `(state, info)` with the initial state and an info
            dictionary holding the transition probability.
        """
        super().reset(seed=seed, options=options)
        self.s = int(categorical_sample(self.initial_state_distrib, self.np_random))
        self.lastaction = None

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1}

    @override
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode == "ansi":
            return self._render_text()
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

    # endregion
    # region TeleportEnv
    @override(check_signature=False)
    def is_terminal(self, state: int) -> bool:
        """Return whether the given state is terminal.

        In the context of FrozenLake, a terminal state is one that is either a
        hole or the goal.

        Args:
            state: the state to check.

        Returns:
            `True` if the state is terminal, `False` otherwise.

        Raises:
            ValueError: if the state is outside the valid state range.
        """
        if state < 0 or state >= self.nrow * self.ncol:
            raise ValueError(
                "State out of bound: ",
                state,
                ". The state must be in [0, ",
                self.nrow * self.ncol,
                " ]",
            )
        row, col = state // self.ncol, state % self.ncol
        return bytes(self.desc[row, col]) in TERMINAL_TILES

    @override(check_signature=False)
    def teleport(self, teleport_distribution: np.ndarray[Any, np.dtype[Any]]) -> int:
        """Teleport the agent to a random state.

        In the context of FrozenLake, the agent is teleported to a random state
        based on the teleportation distribution. The state must be a valid
        non-terminal state.

        Args:
            teleport_distribution: the teleport probability distribution.

        Returns:
            The new state.
        """
        while True:
            s_prime = int(categorical_sample(teleport_distribution, self.np_random))
            if not self.is_terminal(s_prime):
                return s_prime

    # endregion
    # region GUI
    def _render_gui(self, mode: str) -> np.ndarray | None:
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[toy-text]"`'
            ) from e

        self._setup_render(pygame, mode)
        assert self.window_surface is not None, (
            "Something went wrong with pygame. This should never happen."
        )

        desc = self.desc.tolist()
        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"
        self._draw_board(pygame, desc)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            assert self.clock is not None
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)),
                axes=(1, 0, 2),
            )
        return None

    def _setup_render(self, pygame: ModuleType, mode: str) -> None:
        """Initialise the pygame window, clock, and image assets on first use."""
        if self.window_surface is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Frozen Lake")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.elf_images is None:
            self.hole_img = self._load_image(pygame, "img/hole.png")
            self.cracked_hole_img = self._load_image(pygame, "img/cracked_hole.png")
            self.ice_img = self._load_image(pygame, "img/ice.png")
            self.goal_img = self._load_image(pygame, "img/goal.png")
            self.start_img = self._load_image(pygame, "img/stool.png")
            self.elf_images = [
                self._load_image(pygame, file_name)
                for file_name in (
                    "img/elf_left.png",
                    "img/elf_down.png",
                    "img/elf_right.png",
                    "img/elf_up.png",
                )
            ]

    def _load_image(self, pygame: ModuleType, file_name: str) -> Any:
        """Load an image from disk and scale it to the cell size.

        When optional assets are not packaged with the project, fall back to a
        plain colored tile so rendering still works in tests.
        """
        image_path = Path(__file__).parent / file_name
        if image_path.exists():
            return pygame.transform.scale(pygame.image.load(str(image_path)), self.cell_size)

        fallback_surface = pygame.Surface(self.cell_size)
        fallback_surface.fill(self._fallback_color(file_name))
        return fallback_surface

    def _fallback_color(self, file_name: str) -> tuple[int, int, int]:
        """Return a deterministic fallback color for a missing sprite asset."""
        color_map = {
            "img/hole.png": (44, 62, 80),
            "img/cracked_hole.png": (30, 45, 60),
            "img/ice.png": (184, 223, 255),
            "img/goal.png": (255, 215, 0),
            "img/stool.png": (139, 90, 43),
            "img/elf_left.png": (76, 175, 80),
            "img/elf_down.png": (46, 125, 50),
            "img/elf_right.png": (102, 187, 106),
            "img/elf_up.png": (56, 142, 60),
        }
        return color_map.get(file_name, (200, 200, 200))

    def _draw_board(self, pygame: ModuleType, desc: list[list[bytes]]) -> None:
        """Draw the lake grid and the elf onto the pygame surface."""
        assert self.window_surface is not None
        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.ice_img, pos)
                if desc[y][x] == b"H":
                    self.window_surface.blit(self.hole_img, pos)
                elif desc[y][x] == b"G":
                    self.window_surface.blit(self.goal_img, pos)
                elif desc[y][x] == b"S":
                    self.window_surface.blit(self.start_img, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        # paint the elf
        bot_row, bot_col = self.s // self.ncol, self.s % self.ncol
        cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        last_action = self.lastaction if self.lastaction is not None else 1
        assert self.elf_images is not None
        elf_img = self.elf_images[last_action]

        if desc[bot_row][bot_col] == b"H":
            self.window_surface.blit(self.cracked_hole_img, cell_rect)
        else:
            self.window_surface.blit(elf_img, cell_rect)

    def _render_text(self) -> str:
        desc = self.desc.tolist()
        outfile = StringIO()

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(f"  ({ACTION_NAMES[self.lastaction]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        with closing(outfile):
            return outfile.getvalue()

    @override
    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()

    # endregion


# Elf and stool from https://franuka.itch.io/rpg-snow-tileset
# All other assets by Mel Tillery http://www.cyaneus.com/
