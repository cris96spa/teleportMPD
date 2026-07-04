from typing import Literal

# region Actions
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

#: The action type accepted by the FrozenLake environment.
Action = Literal[0, 1, 2, 3]
ACTION_NAMES: dict[int, str] = {LEFT: "Left", DOWN: "Down", RIGHT: "Right", UP: "Up"}
# endregion

# region Tiles
START_TILE = b"S"
FROZEN_TILE = b"F"
HOLE_TILE = b"H"
GOAL_TILE = b"G"
TERMINAL_TILES = b"GH"
# endregion

# region Maps
#: Built-in FrozenLake layouts keyed by map name.
MAPS: dict[str, list[str]] = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}

DEFAULT_MAP_NAME = "4x4"
# endregion

#: Tolerance used when checking that a probability distribution sums to one.
STOCHASTICITY_THRESHOLD = 1e-7
