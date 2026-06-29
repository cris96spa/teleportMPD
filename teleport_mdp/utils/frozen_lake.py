from gymnasium.utils import seeding


def is_valid(board: list[list[str]], max_size: int) -> bool:
    """Check if there is a valid path from start to goal in the board using DFS.

    This function performs a depth-first search (DFS) to determine if there exists
    a path from the starting position (0, 0) to any goal position ('G') in the
    given board. The algorithm explores all reachable frozen ('F') and start ('S')
    tiles while avoiding holes ('H').

    Args:
        board (list[list[str]]): A 2D list representing the game board where:
            - 'S': Start position
            - 'F': Frozen tile (walkable)
            - 'H': Hole (not walkable)
            - 'G': Goal position
        max_size (int): The maximum size (dimension) of the square board.
            Must be a positive integer representing both width and height.

    Returns:
        bool: True if there exists at least one valid path from start to any goal
            position, False otherwise.

    Example:
        >>> board = [["S", "F", "F"], ["F", "H", "F"], ["F", "F", "G"]]
        >>> is_valid(board, 3)
        True

        >>> board = [["S", "H", "F"], ["H", "H", "F"], ["F", "F", "G"]]
        >>> is_valid(board, 3)
        False

    Note:
        - The function assumes the start position is always at (0, 0)
        - Movement is allowed in 4 directions (up, down, left, right)
        - The board is assumed to be a square grid of size max_size x max_size
    """
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if (r, c) not in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
    return False


def generate_random_map(size: int = 8, p: float = 0.8, seed: int | None = None) -> list[str]:
    """Generate a random valid map that has a path from start to goal.

    Args:
        size: size of each side of the grid.
        p: probability that a tile is frozen.
        seed: optional seed to ensure the generation of reproducible maps.

    Returns:
        A list of strings representing the map.
    """
    valid = False

    np_random, _ = seeding.np_random(seed)

    while not valid:
        p = min(1, p)
        board: list[list[str]] = np_random.choice(["F", "H"], (size, size), p=[p, 1 - p]).tolist()
        board[0][0] = "S"
        board[-1][-1] = "G"
        valid = is_valid(board, size)
    return ["".join(x) for x in board]
