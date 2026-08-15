"""Hand-drawn 2D mazes (NOT procedurally generated -- different from
src/maze/, which is a lava-avoidance grid-world game for RL training, not a
wall-and-corridor maze). Each grid was drawn by hand and traced step by step
to find the/a valid path; the trace is kept in a comment so it's auditable.

Grid legend: '#' wall, '.' open floor, 'S' start, 'E' goal. Coordinates are
(row, col), 0-indexed, row 0 at the top. A move sequence is a string of
U/D/L/R (up/down/left/right); grading simulates it against the grid and
accepts ANY sequence that reaches E without ever crossing a wall or going
out of bounds (not required to match one specific canonical path, except
where the maze has only one possible route to begin with).
"""
from __future__ import annotations

from dataclasses import dataclass

_DIR = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


@dataclass(frozen=True)
class Maze:
    grid: tuple  # tuple of equal-length strings
    start: tuple  # (row, col)
    goal: tuple  # (row, col)

    def render(self) -> str:
        return "\n".join(self.grid)

    def make_checker(self):
        def check(extracted: str) -> bool:
            moves = [c for c in extracted.upper() if c in _DIR]
            if not moves:
                return False
            r, c = self.start
            for m in moves:
                dr, dc = _DIR[m]
                r, c = r + dr, c + dc
                if not (0 <= r < len(self.grid) and 0 <= c < len(self.grid[0])):
                    return False
                if self.grid[r][c] == "#":
                    return False
            return (r, c) == self.goal
        return check


# ── easy: 4 moves, single unbranching corridor ─────────────────────────────
# Trace: S(1,1)-R->(1,2)-R->(1,3)-D->(2,3)-D->E(3,3). No branch points at all.
EASY_MAZE = Maze(
    grid=(
        "#####",
        "#S..#",
        "###.#",
        "###E#",
        "#####",
    ),
    start=(1, 1), goal=(3, 3),
)

# ── medium: 7 moves (shortest), one branch point (both branches reach E,
# one is longer -- tests whether it finds the shorter one, not whether it
# can escape a dead end). Verified shortest-path length via manual BFS: the
# R,D,D,R,R,D,D branch reaches E in 7; the other branch (via col1) takes 9.
MEDIUM_MAZE = Maze(
    grid=(
        "######",
        "#S.#.#",
        "##.#.#",
        "#....#",
        "#.##.#",
        "#...E#",
    ),
    start=(1, 1), goal=(5, 4),
)

# ── hard: 12 moves, ONE valid route to E, with a genuine dead-end trap
# (row 6, col 3) branching off the true path right before the final
# stretch -- reaching it requires backtracking to continue.
# Trace: S(1,1)-R->(1,2)-R->(1,3)-D->(2,3)-D->(3,3)-L->(3,2)-L->(3,1)
#   -D->(4,1)-D->(5,1)-R->(5,2)-R->(5,3)-[trap: D->(6,3) is a dead end,
#   walled on all other sides -- must go back to (5,3)]-R->(5,4)-R->E(5,5).
HARD_MAZE = Maze(
    grid=(
        "#######",
        "#S..###",
        "###.###",
        "#...###",
        "#.#####",
        "#....E#",
        "###.###",
        "#######",
    ),
    start=(1, 1), goal=(5, 5),
)

# ── very_hard: 12x12, 21-move solution, THREE separate dead-end traps at
# different points along the route (early/middle/late), one of them 3 cells
# deep. Verified: the 21-move solution passes the checker; entering any
# trap and continuing blindly fails; entering a trap, backtracking out, and
# continuing on the true path still passes (confirms grading doesn't
# penalize legitimate explore-and-backtrack, only unrecovered dead ends).
#
# Path: (1,1)-R->(1,2)-R->(1,3)-R->(1,4)-D->(2,4)-D->(3,4)-R->(3,5)-R->(3,6)
#   [trap1 here: R,R,R into (3,7),(3,8),(3,9), 3-deep dead end]
#   -D->(4,6)-D->(5,6)-D->(6,6)-L->(6,5)-L->(6,4)-L->(6,3)
#   [trap2 here: U,U into (5,3),(4,3) then U again into (4,2), dead end]
#   -D->(7,3)-D->(8,3)-R->(8,4)-R->(8,5)
#   [trap3 here: D,D into (9,5),(10,5), dead end]
#   -R->(8,6)-R->(8,7)-D->(9,7)-D->E(10,7)
# Solution: RRRDDRRDDDLLLDDRRRRDD (21 moves)
VERY_HARD_MAZE = Maze(
    grid=(
        "############",
        "#S...#######",
        "####.#######",
        "####......##",
        "##..##.#####",
        "###.##.#####",
        "###....#####",
        "###.########",
        "###.....####",
        "#####.#.####",
        "#####.#E####",
        "############",
    ),
    start=(1, 1), goal=(10, 7),
)
