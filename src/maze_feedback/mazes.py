"""The 12x12 maze fixture for this experiment (reusing capability_probe's
VERY_HARD_MAZE design/verification approach), plus a BFS distance-to-goal
helper used only for progress *measurement* (thrashing detection), not for
constructing or solving the puzzle on the model's behalf.

Torch-free: pure Python, no model code here.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

_DIR = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


@dataclass(frozen=True)
class Maze:
    grid: tuple  # tuple of equal-length strings
    start: tuple  # (row, col)
    goal: tuple  # (row, col)

    def render(self) -> str:
        return "\n".join(self.grid)

    def in_bounds(self, pos: tuple) -> bool:
        r, c = pos
        return 0 <= r < len(self.grid) and 0 <= c < len(self.grid[0])

    def is_open(self, pos: tuple) -> bool:
        r, c = pos
        return self.in_bounds(pos) and self.grid[r][c] != "#"

    def step(self, pos: tuple, move: str) -> tuple:
        """Apply one U/D/L/R move to `pos`. Returns (new_pos, valid).
        If invalid (wall or out of bounds), new_pos == pos (no-op)."""
        dr, dc = _DIR[move.upper()]
        new_pos = (pos[0] + dr, pos[1] + dc)
        if self.is_open(new_pos):
            return new_pos, True
        return pos, False

    def simulate(self, start_pos: tuple, moves: str) -> dict:
        """Apply a sequence of U/D/L/R moves starting from start_pos.
        Stops at the first invalid move (doesn't skip past it). Returns a
        dict: {"end_pos", "valid_prefix_len", "first_invalid_move" (or
        None), "reached_goal"}."""
        pos = start_pos
        parsed = [c for c in moves.upper() if c in _DIR]
        for i, m in enumerate(parsed):
            new_pos, ok = self.step(pos, m)
            if not ok:
                return {
                    "end_pos": pos, "valid_prefix_len": i,
                    "first_invalid_move": m, "reached_goal": pos == self.goal,
                }
            pos = new_pos
            if pos == self.goal:
                return {
                    "end_pos": pos, "valid_prefix_len": i + 1,
                    "first_invalid_move": None, "reached_goal": True,
                }
        return {
            "end_pos": pos, "valid_prefix_len": len(parsed),
            "first_invalid_move": None, "reached_goal": pos == self.goal,
        }

    def bfs_distance_to_goal(self, pos: tuple) -> int | None:
        """Shortest remaining distance from pos to goal, or None if
        unreachable. Used only for progress MEASUREMENT (thrashing
        detection), never to hand the student a solution."""
        if pos == self.goal:
            return 0
        seen = {pos}
        q = deque([(pos, 0)])
        while q:
            cur, d = q.popleft()
            for m in _DIR:
                nxt, ok = self.step(cur, m)
                if ok and nxt not in seen:
                    if nxt == self.goal:
                        return d + 1
                    seen.add(nxt)
                    q.append((nxt, d + 1))
        return None

    def bfs_path_to_goal(self, pos: tuple) -> str | None:
        """Shortest U/D/L/R move string from pos to goal, or None if
        unreachable. Used by maze_generator.py to record the ground-truth
        solution for a procedurally-generated maze (the generated grid is a
        spanning tree, so this is also the UNIQUE path) -- never exposed to
        the student, only to the overseer's "known correct solution" prompt
        field, same as THE_MAZE_SOLUTION below."""
        if pos == self.goal:
            return ""
        seen = {pos}
        q = deque([(pos, "")])
        while q:
            cur, path = q.popleft()
            for m in _DIR:
                nxt, ok = self.step(cur, m)
                if ok and nxt not in seen:
                    if nxt == self.goal:
                        return path + m
                    seen.add(nxt)
                    q.append((nxt, path + m))
        return None

    def apply_edit(self, row: int, col: int, new_char: str) -> "Maze":
        """Returns a NEW Maze with one cell changed (this Maze is
        immutable). Used by the adversary role to actually alter the
        problem mid-episode -- not just lie about it in words. Refuses to
        move the S/E markers (their positions define the episode's
        start/goal throughout), but freely allows turning path into wall
        or vice versa anywhere else, including sealing off the goal
        entirely -- that's the intended "impossible state" mechanic.

        Bounds are checked explicitly (not just left to fall out of
        `self.grid[row]`) because an out-of-range COLUMN doesn't raise on
        its own: Python string slicing tolerates out-of-range indices
        silently, so `old_row[:col] + new_char + old_row[col+1:]` with
        col >= len(old_row) would just append past the end of the row,
        corrupting the grid into a non-rectangular shape instead of
        failing loudly -- and the caller's `except (ValueError,
        IndexError)` wouldn't catch that corruption, so it would silently
        become `current_maze` for the rest of the episode. Caught by a
        real bounds check here instead of ever reaching that."""
        if (row, col) in (self.start, self.goal):
            raise ValueError(f"cannot edit the S/E marker cell at {(row, col)}")
        if not self.in_bounds((row, col)):
            raise IndexError(f"edit position {(row, col)} is out of bounds for a "
                             f"{len(self.grid)}x{len(self.grid[0])} grid")
        if new_char not in (".", "#"):
            raise ValueError(f"new_char must be '.' or '#', got {new_char!r}")
        old_row = self.grid[row]
        new_row = old_row[:col] + new_char + old_row[col + 1:]
        new_grid = self.grid[:row] + (new_row,) + self.grid[row + 1:]
        return Maze(grid=new_grid, start=self.start, goal=self.goal)

    def make_checker(self):
        """Same style as capability_probe/mazes.py: accepts any valid full
        path from start to goal (used only for the one-shot validation
        pass in §5, not the turn loop itself)."""
        def check(extracted: str) -> bool:
            result = self.simulate(self.start, extracted)
            return result["reached_goal"]
        return check


# 12x12, one valid route of 21 moves (RRRDDRRDDDLLLDDRRRRDD), with three
# separate dead-end traps along the way (one 3 cells deep) -- identical
# fixture to capability_probe's VERY_HARD_MAZE, which confirmed-failed
# Qwen3-4B one-shot (wrong branch at trap 3, then misread a wall as the
# goal). Re-declared here (not imported) so this package doesn't depend on
# capability_probe's internals.
THE_MAZE = Maze(
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
THE_MAZE_SOLUTION = "RRRDDRRDDDLLLDDRRRRDD"
