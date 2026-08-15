"""Torch-free tests for mazes.py, focused on Maze.apply_edit -- the
adversary's grid-mutation mechanic. No episode has ever actually triggered
a grid edit in real GPU runs so far, so this path was only checked via
manual/static review, not exercised end to end; these tests exist because
that review found a real bug (see test_apply_edit_rejects_out_of_bounds_*
below) before it ever fired for real.

Run: python -m src.maze_feedback.tests.test_mazes
(or with pytest). No GPU / model required.
"""
from src.maze_feedback.mazes import Maze

_GRID = ("#####", "#S..#", "#...#", "#..E#", "#####")


def _maze():
    return Maze(grid=_GRID, start=(1, 1), goal=(3, 3))


def test_apply_edit_valid_cell_works():
    m = _maze()
    m2 = m.apply_edit(2, 1, "#")
    assert m2.grid[2] == "##..#"
    assert m2.grid != m.grid, "original Maze must stay unchanged (immutable)"


def test_apply_edit_rejects_start_and_goal():
    m = _maze()
    for pos in (m.start, m.goal):
        try:
            m.apply_edit(pos[0], pos[1], "#")
            assert False, f"expected ValueError editing {pos}"
        except ValueError:
            pass


def test_apply_edit_rejects_bad_char():
    m = _maze()
    try:
        m.apply_edit(2, 1, "X")
        assert False, "expected ValueError for a non-'.'/'#' char"
    except ValueError:
        pass


def test_apply_edit_rejects_out_of_bounds_column():
    """Regression test: Python string slicing tolerates out-of-range
    indices silently (old_row[:col] + ch + old_row[col+1:] with
    col >= len(old_row) just appends past the end), so an out-of-bounds
    COLUMN used to corrupt the grid into a non-rectangular shape instead
    of raising -- and the caller in runner.py only catches
    (ValueError, IndexError), so the corruption would have silently
    become current_maze for the rest of the episode."""
    m = _maze()
    try:
        m.apply_edit(2, 999, "#")
        assert False, "expected IndexError for an out-of-bounds column"
    except IndexError:
        pass


def test_apply_edit_rejects_out_of_bounds_row():
    m = _maze()
    try:
        m.apply_edit(999, 2, "#")
        assert False, "expected IndexError for an out-of-bounds row"
    except IndexError:
        pass


def test_apply_edit_rejects_negative_coordinates():
    m = _maze()
    try:
        m.apply_edit(-1, 2, "#")
        assert False, "expected IndexError for a negative row"
    except IndexError:
        pass
    try:
        m.apply_edit(2, -1, "#")
        assert False, "expected IndexError for a negative column"
    except IndexError:
        pass


def test_apply_edit_never_shrinks_or_grows_grid_shape():
    """Every row must stay the same length and the grid must stay the same
    number of rows after any successful edit -- the invariant the bounds
    check above exists to protect."""
    m = _maze()
    rows, cols = len(m.grid), len(m.grid[0])
    for r in range(rows):
        for c in range(cols):
            if (r, c) in (m.start, m.goal):
                continue
            m2 = m.apply_edit(r, c, "#" if m.grid[r][c] == "." else ".")
            assert len(m2.grid) == rows
            assert all(len(row) == cols for row in m2.grid), f"shape broke editing {(r, c)}"


if __name__ == "__main__":
    import sys
    import traceback

    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError:
            failures += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
