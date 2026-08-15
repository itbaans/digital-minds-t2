"""Torch-free tests for maze_generator.py: connectivity, determinism, and
S/E placement sanity for the batch run's procedurally-generated mazes.

Run: python -m src.maze_feedback.tests.test_maze_generator
(or with pytest). No GPU / model required.
"""
from src.maze_feedback.maze_generator import generate_maze


def test_generated_maze_is_solvable():
    for seed in range(20):
        m = generate_maze(seed, rooms=6, target_moves=32)
        assert m.bfs_distance_to_goal(m.start) is not None, f"seed={seed} unreachable"


def test_generated_maze_deterministic_for_seed():
    a = generate_maze(7, rooms=6, target_moves=32)
    b = generate_maze(7, rooms=6, target_moves=32)
    assert a.grid == b.grid
    assert a.start == b.start
    assert a.goal == b.goal


def test_different_seeds_give_different_layouts():
    grids = {generate_maze(seed, rooms=6, target_moves=32).grid for seed in range(10)}
    assert len(grids) == 10, "expected 10 distinct maze layouts across 10 seeds"


def test_solution_path_matches_bfs_distance():
    for seed in range(10):
        m = generate_maze(seed, rooms=6, target_moves=32)
        path = m.bfs_path_to_goal(m.start)
        assert path is not None
        assert len(path) == m.bfs_distance_to_goal(m.start)
        sim = m.simulate(m.start, path)
        assert sim["reached_goal"], f"seed={seed} recorded solution path doesn't reach the goal"


def test_start_and_goal_are_marked_correctly_in_grid():
    for seed in range(10):
        m = generate_maze(seed, rooms=6, target_moves=32)
        sr, sc = m.start
        gr, gc = m.goal
        assert m.grid[sr][sc] == "S"
        assert m.grid[gr][gc] == "E"
        assert m.start != m.goal


def test_start_is_pinned_to_top_left_corner():
    """Regression test: an earlier version picked start as the farthest
    room FROM the corner (the tree-diameter-endpoint trick), which let S
    land anywhere in the grid depending on the tree's shape -- observed
    once landing near the geometric center on a 13x13 maze. Start must
    always be the corner room (char-grid (1, 1)), matching where the
    original hand-drawn 12x12 pilot fixture put S, so goal is the only
    thing that varies in position across seeds."""
    for rooms in (4, 5, 6, 8):
        for seed in range(5):
            m = generate_maze(seed, rooms=rooms)
            assert m.start == (1, 1), f"rooms={rooms} seed={seed} start={m.start}, expected (1, 1)"


def test_goal_lands_near_bottom_right_corner_without_sacrificing_length():
    """Goal should land in the bottom-right corner region (matching where
    the original hand-drawn pilot fixture put E) AND still hit the
    solution-length cap -- the two shouldn't trade off against each other.
    An earlier version searched for "near the overall max distance found,
    then closest to corner", which could pick a room further from the
    corner than necessary; this version searches the corner region FIRST
    and maximizes distance within it, which empirically still hits the
    cap. Goal position still varies across seeds (a version pinning it to
    one exact fixed room was tried and reverted, see DESIGN.md 12) --
    just always within the corner region, not literally identical."""
    for rooms in (5, 6, 8):
        corner = (2 * rooms - 1, 2 * rooms - 1)
        target_moves = 6 * rooms
        goals = set()
        lengths = []
        corner_dists = []
        for seed in range(10):
            m = generate_maze(seed, rooms=rooms, target_moves=target_moves)
            corner_dists.append(abs(m.goal[0] - corner[0]) + abs(m.goal[1] - corner[1]))
            lengths.append(m.bfs_distance_to_goal(m.start))
            goals.add(m.goal)
        # The tight corner region can genuinely be unreachable within the
        # cap for a given seed's tree shape, in which case the fallback
        # picks the closest-available room instead (still corner-biased,
        # just not guaranteed within the tight radius) -- so check the
        # typical case (mean) across seeds rather than every seed
        # individually for both distance-to-corner and solution length.
        mean_corner_dist = sum(corner_dists) / len(corner_dists)
        assert mean_corner_dist <= 6, (
            f"rooms={rooms}: mean goal-to-corner distance {mean_corner_dist:.1f} too large "
            f"(corner_dists={corner_dists})")
        mean_len = sum(lengths) / len(lengths)
        assert mean_len >= target_moves - 6, (
            f"rooms={rooms}: mean solution length {mean_len:.1f} well short of the "
            f"~{target_moves} target -- corner-region search may be sacrificing length (lengths={lengths})")
        assert len(goals) > 1, f"rooms={rooms}: goal should vary across seeds within the corner region"


def test_grid_dimensions_match_rooms_param():
    for rooms in (4, 6, 8):
        m = generate_maze(0, rooms=rooms)
        expected = 2 * rooms + 1
        assert len(m.grid) == expected
        assert all(len(row) == expected for row in m.grid)


def test_target_moves_caps_solution_length():
    for seed in range(10):
        m = generate_maze(seed, rooms=8, target_moves=20)
        assert m.bfs_distance_to_goal(m.start) <= 20


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
