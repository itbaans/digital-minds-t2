"""Torch-free tests for maze_generator.py: connectivity, determinism, and
S/E placement sanity for the batch run's procedurally-generated mazes.

Run: python -m src.maze_feedback.tests.test_maze_generator
(or with pytest). No GPU / model required.
"""
from src.maze_feedback.maze_generator import generate_maze, generate_sparse_maze, open_fraction


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


def test_pad_to_gives_exact_12x12_dense_maze():
    """rooms=5 naturally gives 11x11 (2*5+1); pad_to=12 should pad to an
    exact 12x12 grid (one wall row/col added) without disturbing S/E or
    solvability -- matching THE_MAZE's dimensions exactly."""
    for seed in range(10):
        m = generate_maze(seed, rooms=5, target_moves=21, pad_to=12)
        assert len(m.grid) == 12
        assert all(len(row) == 12 for row in m.grid)
        assert m.start == (1, 1)
        assert m.bfs_distance_to_goal(m.start) is not None
        # the padded row/col should be solid wall, not touching S/E/path
        assert all(c == "#" for c in m.grid[-1])
        assert all(row[-1] == "#" for row in m.grid)


def test_pad_to_smaller_than_natural_size_is_a_noop():
    a = generate_maze(0, rooms=6, target_moves=32)
    b = generate_maze(0, rooms=6, target_moves=32, pad_to=5)
    assert a.grid == b.grid


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


# ── generate_sparse_maze: THE_MAZE-style sparse layout (path + traps only,
# not a full spanning tree) ──────────────────────────────────────────────

def test_sparse_maze_is_solvable_and_deterministic():
    for seed in range(15):
        a = generate_sparse_maze(seed, rooms=5, target_moves=21)
        assert a.bfs_distance_to_goal(a.start) is not None, f"seed={seed} unreachable"
        b = generate_sparse_maze(seed, rooms=5, target_moves=21)
        assert a.grid == b.grid and a.start == b.start and a.goal == b.goal


def test_sparse_maze_solution_path_matches_bfs_distance():
    """No trap can create a shortcut: the recorded solution path length
    must exactly equal the real BFS distance, every time."""
    for seed in range(15):
        m = generate_sparse_maze(seed, rooms=5, target_moves=21)
        path = m.bfs_path_to_goal(m.start)
        assert path is not None
        assert len(path) == m.bfs_distance_to_goal(m.start)
        assert m.simulate(m.start, path)["reached_goal"]


def test_sparse_maze_start_pinned_to_corner():
    for seed in range(10):
        m = generate_sparse_maze(seed, rooms=5, target_moves=21)
        assert m.start == (1, 1)


def test_sparse_maze_is_meaningfully_sparser_than_full_spanning_tree():
    """The whole point: generate_sparse_maze's open-floor density should
    be much closer to THE_MAZE's ~21% than generate_maze's ~40%+ (a full
    spanning tree necessarily opens every room in the grid)."""
    def open_fraction(m):
        total = sum(len(row) for row in m.grid)
        open_cells = sum(row.count(".") + row.count("S") + row.count("E") for row in m.grid)
        return open_cells / total

    dense_fractions = [open_fraction(generate_maze(seed, rooms=5, target_moves=21)) for seed in range(10)]
    sparse_fractions = [open_fraction(generate_sparse_maze(seed, rooms=5, target_moves=21)) for seed in range(10)]
    mean_dense = sum(dense_fractions) / len(dense_fractions)
    mean_sparse = sum(sparse_fractions) / len(sparse_fractions)
    assert mean_sparse < mean_dense - 0.05, (
        f"sparse mean {mean_sparse:.1%} not meaningfully below dense mean {mean_dense:.1%}")
    # THE_MAZE itself is ~21% -- sparse mazes should land in a similar
    # ballpark (generous band, since per-seed variance is real), not just
    # "somewhat less dense than the full spanning tree".
    assert mean_sparse < 0.35, f"sparse mean {mean_sparse:.1%} still not close to THE_MAZE's ~21%"


def test_sparse_maze_trap_count_is_respected_as_a_ceiling():
    """n_traps=0 should produce a maze with ONLY the main path open --
    no branches at all."""
    for seed in range(10):
        m = generate_sparse_maze(seed, rooms=5, target_moves=21, n_traps=0)
        path = m.bfs_path_to_goal(m.start)
        open_cells = sum(row.count(".") + row.count("S") + row.count("E") for row in m.grid)
        assert open_cells == len(path) + 1, (
            f"seed={seed}: n_traps=0 should open exactly path-length+1 cells "
            f"(the path's rooms), got {open_cells} open cells for a {len(path)}-move path")


def test_sparse_maze_pad_to_gives_exact_12x12():
    """Same padding contract as generate_maze, exercised in sparse mode --
    this is the concrete case the user asked for: 12x12 sparse mazes,
    matching THE_MAZE's dimensions exactly."""
    for seed in range(10):
        m = generate_sparse_maze(seed, rooms=5, target_moves=21, pad_to=12)
        assert len(m.grid) == 12
        assert all(len(row) == 12 for row in m.grid)
        assert m.start == (1, 1)
        path = m.bfs_path_to_goal(m.start)
        assert path is not None
        assert m.simulate(m.start, path)["reached_goal"]
        assert all(c == "#" for c in m.grid[-1])
        assert all(row[-1] == "#" for row in m.grid)


def test_density_range_lands_every_seed_in_band():
    """The narrow (0.19, 0.22) band is only ~4 grid cells wide at
    rooms=5/pad_to=12 -- a single seed's tree sometimes can't reach it via
    trap growth alone (see generate_sparse_maze's docstring), so this also
    exercises the seed-fallback path, not just the in-tree search."""
    lo, hi = 0.19, 0.22
    for seed in range(30):
        m = generate_sparse_maze(seed, rooms=5, target_moves=21, pad_to=12, density_range=(lo, hi))
        d = open_fraction(m)
        assert lo <= d <= hi, f"seed={seed}: density {d:.3f} outside [{lo}, {hi}]"
        assert m.start == (1, 1)
        path = m.bfs_path_to_goal(m.start)
        assert path is not None
        assert m.simulate(m.start, path)["reached_goal"]


def test_density_range_is_deterministic_for_seed():
    a = generate_sparse_maze(7, rooms=5, target_moves=21, pad_to=12, density_range=(0.19, 0.22))
    b = generate_sparse_maze(7, rooms=5, target_moves=21, pad_to=12, density_range=(0.19, 0.22))
    assert a.grid == b.grid and a.start == b.start and a.goal == b.goal


def test_density_range_none_is_unaffected():
    """Passing density_range=None (the default) must reproduce the exact
    same output as before this parameter existed -- no behavior change for
    existing callers that don't opt in."""
    a = generate_sparse_maze(3, rooms=5, target_moves=21, n_traps=3, max_trap_depth=3)
    b = generate_sparse_maze(3, rooms=5, target_moves=21, n_traps=3, max_trap_depth=3, density_range=None)
    assert a.grid == b.grid and a.start == b.start and a.goal == b.goal


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
