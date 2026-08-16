"""Procedural maze generator for the batch experiment.

Unlike capability_probe's/mazes.py's hand-drawn fixtures, the batch run
needs many DIFFERENT verified-solvable mazes (one per run) -- not
practical to hand-author reliably at this scale (see mazes.py's dev notes
on how error-prone that got even at 12x12). This uses the standard
randomized-DFS "recursive backtracker" algorithm: it carves a spanning
tree over a grid of rooms, which by construction is fully connected with
exactly one path between any two rooms, and naturally produces dead-end
branches as a side effect (no manual trap design needed). Start is pinned
to the top-left corner room and goal is preferentially the farthest room
within the bottom-right corner region, both capped at a target solution
length -- matching where the original hand-drawn pilot fixture put S/E,
so the solution path is genuinely long without letting S/E land anywhere
in the grid depending on the tree's shape (see `generate_maze`'s
docstring for the exact selection logic and why it changed twice during
development).

Torch-free: pure Python, no model code here.
"""
from __future__ import annotations

import random
from collections import deque

from .mazes import Maze

_ROOM_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _pad_grid(grid: tuple, size: int) -> tuple:
    """Pad a square grid (each row already the same length, < size) up to
    size x size by adding wall rows at the bottom and wall columns on the
    right. Added because the room-grid construction always produces an
    ODD (2*rooms+1) dimension (room centers at odd coordinates, walls/
    connections at even ones -- a standard technique, not something the
    `rooms` parameter can work around), so there's no `rooms` value that
    lands on an EVEN size like THE_MAZE's 12x12 exactly -- closest is
    rooms=5 (11x11) or rooms=6 (13x13). Padding with wall is inert: it
    can't affect reachability or solution length, since start/goal and
    every carved cell stay at the same coordinates, just with a bit more
    unreachable wall around two edges. No-op if size <= current."""
    current = len(grid)
    if size <= current:
        return grid
    pad_cols = size - len(grid[0])
    padded_rows = tuple(row + "#" * pad_cols for row in grid)
    return padded_rows + ("#" * size,) * (size - current)


def _carve(rng: random.Random, rooms: int) -> set:
    """Randomized DFS over a rooms x rooms grid. Returns the set of open
    passages as frozenset({(r1,c1), (r2,c2)}) pairs -- a spanning tree."""
    visited = {(0, 0)}
    stack = [(0, 0)]
    passages = set()
    while stack:
        r, c = stack[-1]
        neighbors = [(r + dr, c + dc) for dr, dc in _ROOM_DIRS
                    if 0 <= r + dr < rooms and 0 <= c + dc < rooms
                    and (r + dr, c + dc) not in visited]
        if not neighbors:
            stack.pop()
            continue
        nxt = rng.choice(neighbors)
        visited.add(nxt)
        passages.add(frozenset({(r, c), nxt}))
        stack.append(nxt)
    return passages


def _room_bfs(passages: set, rooms: int, start: tuple) -> dict:
    """BFS over rooms (not chars) using `passages` for connectivity.
    Returns {(r,c): distance_from_start}."""
    dist = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        r, c = cur
        for dr, dc in _ROOM_DIRS:
            nxt = (r + dr, c + dc)
            if (0 <= nxt[0] < rooms and 0 <= nxt[1] < rooms
                    and nxt not in dist and frozenset({cur, nxt}) in passages):
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return dist


def _pick_start_goal(passages: set, rooms: int, target_moves: int) -> tuple:
    """Shared by generate_maze and generate_sparse_maze: start `a` pinned
    to the corner room, goal `b` preferentially from the opposite corner
    region within the target-length cap (see generate_maze's docstring
    for the full rationale/history). Returns (a, b, da) where da is the
    room-BFS distance map from a, reused by callers for path
    reconstruction."""
    a = (0, 0)
    da = _room_bfs(passages, rooms, a)
    target_room_dist = target_moves // 2
    corner = (rooms - 1, rooms - 1)

    def _corner_dist(room):
        return abs(room[0] - corner[0]) + abs(room[1] - corner[1])

    corner_radius = 2
    region = [(room, dist) for room, dist in da.items()
             if _corner_dist(room) <= corner_radius and dist <= target_room_dist]
    if region:
        b = max(region, key=lambda x: x[1])[0]
    else:
        capped = [room for room, dist in da.items() if dist <= target_room_dist]
        b = min(capped, key=_corner_dist) if capped else min(da, key=da.get)
    return a, b, da


def generate_maze(seed: int, rooms: int = 9, target_moves: int = 32, pad_to: int | None = None) -> Maze:
    """rooms x rooms room-grid -> a (2*rooms+1) x (2*rooms+1) char Maze,
    matching the '#'/'.'/'S'/'E' convention used throughout this package.
    Deterministic for a given (seed, rooms).

    pad_to: if given and larger than the natural (2*rooms+1) size, pads
    the grid with wall out to an exact pad_to x pad_to size (see
    _pad_grid) -- e.g. rooms=5 (naturally 11x11) with pad_to=12 gives an
    exact 12x12 grid, matching THE_MAZE's dimensions, since the room-grid
    construction can only ever produce odd sizes on its own.

    Start `a` is pinned to the corner room (0, 0) -- the room `_carve`'s
    DFS always begins from -- matching where the original hand-drawn
    12x12 pilot fixture put S (char-grid (1, 1), the top-left corner).
    An earlier version picked `a` as the farthest room FROM (0, 0) instead
    (the standard tree-diameter-endpoint trick, to avoid a trivially-close
    start/goal pair), which technically produced valid, verified-solvable
    mazes but let S land anywhere in the grid depending on the tree's
    shape -- observed once landing near the grid's geometric center on a
    13x13 maze. Pinning S to a fixed, visually-intuitive corner removes
    that as a confound without needing the diameter trick: the
    target-length cap below already guarantees a genuinely long solution
    path regardless of which room S starts from. Goal `b` is the farthest
    room from `a` that's still within target_moves char-grid moves
    (== target_moves // 2 room-hops) -- the absolute tree diameter (argmax
    distance) tends to be wildly long relative to grid size (e.g. 60-70
    moves on a 13x13 grid), too long to reliably solve within a turn
    budget, so this caps difficulty.

    Among candidates within the cap, goal is preferentially chosen from
    the bottom-right corner region (matching where the original
    hand-drawn pilot fixture put E) rather than from anywhere the length
    cap happens to allow -- opposite corners are the geometrically
    farthest-apart points in a grid, so this shouldn't trade away
    difficulty (an earlier version that instead capped to "near the
    overall max distance found, then closest to the corner" could
    accidentally trade distance away for corner-proximity; this version
    searches the corner region FIRST and maximizes distance within it, so
    it still hits the cap whenever the corner region allows it -- verified
    empirically to land on/near the cap in practice, not short of it).

    The tight corner region can genuinely have NO candidate within the
    cap -- on a large `rooms` count with a tighter `target_moves`, the
    tree's actual path to the true corner can be far longer than the cap
    even though the corner is geometrically "far" (confirmed on an 8x8
    room grid: the corner itself was 42 room-hops away against a cap of
    24). The first version of this fallback dropped corner-proximity
    entirely and picked the farthest-within-cap room from ANYWHERE, which
    reintroduced the exact "S/E anywhere in the grid" problem this
    function exists to avoid -- observed landing at the grid's literal
    geometric center on that same 8x8 case. Fixed by keeping
    corner-proximity as the primary criterion even in the fallback: among
    all rooms within the cap, pick whichever is closest to the corner
    (accepting a shorter-than-target path if that's what the corner
    region can offer) rather than maximizing distance with no positional
    preference at all."""
    rng = random.Random(seed)
    passages = _carve(rng, rooms)
    a, b, _da = _pick_start_goal(passages, rooms, target_moves)

    size = 2 * rooms + 1
    grid = [["#"] * size for _ in range(size)]
    for r in range(rooms):
        for c in range(rooms):
            grid[2 * r + 1][2 * c + 1] = "."
    for pair in passages:
        (r1, c1), (r2, c2) = tuple(pair)
        grid[r1 + r2 + 1][c1 + c2 + 1] = "."

    sr, sc = a
    gr, gc = b
    start = (2 * sr + 1, 2 * sc + 1)
    goal = (2 * gr + 1, 2 * gc + 1)
    grid[start[0]][start[1]] = "S"
    grid[goal[0]][goal[1]] = "E"

    grid_tuple = tuple("".join(row) for row in grid)
    if pad_to is not None:
        grid_tuple = _pad_grid(grid_tuple, pad_to)
    return Maze(grid=grid_tuple, start=start, goal=goal)


def _room_path(passages: set, rooms: int, a: tuple, b: tuple) -> list:
    """BFS shortest path from a to b through `passages`, as a list of
    rooms a, ..., b. `passages` is a spanning tree, so this is also the
    UNIQUE path between them, not just *a* shortest one."""
    parent = {a: None}
    q = deque([a])
    while q and b not in parent:
        cur = q.popleft()
        for dr, dc in _ROOM_DIRS:
            nxt = (cur[0] + dr, cur[1] + dc)
            if (0 <= nxt[0] < rooms and 0 <= nxt[1] < rooms
                    and nxt not in parent and frozenset({cur, nxt}) in passages):
                parent[nxt] = cur
                q.append(nxt)
    path = [b]
    while path[-1] != a:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def open_fraction(m: Maze) -> float:
    """Fraction of the grid that's open floor (., S, or E) rather than wall.
    Public so callers (view_mazes.py, analysis scripts) don't need to
    reimplement the same cell-counting logic."""
    total = sum(len(row) for row in m.grid)
    open_cells = sum(row.count(".") + row.count("S") + row.count("E") for row in m.grid)
    return open_cells / total


def _density_gap(d: float, lo: float, hi: float) -> float:
    if lo <= d <= hi:
        return 0.0
    return min(abs(d - lo), abs(d - hi))


def _build_sparse_maze(seed: int, rooms: int, target_moves: int,
                       n_traps: int, max_trap_depth: int,
                       pad_to: int | None) -> Maze:
    """The actual sparse-maze builder (carve tree -> find S->E path -> keep
    path + n_traps branches). Factored out of generate_sparse_maze so the
    density_range search below can call it repeatedly with different
    (n_traps, max_trap_depth) without duplicating the construction logic.
    See generate_sparse_maze's docstring for the full rationale."""
    rng = random.Random(seed)
    passages = _carve(rng, rooms)
    a, b, _da = _pick_start_goal(passages, rooms, target_moves)
    main_path = _room_path(passages, rooms, a, b)

    kept_rooms = set(main_path)
    kept_edges = {frozenset({main_path[i], main_path[i + 1]}) for i in range(len(main_path) - 1)}

    # Candidate trap starts: passages hanging off a room ON the main path,
    # that don't lead back onto the path (a genuine branch, not the path
    # itself). Shuffled so which traps get picked (when there are more
    # candidates than n_traps) varies by seed, not just by path order.
    branch_starts = []
    for room in main_path:
        for dr, dc in _ROOM_DIRS:
            nbr = (room[0] + dr, room[1] + dc)
            edge = frozenset({room, nbr})
            if edge in passages and nbr not in kept_rooms:
                branch_starts.append((room, nbr))
    rng.shuffle(branch_starts)

    traps_added = 0
    for room, nbr in branch_starts:
        if traps_added >= n_traps:
            break
        if nbr in kept_rooms:
            continue  # already pulled in by an earlier trap's growth
        kept_rooms.add(nbr)
        kept_edges.add(frozenset({room, nbr}))
        traps_added += 1
        # Grow this trap deeper along whatever the tree already carved,
        # 0 to max_trap_depth further rooms (0 = a single-cell nub).
        cur = nbr
        for _ in range(rng.randint(0, max_trap_depth)):
            extended = False
            for dr, dc in _ROOM_DIRS:
                nxt = (cur[0] + dr, cur[1] + dc)
                edge = frozenset({cur, nxt})
                if edge in passages and nxt not in kept_rooms:
                    kept_rooms.add(nxt)
                    kept_edges.add(edge)
                    cur = nxt
                    extended = True
                    break
            if not extended:
                break  # this branch was a dead end already, nothing further to grow
    return _grid_from_sparse(rooms, kept_rooms, kept_edges, a, b, pad_to)


def _grid_from_sparse(rooms, kept_rooms, kept_edges, a, b, pad_to):
    size = 2 * rooms + 1
    grid = [["#"] * size for _ in range(size)]
    for (r, c) in kept_rooms:
        grid[2 * r + 1][2 * c + 1] = "."
    for edge in kept_edges:
        (r1, c1), (r2, c2) = tuple(edge)
        grid[r1 + r2 + 1][c1 + c2 + 1] = "."

    sr, sc = a
    gr, gc = b
    start = (2 * sr + 1, 2 * sc + 1)
    goal = (2 * gr + 1, 2 * gc + 1)
    grid[start[0]][start[1]] = "S"
    grid[goal[0]][goal[1]] = "E"

    grid_tuple = tuple("".join(row) for row in grid)
    if pad_to is not None:
        grid_tuple = _pad_grid(grid_tuple, pad_to)
    return Maze(grid=grid_tuple, start=start, goal=goal)


def generate_sparse_maze(seed: int, rooms: int = 9, target_moves: int = 21,
                         n_traps: int = 3, max_trap_depth: int = 3,
                         pad_to: int | None = None,
                         density_range: tuple[float, float] | None = None) -> Maze:
    """Like generate_maze, but produces a sparse, THE_MAZE-style layout:
    ONLY the S->E corridor plus a small number of deliberate dead-end
    branches ("traps"), with everything else left as solid wall --
    rather than a full spanning tree covering the entire room grid.

    pad_to: if given and larger than the natural (2*rooms+1) size, pads
    the grid with wall out to an exact pad_to x pad_to size (see
    _pad_grid) -- e.g. rooms=5 (naturally 11x11) with pad_to=12 gives an
    exact 12x12 grid, matching THE_MAZE's dimensions exactly.

    density_range: if given as (lo, hi) open-floor fractions (e.g.
    (0.19, 0.22)), first tries every (n_traps, max_trap_depth) combination
    on `seed`'s own spanning tree, in increasing order of how much extra
    floor they open, and returns the first maze whose open_fraction()
    lands inside [lo, hi]. At small `rooms` / narrow bands, a given seed's
    tree sometimes just doesn't have branch material at the right
    granularity to land in a narrow band at any (n_traps, depth) --
    empirically about a quarter of seeds at rooms=5/pad_to=12 for a
    (0.19, 0.22) band. When that happens, falls back to trying OTHER
    trees, deterministically derived from `seed` (so the same input seed
    always produces the same output maze, but the tree/S->E path can
    differ from what you'd get with density_range=None -- this is a
    genuine tradeoff, not a bug: strict density compliance costs "same
    seed always gives the same layout otherwise"). Returns the first
    in-band maze found across up to 20 derived trees; if truly none of
    them land in range either (only seen with a band that's unreachable
    for this rooms/target_moves/pad_to combination in general, not a
    per-seed fluke), returns whichever candidate came closest instead of
    raising -- check open_fraction(result) yourself if you need to detect
    that case.

    Why this exists: generate_maze's output is a full spanning tree, which
    necessarily opens every room in the grid (by construction) plus about
    half the connections between them -- empirically ~40-42% open floor.
    The original hand-drawn 12x12 pilot fixture (THE_MAZE) is only ~21%
    open: a human designing a maze by hand doesn't fill the whole grid
    with corridors, they draw ONE deliberate route plus a FEW deliberate
    dead ends (THE_MAZE has exactly 3, one four cells deep). A full
    spanning tree can't produce that character no matter how its start/
    goal are chosen -- the sparseness is a property of what fraction of
    the grid gets carved at all, not of path length or S/E placement.

    Built by carving a full spanning tree (same _carve() as generate_maze,
    for genuine per-seed randomness and guaranteed connectivity), finding
    the actual unique S->E path through it, then keeping ONLY that path
    plus `n_traps` branch-offs from it (existing dead-end branches in the
    tree, encountered while walking the path, each grown 0..max_trap_depth
    further rooms along whatever branch the tree already carved there) --
    everything else in the room grid is walled back off rather than kept
    open. Traps are, by construction, genuine dead ends off the unique
    path: they can never create a shortcut or an alternate route, since
    they're literally leaves hanging off a tree with no other connections
    kept. bfs_distance_to_goal after generation always equals the main
    path's length exactly -- verified in
    tests/test_maze_generator.py."""
    m = _build_sparse_maze(seed, rooms, target_moves, n_traps, max_trap_depth, pad_to)
    if density_range is None:
        return m

    lo, hi = density_range
    best, best_gap = _search_density(seed, rooms, target_moves, n_traps, max_trap_depth, pad_to, lo, hi, m)
    if best_gap == 0.0:
        return best

    # This seed's own tree doesn't have the right granularity of branch
    # material to land in the band at any (n_traps, depth) -- fall back to
    # OTHER trees, deterministically derived from `seed` so the same input
    # always gives the same output. Empirically rescues most misses within
    # 1-2 tries (see docstring); 20 is a generous cap, not a tuned minimum.
    for attempt in range(20):
        alt_seed = seed * 1_000_003 + attempt
        alt_m = _build_sparse_maze(alt_seed, rooms, target_moves, n_traps, max_trap_depth, pad_to)
        alt_best, alt_gap = _search_density(alt_seed, rooms, target_moves, n_traps, max_trap_depth, pad_to, lo, hi, alt_m)
        if alt_gap == 0.0:
            return alt_best
        if alt_gap < best_gap:
            best, best_gap = alt_best, alt_gap
    return best


def _search_density(seed, rooms, target_moves, n_traps, max_trap_depth, pad_to, lo, hi, m):
    """Search a deterministic grid of (depth, n_traps) combinations on one
    seed's tree -- depth on the outer loop because a seed whose tree
    happens to offer few branch points needs deeper growth (not more
    branches) to reach density; more traps alone caps out once
    branch_starts is exhausted. Small, cheap, torch-free grid -- fine to
    brute-force per maze. Returns (best_maze, best_gap), gap==0.0 meaning
    an in-band maze was found."""
    best, best_gap = m, _density_gap(open_fraction(m), lo, hi)
    if best_gap == 0.0:
        return best, best_gap
    max_n_traps = 8 * rooms
    max_depth = 4 * rooms
    for depth in range(1, max_depth + 1):
        for nt in range(0, max_n_traps + 1):
            if nt == n_traps and depth == max_trap_depth:
                continue  # already tried above
            cand = _build_sparse_maze(seed, rooms, target_moves, nt, depth, pad_to)
            gap = _density_gap(open_fraction(cand), lo, hi)
            if gap == 0.0:
                return cand, 0.0
            if gap < best_gap:
                best, best_gap = cand, gap
    return best, best_gap
