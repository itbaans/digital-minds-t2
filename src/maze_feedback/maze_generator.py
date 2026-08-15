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


def generate_maze(seed: int, rooms: int = 9, target_moves: int = 32) -> Maze:
    """rooms x rooms room-grid -> a (2*rooms+1) x (2*rooms+1) char Maze,
    matching the '#'/'.'/'S'/'E' convention used throughout this package.
    Deterministic for a given (seed, rooms).

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
    return Maze(grid=grid_tuple, start=start, goal=goal)
