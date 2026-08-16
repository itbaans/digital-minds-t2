"""Local viewer for procedurally-generated mazes (maze_generator.py).
Standalone, run-locally tool -- torch-free, no GPU or model needed. Useful
for eyeballing maze layouts/difficulty across many seeds during
calibration, without needing a GPU box or the full pipeline.

Two modes:

    # interactive (recommended): a local server with input controls
    # (n, rooms, target_moves, seed_base) that regenerate the gallery live
    # by calling the REAL generate_maze() function server-side -- so what
    # you see is guaranteed to match what the actual experiment produces,
    # not a reimplementation that could drift from it.
    python -m src.maze_feedback.view_mazes --serve --port 8421
    # open http://localhost:8421

    # static one-off: writes a single HTML file for a fixed parameter set
    python -m src.maze_feedback.view_mazes --n 20 --rooms 5 --target-moves 21
    # writes maze_gallery.html -- open it in a browser
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .maze_generator import generate_maze, generate_sparse_maze
from .mazes import THE_MAZE, THE_MAZE_SOLUTION


def _render_maze(m, title: str, meta: str, card_class: str = "maze-card") -> str:
    """Pure rendering: takes an already-built Maze object, not a
    (seed, rooms, target_moves) triple -- shared by the procedurally
    generated cards AND the fixed THE_MAZE fixture reference card, so both
    use exactly the same visual style for direct comparison."""
    rows, cols = len(m.grid), len(m.grid[0])
    cells = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) == m.start:
                cls, label = "start", "S"
            elif (r, c) == m.goal:
                cls, label = "goal", "E"
            elif m.grid[r][c] == "#":
                cls, label = "wall", ""
            else:
                cls, label = "open", ""
            cells.append(f'<div class="cell {cls}">{label}</div>')
    grid_html = (f'<div class="grid" style="grid-template-columns: repeat({cols}, 18px);">'
                + "".join(cells) + "</div>")
    return f'<div class="{card_class}"><h3>{title}</h3>{grid_html}<div class="meta">{meta}</div></div>'


def _render_maze_card(seed: int, rooms: int, target_moves: int, mode: str = "dense", n_traps: int = 3,
                      pad_to: int | None = None, density_range: tuple | None = None) -> str:
    if mode == "sparse":
        m = generate_sparse_maze(seed, rooms=rooms, target_moves=target_moves, n_traps=n_traps, pad_to=pad_to,
                                 density_range=density_range)
    else:
        m = generate_maze(seed, rooms=rooms, target_moves=target_moves, pad_to=pad_to)
    dist = m.bfs_distance_to_goal(m.start)
    rows, cols = len(m.grid), len(m.grid[0])
    total = rows * cols
    open_cells = sum(row.count(".") + row.count("S") + row.count("E") for row in m.grid)
    meta = (f"seed={seed} &middot; grid {rows}x{cols} &middot; "
           f"solution length={dist} (target {target_moves}) &middot; "
           f"open floor {open_cells}/{total} ({open_cells / total:.0%})")
    return _render_maze(m, f"seed {seed}", meta)


def _render_fixture_card() -> str:
    dist = THE_MAZE.bfs_distance_to_goal(THE_MAZE.start)
    total = len(THE_MAZE.grid) * len(THE_MAZE.grid[0])
    open_cells = sum(row.count(".") + row.count("S") + row.count("E") for row in THE_MAZE.grid)
    meta = (f"hand-drawn, 12x12 &middot; solution length={dist} &middot; "
           f"3 dead-end traps &middot; open floor {open_cells}/{total} ({open_cells / total:.0%}) "
           f"&middot; solution: {THE_MAZE_SOLUTION}")
    return _render_maze(THE_MAZE, "THE_MAZE (original fixture)", meta, card_class="maze-card fixture")


def _render_gallery(n: int, rooms: int, target_moves: int, seed_base: int, mode: str = "dense", n_traps: int = 3,
                    pad_to: int | None = None, density_range: tuple | None = None) -> str:
    return "\n".join(_render_maze_card(seed_base + i, rooms, target_moves, mode, n_traps, pad_to, density_range)
                     for i in range(n))


_SHARED_STYLE = """\
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0; padding: 16px;
         background: #0b0d12; color: #e6e6e6; }
  h1 { font-size: 16px; font-weight: 600; margin: 0 0 4px; color: #9aa; }
  .params { font-size: 12px; color: #667; margin: 0 0 16px; }
  .gallery { display: flex; flex-wrap: wrap; gap: 16px; }
  .maze-card { border: 1px solid #2a2e3a; border-radius: 10px; padding: 12px; background: #12151c; }
  .maze-card h3 { margin: 0 0 8px; font-size: 13px; color: #9aa; font-weight: 600; }
  .grid { display: inline-grid; gap: 1px; background: #2a2e3a; border: 1px solid #2a2e3a; margin: 4px 0; }
  .cell { width: 18px; height: 18px; display: flex; align-items: center; justify-content: center; font-size: 10px; }
  .wall { background: #23262e; }
  .open { background: #1a1d24; }
  .goal { background: #4f9; color: #001; font-weight: 700; }
  .start { background: #ffd24f; color: #001; font-weight: 700; }
  .meta { font-size: 11px; color: #667; margin-top: 6px; }
  .maze-card.fixture { border-color: #ffd24f; background: #1a1710; }
  .maze-card.fixture h3 { color: #ffd24f; }
  .fixture-section { margin-bottom: 20px; padding-bottom: 16px; border-bottom: 1px solid #2a2e3a; }
"""

def cmd_static(args):
    density_range = (args.density_min, args.density_max) if args.density_min is not None else None
    cards = _render_gallery(args.n, args.rooms, args.target_moves, args.seed_base, args.mode, args.n_traps,
                            args.pad_to, density_range)
    grid_size = args.pad_to if args.pad_to and args.pad_to > 2 * args.rooms + 1 else 2 * args.rooms + 1
    # f-string, not .format(): _SHARED_STYLE's raw CSS braces would collide
    # with .format()'s placeholder syntax (a value interpolated into an
    # f-string doesn't need its own braces escaped, unlike .format()).
    page = f"""\
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>maze gallery</title>
<style>
{_SHARED_STYLE}
</style>
</head>
<body>
<h1>maze gallery</h1>
<div class="fixture-section">
{_render_fixture_card()}
</div>
<div class="params">mode={args.mode} &middot; n={args.n} &middot; rooms={args.rooms} (grid {grid_size}x{grid_size}) &middot;
  target_moves={args.target_moves} &middot; seed_base={args.seed_base}
  {"&middot; n_traps=" + str(args.n_traps) if args.mode == "sparse" else ""}
  {"&middot; pad_to=" + str(args.pad_to) if args.pad_to else ""}
  {"&middot; density_range=[" + str(args.density_min) + ", " + str(args.density_max) + "]" if density_range else ""}</div>
<div class="gallery">
{cards}
</div>
</body>
</html>
"""
    out_path = Path(args.out)
    out_path.write_text(page)
    print(f"wrote {out_path.resolve()} -- open it in a browser")


def cmd_serve(args):
    from flask import Flask, request, Response

    app = Flask(__name__)

    @app.route("/api/gallery")
    def api_gallery():
        n = request.args.get("n", 10, type=int)
        rooms = request.args.get("rooms", 5, type=int)
        target_moves = request.args.get("target_moves", 21, type=int)
        seed_base = request.args.get("seed_base", 0, type=int)
        mode = request.args.get("mode", "dense")
        n_traps = request.args.get("n_traps", 3, type=int)
        pad_to = request.args.get("pad_to", 0, type=int)
        density_min = request.args.get("density_min", "", type=str)
        density_max = request.args.get("density_max", "", type=str)
        n = max(1, min(n, 200))  # sane bounds -- this runs synchronously per request
        rooms = max(2, min(rooms, 20))
        mode = mode if mode in ("dense", "sparse") else "dense"
        n_traps = max(0, min(n_traps, 20))
        pad_to = max(0, min(pad_to, 60)) or None
        density_range = None
        if density_min.strip() and density_max.strip():
            try:
                lo, hi = float(density_min), float(density_max)
                if 0 < lo < hi < 1:
                    density_range = (lo, hi)
            except ValueError:
                pass
        return Response(_render_gallery(n, rooms, target_moves, seed_base, mode, n_traps, pad_to, density_range),
                        mimetype="text/html")

    @app.route("/")
    def index():
        # .replace(), not .format()/f-string: _SERVE_PAGE's own JS contains
        # ${...} template-literal syntax (e.g. ${n}) which an f-string
        # would misparse as Python interpolation -- a plain marker
        # substitution sidesteps that entirely.
        page = _SERVE_PAGE.replace("__FIXTURE_CARD__", _render_fixture_card())
        return Response(page, mimetype="text/html")

    print(f"[maze viewer] serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


_SERVE_PAGE = """\
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>maze viewer</title>
<style>
""" + _SHARED_STYLE + """\
  .controls { display: flex; gap: 12px; align-items: end; flex-wrap: wrap; margin-bottom: 16px;
             padding: 12px; border: 1px solid #2a2e3a; border-radius: 10px; background: #12151c; }
  .field { display: flex; flex-direction: column; gap: 4px; }
  .field label { font-size: 11px; color: #9aa; }
  .field input, .field select { width: 90px; padding: 5px 7px; border-radius: 6px; border: 1px solid #2a2e3a;
                background: #0e1016; color: #e6e6e6; font-size: 13px; }
  .field button { padding: 6px 14px; border-radius: 6px; border: 1px solid #2a2e3a;
                  background: #1f3a2a; color: #4f9; font-weight: 600; cursor: pointer; font-size: 13px; }
  .status { font-size: 11px; color: #667; }
</style>
</head>
<body>
<h1>maze viewer</h1>
<div class="fixture-section">
__FIXTURE_CARD__
</div>
<div class="params">Adjust parameters and click Generate -- calls the real generator server-side,
  same code the actual experiment uses.</div>
<div class="controls">
  <div class="field"><label>mode</label>
    <select id="mode" onchange="toggleTrapsField()">
      <option value="dense">dense (generate_maze -- full spanning tree)</option>
      <option value="sparse" selected>sparse (generate_sparse_maze -- path + traps only, like THE_MAZE)</option>
    </select>
  </div>
  <div class="field"><label>n (count)</label><input type="number" id="n" value="10" min="1" max="200"></div>
  <div class="field"><label>rooms</label><input type="number" id="rooms" value="5" min="2" max="20"></div>
  <div class="field"><label>target_moves</label><input type="number" id="target_moves" value="21" min="1"></div>
  <div class="field"><label>seed_base</label><input type="number" id="seed_base" value="0" min="0"></div>
  <div class="field" id="traps_field"><label>n_traps (sparse only)</label><input type="number" id="n_traps" value="3" min="0" max="20"></div>
  <div class="field"><label>pad_to (0 = off)</label><input type="number" id="pad_to" value="12" min="0" max="60"></div>
  <div class="field" id="density_field"><label>density_min / max (sparse only, blank = off)</label>
    <div style="display:flex; gap:4px;">
      <input type="number" id="density_min" value="0.19" min="0" max="1" step="0.01" style="width:42px;">
      <input type="number" id="density_max" value="0.22" min="0" max="1" step="0.01" style="width:42px;">
    </div>
  </div>
  <div class="field"><button onclick="generate()">Generate</button></div>
  <div class="status" id="status"></div>
</div>
<div class="gallery" id="gallery"></div>
<script>
function toggleTrapsField() {
  const mode = document.getElementById('mode').value;
  document.getElementById('traps_field').style.display = mode === 'sparse' ? 'flex' : 'none';
  document.getElementById('density_field').style.display = mode === 'sparse' ? 'flex' : 'none';
}
async function generate() {
  const mode = document.getElementById('mode').value;
  const n = document.getElementById('n').value;
  const rooms = document.getElementById('rooms').value;
  const target_moves = document.getElementById('target_moves').value;
  const seed_base = document.getElementById('seed_base').value;
  const n_traps = document.getElementById('n_traps').value;
  const pad_to = document.getElementById('pad_to').value;
  const density_min = mode === 'sparse' ? document.getElementById('density_min').value : '';
  const density_max = mode === 'sparse' ? document.getElementById('density_max').value : '';
  const status = document.getElementById('status');
  status.textContent = 'generating...';
  const res = await fetch(`/api/gallery?mode=${mode}&n=${n}&rooms=${rooms}&target_moves=${target_moves}&seed_base=${seed_base}&n_traps=${n_traps}&pad_to=${pad_to}&density_min=${density_min}&density_max=${density_max}`);
  document.getElementById('gallery').innerHTML = await res.text();
  status.textContent = `mode=${mode}, ${n} mazes, rooms=${rooms}, target_moves=${target_moves}, seed_base=${seed_base}` +
    (pad_to > 0 ? `, pad_to=${pad_to}` : '') +
    (density_min && density_max ? `, density=[${density_min}, ${density_max}]` : '');
}
document.querySelectorAll('.controls input').forEach(el =>
  el.addEventListener('keydown', e => { if (e.key === 'Enter') generate(); }));
toggleTrapsField();
generate();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true", help="run an interactive local server instead of writing a static file")
    ap.add_argument("--port", type=int, default=8421)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--n", type=int, default=10, help="number of mazes to render (static mode only)")
    ap.add_argument("--mode", choices=["dense", "sparse"], default="sparse",
                    help="dense = generate_maze (full spanning tree); "
                         "sparse = generate_sparse_maze (path + a few traps, like THE_MAZE)")
    ap.add_argument("--rooms", type=int, default=5)
    ap.add_argument("--target-moves", type=int, default=21)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--n-traps", type=int, default=3, help="sparse mode only")
    ap.add_argument("--pad-to", type=int, default=None,
                    help="pad grid with wall out to an exact size (e.g. 12), matching THE_MAZE's dimensions")
    ap.add_argument("--density-min", type=float, default=None,
                    help="sparse mode only -- min open-floor fraction (e.g. 0.19); requires --density-max too")
    ap.add_argument("--density-max", type=float, default=None,
                    help="sparse mode only -- max open-floor fraction (e.g. 0.22); requires --density-min too")
    ap.add_argument("--out", default="maze_gallery.html", help="static mode only")
    args = ap.parse_args()

    if args.serve:
        cmd_serve(args)
    else:
        cmd_static(args)


if __name__ == "__main__":
    main()
