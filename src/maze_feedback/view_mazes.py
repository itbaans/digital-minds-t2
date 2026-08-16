"""Local static-HTML viewer for procedurally-generated mazes
(maze_generator.py). Standalone, run-locally tool -- torch-free, no GPU or
model needed. Useful for eyeballing maze layouts/difficulty across many
seeds during calibration, without needing a GPU box or the full pipeline.

    python -m src.maze_feedback.view_mazes --n 20 --rooms 5 --target-moves 21
    # writes maze_gallery.html in the current directory -- open it in a browser

Generates a static page, not a live server: re-run the command (optionally
with --out) to regenerate after changing parameters.
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

from .maze_generator import generate_maze


def _render_maze_card(seed: int, rooms: int, target_moves: int) -> str:
    m = generate_maze(seed, rooms=rooms, target_moves=target_moves)
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

    dist = m.bfs_distance_to_goal(m.start)
    meta = (f"seed={seed} &middot; grid {rows}x{cols} &middot; "
           f"solution length={dist} (target {target_moves})")
    return f'<div class="maze-card"><h3>seed {seed}</h3>{grid_html}<div class="meta">{meta}</div></div>'


_PAGE_TEMPLATE = """\
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>maze gallery</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0; padding: 16px;
         background: #0b0d12; color: #e6e6e6; }}
  h1 {{ font-size: 16px; font-weight: 600; margin: 0 0 4px; color: #9aa; }}
  .params {{ font-size: 12px; color: #667; margin: 0 0 16px; }}
  .gallery {{ display: flex; flex-wrap: wrap; gap: 16px; }}
  .maze-card {{ border: 1px solid #2a2e3a; border-radius: 10px; padding: 12px; background: #12151c; }}
  .maze-card h3 {{ margin: 0 0 8px; font-size: 13px; color: #9aa; font-weight: 600; }}
  .grid {{ display: inline-grid; gap: 1px; background: #2a2e3a; border: 1px solid #2a2e3a; margin: 4px 0; }}
  .cell {{ width: 18px; height: 18px; display: flex; align-items: center; justify-content: center; font-size: 10px; }}
  .wall {{ background: #23262e; }}
  .open {{ background: #1a1d24; }}
  .goal {{ background: #4f9; color: #001; font-weight: 700; }}
  .start {{ background: #ffd24f; color: #001; font-weight: 700; }}
  .meta {{ font-size: 11px; color: #667; margin-top: 6px; }}
</style>
</head>
<body>
<h1>maze gallery</h1>
<div class="params">n={n} &middot; rooms={rooms} (grid {grid_size}x{grid_size}) &middot;
  target_moves={target_moves} &middot; seed_base={seed_base}</div>
<div class="gallery">
{cards}
</div>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="number of mazes to render")
    ap.add_argument("--rooms", type=int, default=5)
    ap.add_argument("--target-moves", type=int, default=21)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--out", default="maze_gallery.html")
    args = ap.parse_args()

    cards = [_render_maze_card(args.seed_base + i, args.rooms, args.target_moves)
            for i in range(args.n)]
    page = _PAGE_TEMPLATE.format(
        cards="\n".join(cards), n=args.n, rooms=args.rooms,
        grid_size=2 * args.rooms + 1, target_moves=args.target_moves, seed_base=args.seed_base,
    )
    out_path = Path(args.out)
    out_path.write_text(page)
    print(f"wrote {out_path.resolve()} -- open it in a browser")


if __name__ == "__main__":
    main()
