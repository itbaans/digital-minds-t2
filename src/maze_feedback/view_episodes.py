"""Local static-HTML browser for a completed batch run's episodes.
Standalone, run-locally tool -- torch-free, no GPU/model/server needed.
Reads runs/batch_<timestamp>/episodes/*.json (and mazes/, summary.json if
present) and writes a single self-contained HTML file with everything
embedded inline -- open it directly in a browser, no local server required.

    python -m src.maze_feedback.view_episodes runs/batch_1786835957
    # writes episode_viewer.html -- open it in a browser
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_batch(batch_dir: Path) -> dict:
    episodes = []
    for p in sorted((batch_dir / "episodes").glob("*.json")):
        episodes.append(json.load(open(p)))

    mazes = {}
    mazes_dir = batch_dir / "mazes"
    if mazes_dir.exists():
        for p in sorted(mazes_dir.glob("*.json")):
            m = json.load(open(p))
            mazes[m["seed"]] = m

    summary = None
    summary_path = batch_dir / "summary.json"
    if summary_path.exists():
        summary = json.load(open(summary_path))

    return {"episodes": episodes, "mazes": mazes, "summary": summary}


_PAGE_TEMPLATE = """\
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>episode viewer</title>
<style>
  :root {{ color-scheme: light dark; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0;
         background: #0b0d12; color: #e6e6e6; height: 100vh; overflow: hidden; }}
  .layout {{ display: flex; height: 100vh; }}
  .sidebar {{ width: 340px; flex-shrink: 0; border-right: 1px solid #2a2e3a; overflow-y: auto;
             padding: 12px; background: #0e1016; }}
  .sidebar h1 {{ font-size: 14px; font-weight: 600; margin: 0 0 4px; color: #9aa; }}
  .sidebar .params {{ font-size: 11px; color: #667; margin-bottom: 10px; }}
  .filters {{ display: flex; gap: 6px; margin-bottom: 10px; flex-wrap: wrap; }}
  .filters button {{ font-size: 11px; padding: 3px 8px; border-radius: 6px; border: 1px solid #2a2e3a;
                     background: #12151c; color: #9aa; cursor: pointer; }}
  .filters button.active {{ background: #1f3a2a; color: #4f9; border-color: #1f3a2a; }}
  .ep-row {{ padding: 7px 8px; margin-bottom: 4px; border-radius: 7px; cursor: pointer;
            border: 1px solid transparent; font-size: 12px; display: flex; justify-content: space-between; }}
  .ep-row:hover {{ background: #161923; }}
  .ep-row.selected {{ background: #16202e; border-color: #4f9cff; }}
  .ep-row .label {{ color: #cdd; }}
  .ep-row .badge {{ font-size: 10px; padding: 1px 6px; border-radius: 999px; font-weight: 600; }}
  .badge.solved {{ background: #1f3a2a; color: #4f9; }}
  .badge.max_turns_exceeded {{ background: #3a1f1f; color: #f66; }}
  .main {{ flex: 1; overflow-y: auto; padding: 16px 24px; }}
  .main .empty {{ color: #666; font-style: italic; margin-top: 40px; text-align: center; }}
  .ep-header {{ display: flex; align-items: baseline; gap: 10px; margin-bottom: 4px; }}
  .ep-header h2 {{ font-size: 16px; margin: 0; }}
  .ep-meta {{ font-size: 12px; color: #9aa; margin-bottom: 12px; }}
  .valence {{ display: flex; align-items: flex-end; gap: 2px; height: 44px; margin: 10px 0 16px; }}
  .vbar {{ width: 6px; background: #4f9cff; border-radius: 1px; }}
  .vbar.neg {{ background: #ff6b6b; }}
  .log {{ font-size: 13px; line-height: 1.5; }}
  .entry {{ padding: 8px 10px; margin-bottom: 8px; border-radius: 7px; white-space: pre-wrap; }}
  .entry.student {{ background: #16202e; border-left: 3px solid #4f9cff; }}
  .entry.overseer {{ background: #241621; border-left: 3px solid #d16bd1; }}
  .entry.grid_edit {{ background: #2e1616; border-left: 3px solid #ff6b6b; font-weight: 600; }}
  .entry.student_prompt {{ background: #14161b; border-left: 3px solid #444; color: #99a; font-size: 12px; }}
  .entry .tag {{ font-weight: 700; font-size: 10px; text-transform: uppercase; opacity: 0.7;
                display: block; margin-bottom: 4px; }}
</style>
</head>
<body>
<div class="layout">
  <div class="sidebar">
    <h1>episode viewer</h1>
    <div class="params">{n_episodes} episodes &middot; {params}</div>
    <div class="filters" id="filters"></div>
    <div id="ep-list"></div>
  </div>
  <div class="main" id="main"><div class="empty">select an episode on the left</div></div>
</div>
<script>
const DATA = {data_json};

function escapeHtml(s) {{
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}}

function renderValence(vt) {{
  if (!vt || vt.length === 0) return '';
  const max = Math.max(...vt.map(Math.abs), 0.01);
  let html = '<div class="valence">';
  for (const v of vt) {{
    const h = Math.max(2, Math.round(Math.abs(v) / max * 40));
    html += `<div class="vbar ${{v < 0 ? 'neg' : ''}}" style="height:${{h}}px" title="${{v.toFixed(3)}}"></div>`;
  }}
  return html + '</div>';
}}

function renderLog(log) {{
  let html = '<div class="log">';
  for (const e of log) {{
    html += `<div class="entry ${{e.role}}"><span class="tag">turn ${{e.turn}} &middot; ${{e.role}}</span>${{escapeHtml(e.text)}}</div>`;
  }}
  return html + '</div>';
}}

let currentFilter = 'all';

function showEpisode(idx) {{
  const ep = DATA.episodes[idx];
  document.querySelectorAll('.ep-row').forEach(el => el.classList.remove('selected'));
  document.getElementById('ep-row-' + idx).classList.add('selected');
  const r = ep.result;
  const main = document.getElementById('main');
  main.innerHTML = `
    <div class="ep-header">
      <h2>maze ${{String(ep.maze_index).padStart(2,'0')}} &middot; ${{ep.role}}</h2>
      <span class="badge ${{r.status}}">${{r.status}}</span>
    </div>
    <div class="ep-meta">turns=${{r.turns}} &middot; solution_length=${{ep.solution_length}} &middot;
      seed=${{ep.seed}} &middot; final position=[${{r.true_position}}]</div>
    ${{renderValence(r.valence_trajectory)}}
    ${{renderLog(r.log)}}
  `;
}}

function passesFilter(ep) {{
  if (currentFilter === 'all') return true;
  if (currentFilter === 'teacher' || currentFilter === 'adversary') return ep.role === currentFilter;
  if (currentFilter === 'solved') return ep.result.status === 'solved';
  if (currentFilter === 'capped') return ep.result.status !== 'solved';
  return true;
}}

function renderList() {{
  const list = document.getElementById('ep-list');
  list.innerHTML = DATA.episodes.map((ep, i) => {{
    if (!passesFilter(ep)) return '';
    const r = ep.result;
    return `<div class="ep-row" id="ep-row-${{i}}" onclick="showEpisode(${{i}})">
      <span class="label">maze ${{String(ep.maze_index).padStart(2,'0')}} &middot; ${{ep.role}}</span>
      <span class="badge ${{r.status}}">${{r.status === 'solved' ? 'solved t'+r.turns : 'capped'}}</span>
    </div>`;
  }}).join('');
}}

document.getElementById('filters').innerHTML =
  ['all', 'teacher', 'adversary', 'solved', 'capped'].map(f =>
    `<button id="filter-${{f}}" onclick="setFilter('${{f}}')">${{f}}</button>`).join('');

function setFilter(f) {{
  currentFilter = f;
  document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
  document.getElementById('filter-' + f).classList.add('active');
  renderList();
}}

setFilter('all');
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_dir", help="path to a runs/batch_<timestamp>/ directory")
    ap.add_argument("--out", default="episode_viewer.html")
    args = ap.parse_args()

    batch_dir = Path(args.batch_dir)
    data = _load_batch(batch_dir)
    n = len(data["episodes"])
    if n == 0:
        raise SystemExit(f"no episode JSON found under {batch_dir}/episodes/")

    summary = data["summary"]
    if summary:
        t, a = summary["teacher"], summary["adversary"]
        params = (f"teacher solve {t['solve_rate']:.0%} &middot; "
                 f"adversary solve {a['solve_rate']:.0%}")
    else:
        params = str(batch_dir)

    page = _PAGE_TEMPLATE.format(
        n_episodes=n, params=params,
        data_json=json.dumps({"episodes": data["episodes"]}),
    )
    out_path = Path(args.out)
    out_path.write_text(page)
    print(f"wrote {out_path.resolve()} ({out_path.stat().st_size / 1e6:.1f} MB) -- open it in a browser")


if __name__ == "__main__":
    main()
