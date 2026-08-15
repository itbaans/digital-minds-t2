"""Simple live viewer for running maze_feedback episode(s).

    python -m src.maze_feedback.webapp --port 8420

Serves a single page that polls /api/state every ~1.5s and shows one panel
per live-state file currently present in runs/ -- `maze_feedback_live_teacher
.json` / `..._adversary.json` from a single `experiment.py run`, or
`..._worker0.json`, `..._worker1.json`, ... from a parallel `experiment.py
batch --workers N` (each worker process owns one live-state file, updated
in place as it works through its assigned episodes). Panel count adapts to
however many files exist -- nothing here needs to know in advance whether
it's watching 2 loops or N workers. Torch-free: just reads whatever JSON
runner.py's run_episode() is writing to disk, doesn't touch the model.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from flask import Flask, jsonify, Response

app = Flask(__name__)
STATE_DIR = Path("runs")
_LIVE_RE = re.compile(r"^maze_feedback_live_(.+)\.json$")


@app.route("/api/state")
def api_state():
    result = {}
    for path in sorted(STATE_DIR.glob("maze_feedback_live_*.json")):
        m = _LIVE_RE.match(path.name)
        if not m:
            continue
        try:
            result[m.group(1)] = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            result[m.group(1)] = None  # mid-write; frontend just retries next poll
    return jsonify(result)


@app.route("/")
def index():
    return Response(_PAGE, mimetype="text/html")


_PAGE = """\
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>maze_feedback live</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0; padding: 16px;
         background: #0b0d12; color: #e6e6e6; }
  h1 { font-size: 16px; font-weight: 600; margin: 0 0 12px; color: #9aa; }
  .cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }
  .panel { border: 1px solid #2a2e3a; border-radius: 10px; padding: 12px; background: #12151c; min-width: 0; }
  .panel h2 { margin: 0 0 8px; font-size: 14px; text-transform: capitalize; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; margin-left: 6px; }
  .badge.running { background: #2a3a2a; color: #8f8; }
  .badge.solved { background: #1f3a2a; color: #4f9; }
  .badge.stuck { background: #3a2a1f; color: #fa6; }
  .badge.max_turns_exceeded { background: #3a1f1f; color: #f66; }
  .badge.none { background: #2a2a2a; color: #888; }
  .grid { display: inline-grid; gap: 1px; background: #2a2e3a; border: 1px solid #2a2e3a; margin: 8px 0; }
  .cell { width: 18px; height: 18px; display: flex; align-items: center; justify-content: center; font-size: 10px; }
  .wall { background: #23262e; }
  .open { background: #1a1d24; }
  .pos { background: #4f9cff; color: #001; font-weight: 700; }
  .goal { background: #4f9; color: #001; font-weight: 700; }
  .start { background: #ffd24f; color: #001; font-weight: 700; }
  .meta { font-size: 12px; color: #9aa; margin: 6px 0; }
  .valence { display: flex; align-items: flex-end; gap: 1px; height: 40px; margin: 8px 0; }
  .vbar { width: 4px; background: #4f9cff; }
  .vbar.neg { background: #ff6b6b; }
  .log { max-height: 420px; overflow-y: auto; font-size: 12px; line-height: 1.4; }
  .entry { padding: 6px 8px; margin-bottom: 6px; border-radius: 6px; white-space: pre-wrap; }
  .entry.student { background: #16202e; border-left: 3px solid #4f9cff; }
  .entry.overseer { background: #241621; border-left: 3px solid #d16bd1; }
  .entry.grid_edit { background: #2e1616; border-left: 3px solid #ff6b6b; font-weight: 600; }
  .entry.student_prompt { display: none; }
  .entry .tag { font-weight: 700; font-size: 10px; text-transform: uppercase; opacity: 0.7; display: block; margin-bottom: 2px; }
  .empty { color: #666; font-style: italic; padding: 20px 0; text-align: center; }
</style>
</head>
<body>
<h1>maze_feedback — live <span id="waiting-note" class="meta"></span></h1>
<div class="cols" id="cols"></div>
<script>
const panels = {};  // name -> {panelEl, badgeEl, bodyEl}, created once and reused
                     // across polls so preserveScroll's DOM references stay valid.

function getOrCreatePanel(name) {
  if (panels[name]) return panels[name];
  const panelEl = document.createElement('div');
  panelEl.className = 'panel';
  panelEl.innerHTML = `<h2>${name}<span class="badge none" id="badge-${name}">no data</span></h2>
    <div id="body-${name}" class="empty">waiting for runs/maze_feedback_live_${name}.json ...</div>`;
  document.getElementById('cols').appendChild(panelEl);
  const entry = {
    panelEl,
    badgeEl: panelEl.querySelector(`#badge-${name}`),
    bodyEl: panelEl.querySelector(`#body-${name}`),
  };
  panels[name] = entry;
  return entry;
}

function renderGrid(grid, pos, goal, start) {
  const rows = grid.length, cols = grid[0].length;
  let html = `<div class="grid" style="grid-template-columns: repeat(${cols}, 18px);">`;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const ch = grid[r][c];
      let cls = ch === '#' ? 'wall' : 'open';
      let label = '';
      if (pos && r === pos[0] && c === pos[1]) { cls = 'pos'; label = '●'; }
      else if (goal && r === goal[0] && c === goal[1]) { cls = 'goal'; label = 'E'; }
      else if (start && r === start[0] && c === start[1]) { cls = 'start'; label = 'S'; }
      html += `<div class="cell ${cls}">${label}</div>`;
    }
  }
  return html + '</div>';
}

function renderValence(vt) {
  if (!vt || vt.length === 0) return '';
  const max = Math.max(...vt.map(Math.abs), 0.01);
  let html = '<div class="valence">';
  for (const v of vt) {
    const h = Math.max(2, Math.round(Math.abs(v) / max * 38));
    html += `<div class="vbar ${v < 0 ? 'neg' : ''}" style="height:${h}px" title="${v.toFixed(3)}"></div>`;
  }
  return html + '</div>';
}

function renderLog(log) {
  if (!log || log.length === 0) return '';
  let html = '<div class="log">';
  for (const e of log) {
    if (e.role === 'student_prompt') continue;
    html += `<div class="entry ${e.role}"><span class="tag">turn ${e.turn} · ${e.role}</span>${escapeHtml(e.text)}</div>`;
  }
  return html + '</div>';
}

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function preserveScroll(body, renderFn) {
  // Re-rendering via innerHTML destroys and recreates the scrollable .log
  // div each poll, silently resetting scrollTop to 0. Save it (and whether
  // the user was following the bottom of the log) before replacing the
  // DOM, then restore/re-follow after -- so scrolling up to read an older
  // turn doesn't get yanked back on the next 1.5s poll.
  const oldLog = body.querySelector('.log');
  let scrollTop = 0, wasAtBottom = true;
  if (oldLog) {
    scrollTop = oldLog.scrollTop;
    wasAtBottom = oldLog.scrollHeight - oldLog.scrollTop - oldLog.clientHeight < 40;
  }
  renderFn();
  const newLog = body.querySelector('.log');
  if (newLog) {
    newLog.scrollTop = wasAtBottom ? newLog.scrollHeight : scrollTop;
  }
}

async function poll() {
  try {
    const res = await fetch('/api/state');
    const data = await res.json();
    const names = Object.keys(data).sort();
    document.getElementById('waiting-note').textContent =
      names.length === 0 ? '(no live-state files found yet)' : '';
    for (const name of names) {
      const st = data[name];
      const { badgeEl, bodyEl } = getOrCreatePanel(name);
      if (!st) {
        badgeEl.textContent = 'no data'; badgeEl.className = 'badge none';
        continue;
      }
      badgeEl.textContent = `${st.status} · turn ${st.turn}/${st.max_turns}`;
      badgeEl.className = 'badge ' + st.status;
      preserveScroll(bodyEl, () => {
        bodyEl.innerHTML =
          renderGrid(st.maze_grid, st.true_position, st.goal_position, st.start_position) +
          `<div class="meta">position: [${st.true_position}]  ·  ${st.valence_trajectory.length} valence reads</div>` +
          renderValence(st.valence_trajectory) +
          renderLog(st.log);
      });
    }
  } catch (e) { /* transient fetch/parse error, just retry next tick */ }
}
poll();
setInterval(poll, 1500);
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    print(f"[maze_feedback webapp] serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
