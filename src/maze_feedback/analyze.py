"""Post-hoc aggregation over a completed batch run's episodes/ directory.

Torch-free -- reads the JSON results already written by experiment.py's
`batch` command; no model or GPU needed, safe to re-run any time against
a `runs/batch_*/` directory. Produces summary.json (for further
programmatic analysis later) and summary.md (human-readable) in the
batch's output directory, plus PNG plots if matplotlib is installed
(skipped with a warning, not an error, if it isn't -- plotting is a
nice-to-have on top of the saved raw data, not a hard requirement).

Can also be run standalone against an existing batch dir:
    python -m src.maze_feedback.analyze runs/batch_1234567890
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, stdev


def _load_episodes(out_dir: Path) -> list:
    episodes_dir = Path(out_dir) / "episodes"
    return [json.load(open(p)) for p in sorted(episodes_dir.glob("maze_*_*.json"))]


def _condition_stats(records: list, role: str) -> dict:
    rows = [r for r in records if r["role"] == role]
    n = len(rows)
    solved = [r for r in rows if r["result"]["status"] == "solved"]
    turns_to_solve = [r["result"]["turns"] for r in solved]
    all_valence = [v for r in rows for v in r["result"]["valence_trajectory"]]
    per_episode_mean_valence = [mean(r["result"]["valence_trajectory"])
                                for r in rows if r["result"]["valence_trajectory"]]
    return {
        "n_episodes": n,
        "n_solved": len(solved),
        "solve_rate": (len(solved) / n) if n else None,
        "turns_to_solve_mean": mean(turns_to_solve) if turns_to_solve else None,
        "turns_to_solve_stdev": stdev(turns_to_solve) if len(turns_to_solve) > 1 else None,
        "valence_mean": mean(all_valence) if all_valence else None,
        "valence_stdev": stdev(all_valence) if len(all_valence) > 1 else None,
        "per_episode_mean_valence": per_episode_mean_valence,
    }


def _fmt(x, spec="{:.2f}"):
    return spec.format(x) if x is not None else "n/a"


def _maze_role_cell(records: list, maze_index: int, role: str) -> str:
    """One markdown-table cell summarizing all repeats of (maze, role):
    'solved 2/3, turns 8.5' or, for repeats=1, just 'solved, turns 8'."""
    rows = [r for r in records if r["maze_index"] == maze_index and r["role"] == role]
    if not rows:
        return "n/a"
    n = len(rows)
    solved = [r for r in rows if r["result"]["status"] == "solved"]
    turns = [r["result"]["turns"] for r in solved]
    turns_text = f", turns {_fmt(mean(turns))}" if turns else ""
    if n == 1:
        return f"{rows[0]['result']['status']}, turns {rows[0]['result']['turns']}"
    return f"solved {len(solved)}/{n}{turns_text}"


def _write_markdown(out_dir: Path, records: list, summary: dict):
    t, a = summary["teacher"], summary["adversary"]
    repeats = summary["repeats"]
    lines = [
        "# Batch results summary",
        "",
        f"{summary['n_mazes']} mazes x 2 conditions (teacher, adversary) x "
        f"{repeats} repeat(s) = {summary['n_episodes']} episodes.",
        "",
        "**This is an initial pass, not a final analysis** -- treat gaps here as a "
        "starting point for the deeper offline analysis the raw per-episode JSON "
        "in `episodes/` supports, not a conclusion on their own"
        + (" (repeats=1: each maze/condition is a single trajectory, not a "
           "distribution)." if repeats == 1 else "."),
        "",
        "| condition | solve rate | turns-to-solve (mean +/- sd) | valence (mean +/- sd) |",
        "|---|---|---|---|",
        f"| teacher | {_fmt(t['solve_rate'], '{:.0%}')} ({t['n_solved']}/{t['n_episodes']}) | "
        f"{_fmt(t['turns_to_solve_mean'])} +/- {_fmt(t['turns_to_solve_stdev'])} | "
        f"{_fmt(t['valence_mean'], '{:+.3f}')} +/- {_fmt(t['valence_stdev'])} |",
        f"| adversary | {_fmt(a['solve_rate'], '{:.0%}')} ({a['n_solved']}/{a['n_episodes']}) | "
        f"{_fmt(a['turns_to_solve_mean'])} +/- {_fmt(a['turns_to_solve_stdev'])} | "
        f"{_fmt(a['valence_mean'], '{:+.3f}')} +/- {_fmt(a['valence_stdev'])} |",
        "",
        "## Per-maze (paired: same maze, both conditions"
        + (f", {repeats} repeats each)" if repeats > 1 else ")"),
        "",
        "| maze | solution length | teacher | adversary |",
        "|---|---|---|---|",
    ]
    for row in summary["per_maze"]:
        i = row["maze_index"]
        lines.append(
            f"| {i:02d} | {row['solution_length']} | "
            f"{_maze_role_cell(records, i, 'teacher')} | "
            f"{_maze_role_cell(records, i, 'adversary')} |"
        )
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines))


def _write_plots(out_dir: Path, records: list, summary: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze] matplotlib not installed -- skipping plots "
              "(summary.json / summary.md still written).")
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # valence trajectories, one line per episode, colored by condition
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"teacher": "tab:blue", "adversary": "tab:red"}
    seen_label = set()
    for r in records:
        vt = r["result"]["valence_trajectory"]
        if not vt:
            continue
        label = r["role"] if r["role"] not in seen_label else None
        seen_label.add(r["role"])
        ax.plot(range(1, len(vt) + 1), vt, color=colors[r["role"]], alpha=0.5, label=label)
    ax.set_xlabel("turn")
    ax.set_ylabel("valence projection")
    ax.set_title("Valence trajectory per episode")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "valence_trajectories.png", dpi=150)
    plt.close(fig)

    # solve rate bar chart
    fig, ax = plt.subplots(figsize=(4, 5))
    conds = ["teacher", "adversary"]
    rates = [summary[c]["solve_rate"] or 0.0 for c in conds]
    ax.bar(conds, rates, color=[colors[c] for c in conds])
    ax.set_ylim(0, 1)
    ax.set_ylabel("solve rate")
    ax.set_title("Solve rate by condition")
    fig.tight_layout()
    fig.savefig(plots_dir / "solve_rates.png", dpi=150)
    plt.close(fig)

    print(f"[analyze] wrote plots to {plots_dir}")


def analyze_batch(out_dir) -> dict:
    out_dir = Path(out_dir)
    records = _load_episodes(out_dir)
    teacher = _condition_stats(records, "teacher")
    adversary = _condition_stats(records, "adversary")

    n_mazes = max((r["maze_index"] for r in records), default=-1) + 1
    repeats = max((r.get("repeat", 0) for r in records), default=0) + 1
    per_maze = []
    for i in range(n_mazes):
        rows_here = [r for r in records if r["maze_index"] == i]
        per_maze.append({
            "maze_index": i,
            "solution_length": rows_here[0]["solution_length"] if rows_here else None,
            "teacher": {
                "n_repeats": sum(1 for r in rows_here if r["role"] == "teacher"),
                "n_solved": sum(1 for r in rows_here if r["role"] == "teacher" and r["result"]["status"] == "solved"),
                "statuses": [r["result"]["status"] for r in rows_here if r["role"] == "teacher"],
                "turns": [r["result"]["turns"] for r in rows_here if r["role"] == "teacher"],
            },
            "adversary": {
                "n_repeats": sum(1 for r in rows_here if r["role"] == "adversary"),
                "n_solved": sum(1 for r in rows_here if r["role"] == "adversary" and r["result"]["status"] == "solved"),
                "statuses": [r["result"]["status"] for r in rows_here if r["role"] == "adversary"],
                "turns": [r["result"]["turns"] for r in rows_here if r["role"] == "adversary"],
            },
        })

    summary = {
        "n_mazes": n_mazes, "repeats": repeats, "n_episodes": len(records),
        "teacher": teacher, "adversary": adversary, "per_maze": per_maze,
    }
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    _write_markdown(out_dir, records, summary)
    _write_plots(out_dir, records, summary)
    print(f"[analyze] wrote {out_dir / 'summary.json'} and {out_dir / 'summary.md'}")
    return summary


if __name__ == "__main__":
    analyze_batch(Path(sys.argv[1]))
