"""Turn a results JSON into the figures and summary for the report."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROLES = ("honest", "dismissive")
COLORS = {"honest": "#2a9d8f", "dismissive": "#e76f51"}


def _valence_by_trial(sessions, role):
    """Mean valence at each trial index across sessions (induction phase)."""
    per_trial: dict[int, list[float]] = {}
    for s in sessions:
        for tr in s["induction"][role]:
            if tr["valence"] is not None:
                per_trial.setdefault(tr["trial"], []).append(tr["valence"])
    xs = sorted(per_trial)
    return xs, [mean(per_trial[x]) for x in xs]


def _transfer_solve(sessions, role):
    rates = []
    for s in sessions:
        tr = s["transfer"][role]
        rates.append(mean(t["was_correct"] for t in tr) if tr else 0.0)
    return rates


def analyze(results_path: str, fig_dir: str):
    data = json.load(open(results_path))
    sessions = data["sessions"]
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    # ── figure 1: induction valence trajectory ──────────────────────────────
    plt.figure(figsize=(7, 4.5))
    for role in ROLES:
        xs, ys = _valence_by_trial(sessions, role)
        if xs:
            plt.plot([x + 1 for x in xs], ys, marker="o", ms=3,
                     color=COLORS[role], label=role)
    plt.xlabel("trial")
    plt.ylabel("valence projection onto axis")
    plt.title("Induction: valence read-out over trials")
    plt.legend()
    plt.tight_layout()
    f1 = Path(fig_dir) / "valence_trajectory.png"
    plt.savefig(f1, dpi=150)
    plt.close()

    # ── figure 2: transfer solve-rate ───────────────────────────────────────
    means = {r: _transfer_solve(sessions, r) for r in ROLES}
    plt.figure(figsize=(5.5, 4.5))
    xs = range(len(ROLES))
    bars = [mean(means[r]) for r in ROLES]
    errs = [pstdev(means[r]) if len(means[r]) > 1 else 0.0 for r in ROLES]
    plt.bar(xs, bars, yerr=errs, capsize=5, color=[COLORS[r] for r in ROLES])
    plt.xticks(list(xs), ROLES)
    plt.ylabel("transfer solve rate")
    plt.title("Downstream transfer (fresh solvable task, honest feedback for both)")
    plt.ylim(0, 1)
    plt.tight_layout()
    f2 = Path(fig_dir) / "transfer_solve_rate.png"
    plt.savefig(f2, dpi=150)
    plt.close()

    # ── summary + sanity ────────────────────────────────────────────────────
    def agg(key):
        vals = [s["sanity"][key] for s in sessions if s["sanity"][key] is not None]
        return mean(vals) if vals else None

    honest_ok = sum(s["sanity"]["honest_told_correct_matches_truth"] for s in sessions)
    dismissive_ok = sum(s["sanity"]["dismissive_always_told_wrong"] for s in sessions)
    print("\n===== summary =====")
    for r in ROLES:
        print(f"  {r:>11}  transfer solve rate = {mean(means[r]):.3f} "
              f"(+/- {pstdev(means[r]) if len(means[r])>1 else 0:.3f})")
    print("\n===== sanity (must hold) =====")
    print(f"  honest feedback matches truth, all trials  : {honest_ok}/{len(sessions)} sessions")
    print(f"  dismissive always told wrong, all trials   : {dismissive_ok}/{len(sessions)} sessions")
    print("\n===== transparency (dismissive's real competence, hidden from it) =====")
    print(f"  honest true solve rate     (induction) : {agg('honest_true_solve_rate'):.3f}")
    print(f"  dismissive true solve rate (induction) : {agg('dismissive_true_solve_rate'):.3f}")
    print(f"\nfigures: {f1}  |  {f2}")
    return {"figures": [str(f1), str(f2)]}
