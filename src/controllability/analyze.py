"""Turn a results JSON into the figures and summary for the report."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ("contingent", "yoked", "control")
COLORS = {"contingent": "#2a9d8f", "yoked": "#e76f51", "control": "#8d99ae"}


def _valence_by_trial(sessions, arm):
    """Mean valence at each trial index across sessions (induction phase)."""
    per_trial: dict[int, list[float]] = {}
    for s in sessions:
        for tr in s["induction"][arm]:
            if tr["valence"] is not None:
                per_trial.setdefault(tr["trial"], []).append(tr["valence"])
    xs = sorted(per_trial)
    return xs, [mean(per_trial[x]) for x in xs]


def _transfer_solve(sessions, arm):
    rates = []
    for s in sessions:
        tr = s["transfer"][arm]
        rates.append(mean(t["was_correct"] for t in tr) if tr else 0.0)
    return rates


def analyze(results_path: str, fig_dir: str):
    data = json.load(open(results_path))
    sessions = data["sessions"]
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    # ── figure 1: induction valence trajectory ──────────────────────────────
    plt.figure(figsize=(7, 4.5))
    for arm in ARMS:
        xs, ys = _valence_by_trial(sessions, arm)
        if xs:
            plt.plot([x + 1 for x in xs], ys, marker="o", ms=3,
                     color=COLORS[arm], label=arm)
    plt.xlabel("trial")
    plt.ylabel("valence projection onto axis")
    plt.title("Induction: valence read-out over trials")
    plt.legend()
    plt.tight_layout()
    f1 = Path(fig_dir) / "valence_trajectory.png"
    plt.savefig(f1, dpi=150)
    plt.close()

    # ── figure 2: transfer solve-rate ───────────────────────────────────────
    means = {a: _transfer_solve(sessions, a) for a in ARMS}
    plt.figure(figsize=(5.5, 4.5))
    xs = range(len(ARMS))
    bars = [mean(means[a]) for a in ARMS]
    errs = [pstdev(means[a]) if len(means[a]) > 1 else 0.0 for a in ARMS]
    plt.bar(xs, bars, yerr=errs, capsize=5, color=[COLORS[a] for a in ARMS])
    plt.xticks(list(xs), ARMS)
    plt.ylabel("transfer solve rate")
    plt.title("Downstream transfer (fresh solvable task)")
    plt.ylim(0, 1)
    plt.tight_layout()
    f2 = Path(fig_dir) / "transfer_solve_rate.png"
    plt.savefig(f2, dpi=150)
    plt.close()

    # ── summary + sanity ────────────────────────────────────────────────────
    def agg(key, sign=1):
        vals = [s["sanity"][key] for s in sessions if s["sanity"][key] is not None]
        return mean(vals) if vals else None

    matched = sum(s["sanity"]["outcomes_matched"] for s in sessions)
    print("\n===== summary =====")
    for a in ARMS:
        print(f"  {a:>11}  transfer solve rate = {mean(means[a]):.3f} "
              f"(+/- {pstdev(means[a]) if len(means[a])>1 else 0:.3f})")
    print("\n===== sanity (must hold) =====")
    print(f"  corr(action, outcome) master ~ 1 : {agg('corr_action_outcome_master'):.3f}")
    print(f"  corr(action, outcome) yoked  ~ 0 : {agg('corr_action_outcome_yoked'):.3f}")
    print(f"  outcomes matched master==yoked   : {matched}/{len(sessions)}")
    print(f"\nfigures: {f1}  |  {f2}")
    return {"figures": [str(f1), str(f2)]}
