"""Run the feedback-accuracy experiment end to end.

    python -m src.controllability.experiment \
        --vaa-dir artifacts/concept_vectors/vaa_qwen3_4b_instruct/baseline/vaa \
        --n-sessions 20 --n-trials 20 --probe-every 5

Two roles share one diverse problem set per session: "honest" gets accurate
feedback, "dismissive" is always told it was wrong regardless of truth. See
env.py's module docstring for the confound this design accepts (dismissive
feedback is not rate-matched to truth, unlike an earlier "yoked" version of
this experiment).

Requires the VAA artifact, produced once (no training) by:
    python -m vaa.extract_vaa --base-model Qwen/Qwen3-4B-Instruct-2507
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import torch

from . import env as E
from . import runner as R
from .axis import load_axis, direction_from_contrast


def _solve(trials):
    return mean(t.was_correct for t in trials) if trials else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vaa-dir", required=True)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-sessions", type=int, default=10)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--probe-every", type=int, default=0)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--steer-rescue", type=float, default=0.0)
    ap.add_argument("--out", default="runs/controllability.json")
    ap.add_argument("--fig-dir", default="runs/figures")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--verbose", action="store_true",
                    help="print each round's prompt/reply/feedback live")
    args = ap.parse_args()
    if args.smoke:
        args.n_sessions, args.n_trials = 2, 6

    model, tok, blocks = R.load(args.model)
    layer, cv_raw, cv_unit, n_layers = load_axis(args.vaa_dir, args.layer)
    print(f"[axis] layer {layer} | |cv|={float(cv_raw.norm()):.3f}")
    gen_kw = dict(sample=args.sample)

    sessions = []
    pos_pool, neg_pool = [], []   # for the self-derived axis (honest vs dismissive)

    for s in range(args.n_sessions):
        seed = args.base_seed + s
        spec = E.make_spec(seed, args.n_trials)
        # transfer: fresh, easy (difficulty=1 only) so it's reliably solvable
        transfer = E.make_spec(seed + 500_000, args.n_trials, difficulties=(1,))
        common = dict(layer=layer, cv_unit=cv_unit, n_layers=n_layers,
                      probe_every=args.probe_every, gen_kw=gen_kw, verbose=args.verbose)

        h_tr, h_msgs, h_acts = R.run_block(model, tok, blocks, spec, "honest", **common)
        d_tr, d_msgs, d_acts = R.run_block(model, tok, blocks, spec, "dismissive", **common)

        pos_pool += h_acts
        neg_pool += d_acts

        # Transfer feedback is always honest for BOTH groups -- this isolates
        # whether the induction history (honest vs dismissive) carries into
        # behavior/valence on a fresh, fairly-graded task.
        steer = (layer, cv_raw, args.steer_rescue) if args.steer_rescue > 0 else None
        h_xf, *_ = R.run_block(model, tok, blocks, transfer, "honest", messages=h_msgs, **common)
        d_xf, *_ = R.run_block(model, tok, blocks, transfer, "honest", messages=d_msgs, steer=steer, **common)

        sessions.append({
            "seed": seed,
            "induction": {r: [asdict(t) for t in tr]
                          for r, tr in (("honest", h_tr), ("dismissive", d_tr))},
            "transfer": {r: [asdict(t) for t in tr]
                         for r, tr in (("honest", h_xf), ("dismissive", d_xf))},
            "sanity": {
                "honest_told_correct_matches_truth": all(t.told_correct == t.was_correct for t in h_tr),
                "dismissive_always_told_wrong": all(t.told_correct is False for t in d_tr),
                "honest_true_solve_rate": _solve(h_tr),
                "dismissive_true_solve_rate": _solve(d_tr),
            },
        })
        print(f"[session {s:>2}] transfer solve_rate  honest={_solve(h_xf):.2f}  "
              f"dismissive={_solve(d_xf):.2f}   (induction true-solve: "
              f"honest={_solve(h_tr):.2f} dismissive={_solve(d_tr):.2f})")

    # self-derived axis (honest vs dismissive) vs the VAA -- a track question
    self_cos = None
    if pos_pool and neg_pool:
        self_axis = direction_from_contrast(torch.stack(pos_pool), torch.stack(neg_pool))
        self_cos = float(torch.dot(self_axis, cv_unit))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"self_axis": self_axis, "layer": layer, "cos_with_vaa": self_cos},
                   str(Path(args.out).with_suffix(".self_axis.pt")))
        print(f"[axis] cosine(self-derived, VAA) = {self_cos:+.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": vars(args), "layer": layer, "self_axis_cos_vaa": self_cos,
               "sessions": sessions}, open(out, "w"), indent=2)
    print(f"[done] wrote {out}")

    try:
        from . import analyze
        analyze.analyze(str(out), args.fig_dir)
    except Exception as e:  # analysis is optional; don't lose the run
        print(f"[warn] analysis skipped: {e}")


if __name__ == "__main__":
    main()
