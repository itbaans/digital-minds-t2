"""Run the controllability experiment end to end.

    python -m src.controllability.experiment \
        --vaa-dir artifacts/concept_vectors/vaa_qwen3_4b_instruct/baseline/vaa \
        --n-sessions 20 --n-trials 20 --probe-every 5

Requires the VAA artifact, produced once (no training) by:
    python -m vaa.extract_vaa --base-model Qwen/Qwen3-4B-Instruct-2507
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev

import torch

from . import env as E
from . import runner as R
from .axis import load_axis, direction_from_contrast, project


def _corr(xs, ys):
    n = len(xs)
    if n == 0:
        return None
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return (num / (dx * dy)) if dx > 0 and dy > 0 else 0.0


def _solve(trials):
    return mean(t.was_correct for t in trials) if trials else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vaa-dir", required=True)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-sessions", type=int, default=10)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--n-units", type=int, default=4)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--probe-every", type=int, default=0)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--steer-rescue", type=float, default=0.0)
    ap.add_argument("--out", default="runs/controllability.json")
    ap.add_argument("--fig-dir", default="runs/figures")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_sessions, args.n_trials = 2, 6

    model, tok, blocks = R.load(args.model)
    layer, cv_raw, cv_unit, n_layers = load_axis(args.vaa_dir, args.layer)
    print(f"[axis] layer {layer} | |cv|={float(cv_raw.norm()):.3f}")
    gen_kw = dict(sample=args.sample)

    sessions = []
    pos_pool, neg_pool = [], []   # for the self-derived axis (contingent vs yoked)

    for s in range(args.n_sessions):
        seed = args.base_seed + s
        spec = E.make_spec(seed, args.n_trials, args.n_units)
        transfer = E.make_spec(seed + 500_000, args.n_trials, args.n_units)
        common = dict(layer=layer, cv_unit=cv_unit, n_layers=n_layers,
                      probe_every=args.probe_every, gen_kw=gen_kw)

        m_tr, m_msgs, m_out, m_acts = R.run_block(model, tok, blocks, spec, "contingent", **common)
        y_tr, y_msgs, _, y_acts = R.run_block(model, tok, blocks, spec, "yoked", replay=m_out, **common)
        c_tr, c_msgs, _, _ = R.run_block(model, tok, blocks, spec, "control", **common)

        pos_pool += m_acts
        neg_pool += y_acts

        steer = (layer, cv_raw, args.steer_rescue) if args.steer_rescue > 0 else None
        m_xf, *_ = R.run_block(model, tok, blocks, transfer, "contingent", messages=m_msgs, **common)
        y_xf, *_ = R.run_block(model, tok, blocks, transfer, "contingent", messages=y_msgs, steer=steer, **common)
        c_xf, *_ = R.run_block(model, tok, blocks, transfer, "contingent", messages=c_msgs, **common)

        sessions.append({
            "seed": seed,
            "induction": {a: [asdict(t) for t in tr]
                          for a, tr in (("contingent", m_tr), ("yoked", y_tr), ("control", c_tr))},
            "transfer": {a: [asdict(t) for t in tr]
                         for a, tr in (("contingent", m_xf), ("yoked", y_xf), ("control", c_xf))},
            "sanity": {
                "corr_action_outcome_master": _corr([int(t.was_correct) for t in m_tr],
                                                    [int(t.resolved) for t in m_tr]),
                "corr_action_outcome_yoked": _corr([int(t.was_correct) for t in y_tr],
                                                   [int(t.resolved) for t in y_tr]),
                "outcomes_matched": [t.resolved for t in m_tr] == [t.resolved for t in y_tr[:len(m_tr)]],
            },
        })
        print(f"[session {s:>2}] transfer solve_rate  "
              f"master={_solve(m_xf):.2f}  yoked={_solve(y_xf):.2f}  control={_solve(c_xf):.2f}")

    # self-derived axis (contingent vs yoked) vs the VAA -- a track question
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
