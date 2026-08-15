"""Benchmark a model on ZebraLogic grid puzzles.

    python -m src.zebra_logic.benchmark --sizes small
    python -m src.zebra_logic.benchmark --sizes small --n 10 --verbose   # quick check
    python -m src.zebra_logic.benchmark --fresh --sizes medium           # contamination check:
                                                                          # freshly-generated puzzles,
                                                                          # never published anywhere

Ground truth comes from WildEval/ZebraLogic (see data.py for why, not the
allenai/ZebraLogicBench public repo, which masks all solutions). Prompt
format and scoring match ZeroEval (https://github.com/yuchenlin/ZeroEval)
so numbers are comparable to the public leaderboard.

--fresh switches to generate.py's own generator instead: same size/clue
style, but never-published puzzles, so a solve-rate GAP between this and
the real benchmark (at the same size) is the contamination signature --
comparable numbers instead suggest genuine reasoning, not memorization.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from . import data as D
from . import generate as G
from . import prompts as P
from . import runner as R
from . import scoring as S

SIZE_GROUPS = {
    "small": D.SMALL_SIZES,
    "medium": D.MEDIUM_SIZES,
    "large": D.LARGE_SIZES,
    "xl": D.XL_SIZES,
    "easy": D.EASY_SIZES,
    "hard": D.HARD_SIZES,
    "all": None,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=None, choices=list(SIZE_GROUPS),
                    help="default: 'small' normally, G.DEFAULT_SIZE_MIX (skewed larger) with --fresh")
    ap.add_argument("--fresh", action="store_true",
                    help="use freshly-generated puzzles (generate.py) instead of WildEval/ZebraLogic")
    ap.add_argument("--fresh-per-size", type=int, default=None,
                    help="puzzles per size bucket when --fresh (default: G.DEFAULT_SIZE_MIX's skewed counts)")
    ap.add_argument("--n", type=int, default=None, help="cap number of puzzles (random subsample)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/zebra_logic_results.json")
    ap.add_argument("--verbose", action="store_true", help="print each puzzle's result live")
    args = ap.parse_args()

    if args.fresh:
        if args.sizes is None and args.fresh_per_size is None:
            puzzles = G.generate_puzzles(seed=args.seed)  # DEFAULT_SIZE_MIX: skewed larger
            label = "default mix, skewed larger"
        else:
            sizes = SIZE_GROUPS[args.sizes or "small"] or tuple(
                f"{h}*{a}" for h in range(2, 7) for a in range(2, 7))
            per_size = args.fresh_per_size if args.fresh_per_size is not None else 10
            puzzles = G.generate_puzzles(seed=args.seed, size_counts={s: per_size for s in sizes})
            label = args.sizes or "small"
        print(f"[data] {len(puzzles)} FRESH (never-published) puzzles ({label})")
    else:
        puzzles = D.load_puzzles(sizes=SIZE_GROUPS[args.sizes or "small"])
        print(f"[data] {len(puzzles)} puzzles ({args.sizes or 'small'})")
    if args.n is not None:
        random.Random(args.seed).shuffle(puzzles)
        puzzles = puzzles[:args.n]

    model, tok = R.load(args.model)

    results = []
    for i, puzzle in enumerate(puzzles):
        prompt = P.build_prompt(puzzle)
        output = R.generate(model, tok, prompt, max_new_tokens=args.max_new_tokens, sample=args.sample)
        score = S.score_puzzle(puzzle, output)
        if args.verbose:
            print(f"[{i + 1}/{len(puzzles)}] {puzzle.id} ({puzzle.size}) "
                  f"solved={score.solved} parsed={score.parsed} "
                  f"cells={score.correct_cells}/{score.total_cells}", flush=True)
        results.append({
            "id": puzzle.id, "size": puzzle.size, "solved": score.solved,
            "correct_cells": score.correct_cells, "total_cells": score.total_cells,
            "parsed": score.parsed, "output": output,
        })

    puzzle_acc = mean(r["solved"] for r in results)
    cell_acc = sum(r["correct_cells"] for r in results) / sum(r["total_cells"] for r in results)
    no_answer = mean(not r["parsed"] for r in results)

    print(f"\n===== {args.sizes} ({len(results)} puzzles) =====")
    print(f"  Puzzle Acc: {puzzle_acc * 100:.2f}%")
    print(f"  Cell Acc:   {cell_acc * 100:.2f}%")
    print(f"  No answer:  {no_answer * 100:.2f}%")

    by_size = defaultdict(list)
    for r in results:
        by_size[r["size"]].append(r["solved"])
    print("\n  by size:")
    for size in sorted(by_size):
        vals = by_size[size]
        print(f"    {size}: {sum(vals)}/{len(vals)} = {mean(vals) * 100:.1f}%")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "config": vars(args), "puzzle_acc": puzzle_acc, "cell_acc": cell_acc,
        "no_answer": no_answer, "results": results,
    }, open(out, "w"), indent=2)
    print(f"\n[done] wrote {out}")


if __name__ == "__main__":
    main()
