"""Run the hand-authored problems in problems.py against a model.

    python -m src.capability_probe.probe
    python -m src.capability_probe.probe --verbose --max-new-tokens 2048
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from . import runner as R
from .problems import (
    PROBLEMS, PROBLEMS_SET2, PROBLEMS_SET3, PROBLEMS_SET4, PROBLEMS_SET5, PROBLEMS_SET6,
)

_ALL_SETS = [PROBLEMS, PROBLEMS_SET2, PROBLEMS_SET3, PROBLEMS_SET4, PROBLEMS_SET5, PROBLEMS_SET6]
PROBLEM_SETS = {str(i + 1): s for i, s in enumerate(_ALL_SETS)}
PROBLEM_SETS["all"] = [p for s in _ALL_SETS for p in s]

_FINAL_RE = re.compile(r"^\s*FINAL ANSWER:[ \t]*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def build_prompt(problem) -> str:
    return (
        f"{problem.prompt}\n\n"
        "Work through it step by step, then end your response with a final line in "
        "EXACTLY this format (no extra words on that line):\n"
        "FINAL ANSWER: <your answer>"
    )


def extract_final_answer(reply: str) -> str | None:
    matches = _FINAL_RE.findall(reply)
    return matches[-1].strip() if matches else None


def grade(problem, extracted: str | None) -> str:
    """Returns "correct" | "wrong" | "unparsed". If problem.checker is set
    (e.g. mazes: any valid path counts, not just one canonical route), it's
    used instead of string matching. Otherwise exact-match first (case
    matters for the code-output problem), then a normalized fallback."""
    if extracted is None:
        return "unparsed"
    if problem.checker is not None:
        return "correct" if problem.checker(extracted) else "wrong"
    if extracted.strip() == problem.answer:
        return "correct"
    if extracted.strip().lower().strip(".\"'") == problem.answer.lower():
        return "correct"
    return "wrong"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=1500)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--out", default="runs/capability_probe.json")
    ap.add_argument("--verbose", action="store_true", help="print full replies live")
    ap.add_argument("--domain", default=None, help="only run problems from this domain")
    ap.add_argument("--set", default="1", choices=list(PROBLEM_SETS),
                    help="1=original hand-authored set, 2=user-supplied set, all=both")
    args = ap.parse_args()

    problems = [p for p in PROBLEM_SETS[args.set] if args.domain is None or p.domain == args.domain]
    print(f"[data] {len(problems)} hand-authored problems")
    model, tok = R.load(args.model)

    results = []
    for i, problem in enumerate(problems):
        prompt = build_prompt(problem)
        reply = R.generate(model, tok, [{"role": "user", "content": prompt}],
                           max_new_tokens=args.max_new_tokens, sample=args.sample)
        extracted = extract_final_answer(reply)
        outcome = grade(problem, extracted)

        expected_display = problem.answer if problem.checker is None else f"(checker: {problem.notes})"
        print(f"\n[{i + 1}/{len(problems)}] {problem.domain} ({problem.difficulty}) -> {outcome.upper()}")
        print(f"  expected: {expected_display}   model said: {extracted!r}")
        if args.verbose:
            print(f"  --- full reply ---\n{reply}\n  ------------------")

        results.append({
            "domain": problem.domain, "difficulty": problem.difficulty,
            "prompt": problem.prompt, "expected": problem.answer or problem.notes,
            "extracted": extracted, "outcome": outcome, "reply": reply,
        })

    n_correct = sum(r["outcome"] == "correct" for r in results)
    n_unparsed = sum(r["outcome"] == "unparsed" for r in results)
    print(f"\n===== summary: {n_correct}/{len(results)} correct "
          f"({n_unparsed} unparsed) =====")
    for r in results:
        print(f"  [{r['outcome']:>9}] {r['domain']:<13} {r['difficulty']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": vars(args), "results": results}, open(out, "w"), indent=2)
    print(f"\n[done] wrote {out}")


if __name__ == "__main__":
    main()
