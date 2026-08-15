"""JSON extraction + scoring, faithfully matching ZeroEval's
src/evaluation/zebra_grid_eval.py + eval_utils.py logic: brace-matching
extraction of the LAST complete top-level JSON object in the model's raw
output, then case-insensitive/stripped cell-by-cell comparison.

Torch-free: pure string/dict logic, no model code here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional


def extract_last_complete_json(text: str) -> Optional[dict]:
    """Same brace-matching approach as ZeroEval's extract_last_complete_json
    -- finds the LAST balanced top-level {...} span (so preamble/reasoning
    text before the JSON block doesn't confuse it), tolerant of models that
    restate the JSON or add trailing commentary after it."""
    stack = []
    start = None
    last = None
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
            if start is None:
                start = i
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack:
                    last = text[start:i + 1]
                    start = None
    if last is None:
        return None
    try:
        return json.loads(last.replace("\n", ""))
    except json.JSONDecodeError:
        return None


@dataclass
class ScoreResult:
    solved: bool
    correct_cells: int
    total_cells: int
    parsed: bool        # False if no usable "solution" JSON was found at all


def score_puzzle(puzzle, model_output: str) -> ScoreResult:
    total_cells = sum(len(attrs) for attrs in puzzle.solution.values())
    parsed = extract_last_complete_json(model_output)
    if parsed is None or not isinstance(parsed, dict) or "solution" not in parsed \
            or not isinstance(parsed["solution"], dict):
        return ScoreResult(solved=False, correct_cells=0, total_cells=total_cells, parsed=False)

    pred = parsed["solution"]
    correct = 0
    for house, attrs in puzzle.solution.items():
        pred_house = pred.get(house)
        if not isinstance(pred_house, dict):
            continue
        for attr, truth_val in attrs.items():
            pv = pred_house.get(attr)
            if isinstance(pv, list):
                pv = pv[0] if pv else None
            if not isinstance(pv, str):
                continue
            if truth_val.lower().strip() == pv.lower().strip():
                correct += 1
    return ScoreResult(solved=(correct == total_cells), correct_cells=correct,
                       total_cells=total_cells, parsed=True)
