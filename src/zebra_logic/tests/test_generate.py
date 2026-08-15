"""Torch-free tests for the fresh zebra-puzzle generator: uniqueness,
determinism, and compatibility with prompts.py/scoring.py.

Run: python -m src.zebra_logic.tests.test_generate
(Slower than the other suites -- generates real puzzles, including a couple
of large ones -- expect several seconds, not instant.)
"""
import random

from src.zebra_logic import generate as G
from src.zebra_logic import prompts as P
from src.zebra_logic import scoring as S


def test_generated_puzzle_is_genuinely_unique_small():
    rng = random.Random(123)
    categories, assignment = G._random_assignment(rng, n_houses=3, n_attrs=3)
    clues = G._minimal_clue_set(rng, categories, assignment, n_houses=3)
    values_per_cat = {cat: assignment[cat] for cat in categories}
    # re-check with a much larger budget than generation used, cap=3, to be extra sure
    count = G._count_solutions(categories, values_per_cat, clues, n_houses=3, cap=3, node_budget=10_000_000)
    assert count == 1


def test_generated_puzzle_is_genuinely_unique_larger():
    rng = random.Random(456)
    categories, assignment = G._random_assignment(rng, n_houses=5, n_attrs=5)
    clues = G._minimal_clue_set(rng, categories, assignment, n_houses=5)
    values_per_cat = {cat: assignment[cat] for cat in categories}
    count = G._count_solutions(categories, values_per_cat, clues, n_houses=5, cap=3, node_budget=10_000_000)
    assert count == 1


def test_generate_puzzle_deterministic_for_seed():
    a = G.generate_puzzle(random.Random(7), "3*3", 0)
    b = G.generate_puzzle(random.Random(7), "3*3", 0)
    assert a.puzzle_text == b.puzzle_text
    assert a.solution == b.solution


def test_generate_puzzle_solution_matches_declared_values():
    """Every value the preamble declares for a category must appear exactly
    once across the houses in the solution (each person unique)."""
    p = G.generate_puzzle(random.Random(8), "4*3", 0)
    for cat in p.header[1:]:
        values_in_solution = sorted(p.solution[f"House {h + 1}"][cat] for h in range(p.n_houses))
        assert len(set(values_in_solution)) == p.n_houses  # all distinct


def test_generated_puzzle_compatible_with_prompts_and_scoring():
    """A generated Puzzle should work with the exact same build_prompt /
    score_puzzle pipeline as real WildEval/ZebraLogic puzzles."""
    p = G.generate_puzzle(random.Random(9), "2*3", 0)
    prompt = P.build_prompt(p)
    assert p.puzzle_text in prompt

    import json
    perfect_output = json.dumps({"reasoning": "...", "solution": p.solution})
    result = S.score_puzzle(p, perfect_output)
    assert result.solved
    assert result.correct_cells == result.total_cells


def test_generate_puzzles_respects_size_counts():
    puzzles = G.generate_puzzles(seed=1, size_counts={"2*2": 3, "3*2": 2})
    sizes = sorted(p.size for p in puzzles)
    assert sizes == sorted(["2*2"] * 3 + ["3*2"] * 2)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
