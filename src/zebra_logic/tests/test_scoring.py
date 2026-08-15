"""Torch-free tests for JSON extraction, scoring, prompt building, and the
solution-parsing logic. No network / no GPU required.

Run: python -m src.zebra_logic.tests.test_scoring
"""
from src.zebra_logic import data as D
from src.zebra_logic import prompts as P
from src.zebra_logic import scoring as S


def _puzzle():
    return D.Puzzle(
        id="t1", size="2*2", n_houses=2, puzzle_text="There are 2 houses...",
        header=("House", "Name", "Pet"),
        solution={"House 1": {"Name": "Eric", "Pet": "cat"},
                  "House 2": {"Name": "Arnold", "Pet": "dog"}},
    )


# ── data parsing ──────────────────────────────────────────────────────────

def test_row_to_puzzle_parses_header_and_solution():
    raw = {"header": ["House", "Name", "Pet"],
           "rows": [["1", "Eric", "cat"], ["2", "Arnold", "dog"]]}
    puzzle = D._row_to_puzzle("t1", "2*2", "puzzle text", raw)
    assert puzzle.n_houses == 2
    assert puzzle.header == ("House", "Name", "Pet")
    assert puzzle.solution["House 1"] == {"Name": "Eric", "Pet": "cat"}
    assert puzzle.solution["House 2"] == {"Name": "Arnold", "Pet": "dog"}


def test_size_groups_are_disjoint_and_cover_all_25_sizes():
    groups = [D.SMALL_SIZES, D.MEDIUM_SIZES, D.LARGE_SIZES, D.XL_SIZES]
    seen = set()
    for g in groups:
        assert not (seen & set(g)), f"overlap in size groups: {seen & set(g)}"
        seen |= set(g)
    all_sizes = {f"{h}*{a}" for h in range(2, 7) for a in range(2, 7)}
    assert seen == all_sizes, f"missing: {all_sizes - seen}"
    # easy/hard is a different split of the same 25 sizes
    assert set(D.EASY_SIZES) | set(D.HARD_SIZES) == all_sizes
    assert not (set(D.EASY_SIZES) & set(D.HARD_SIZES))


# ── prompt building ───────────────────────────────────────────────────────

def test_build_prompt_includes_puzzle_text_and_json_skeleton():
    puzzle = _puzzle()
    prompt = P.build_prompt(puzzle)
    assert puzzle.puzzle_text in prompt
    assert '"House 1"' in prompt and '"House 2"' in prompt
    assert '"Name": "___"' in prompt and '"Pet": "___"' in prompt
    assert "{puzzle}" not in prompt and "{json_template}" not in prompt  # both substituted


# ── JSON extraction ───────────────────────────────────────────────────────

def test_extract_last_complete_json_basic():
    text = 'blah {"a": 1} more text {"reasoning": "x", "solution": {}}'
    assert S.extract_last_complete_json(text) == {"reasoning": "x", "solution": {}}


def test_extract_last_complete_json_none_when_absent():
    assert S.extract_last_complete_json("no json here at all") is None


def test_extract_last_complete_json_handles_nested_braces():
    text = '{"reasoning": "a {nested} thing", "solution": {"House 1": {"Name": "Eric"}}}'
    result = S.extract_last_complete_json(text)
    assert result["solution"]["House 1"]["Name"] == "Eric"


def test_extract_last_complete_json_survives_trailing_commentary():
    text = '{"reasoning": "x", "solution": {"House 1": {"Name": "Eric"}}}\n\nHope that helps!'
    result = S.extract_last_complete_json(text)
    assert result["solution"]["House 1"]["Name"] == "Eric"


# ── scoring ───────────────────────────────────────────────────────────────

def test_score_puzzle_fully_correct():
    puzzle = _puzzle()
    output = ('{"reasoning": "...", "solution": {'
              '"House 1": {"Name": "Eric", "Pet": "cat"}, '
              '"House 2": {"Name": "Arnold", "Pet": "dog"}}}')
    result = S.score_puzzle(puzzle, output)
    assert result.solved
    assert result.correct_cells == 4
    assert result.total_cells == 4
    assert result.parsed


def test_score_puzzle_case_insensitive_and_partial():
    puzzle = _puzzle()
    # Name/House1 correct (case-insensitive), Pet/House1 wrong, House2 both correct
    output = ('{"solution": {'
              '"House 1": {"Name": "ERIC", "Pet": "dog"}, '
              '"House 2": {"Name": "Arnold", "Pet": "dog"}}}')
    result = S.score_puzzle(puzzle, output)
    assert not result.solved
    assert result.correct_cells == 3
    assert result.total_cells == 4


def test_score_puzzle_no_json_found():
    puzzle = _puzzle()
    result = S.score_puzzle(puzzle, "I'm not sure how to solve this.")
    assert not result.solved
    assert not result.parsed
    assert result.correct_cells == 0


def test_score_puzzle_missing_solution_key():
    puzzle = _puzzle()
    result = S.score_puzzle(puzzle, '{"reasoning": "..."}')
    assert not result.parsed
    assert not result.solved


def test_score_puzzle_null_cell_values_dont_crash():
    puzzle = _puzzle()
    output = '{"solution": {"House 1": {"Name": null, "Pet": "cat"}, "House 2": {}}}'
    result = S.score_puzzle(puzzle, output)
    assert result.parsed
    assert result.correct_cells == 1  # only Pet/House1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
