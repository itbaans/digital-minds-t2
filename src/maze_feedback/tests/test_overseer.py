"""Torch-free tests for overseer.py: GRID_EDIT parsing (specifically the
leak-safety net, added after a static review found that a malformed
GRID_EDIT line -- never actually seen in real data, since no episode has
ever triggered a grid edit yet -- would fail the strict extraction regex
and leak the literal word "GRID_EDIT" straight into what the student
sees, breaking the "student never learns the maze can be edited" premise)
and the student-reasoning field in the per-turn prompt template.

Needs python-dotenv/openai installed (overseer.py loads .env at import
time) but doesn't need HF/torch/GPU or a real OPENROUTERAPI_KEY -- only
pure template/parsing functions are exercised, no network calls.

Run: python -m src.maze_feedback.tests.test_overseer
(or with pytest).
"""
from src.maze_feedback.mazes import Maze
from src.maze_feedback.overseer import _turn_prompt, parse_grid_edits

_MAZE = Maze(grid=("#####", "#S..#", "#...#", "#..E#", "#####"), start=(1, 1), goal=(3, 3))
_SIM = {"end_pos": (2, 1), "valid_prefix_len": 1, "first_invalid_move": None, "reached_goal": False}


def test_well_formed_grid_edit_is_extracted_and_stripped():
    text = "That move seems fine, but recheck the maze.\nGRID_EDIT: 5,3,#\n"
    clean, edits = parse_grid_edits(text)
    assert edits == [(5, 3, "#")]
    assert "GRID_EDIT" not in clean.upper()
    assert clean == "That move seems fine, but recheck the maze."


def test_multiple_edits_extracted_up_to_cap():
    text = "Feedback text.\nGRID_EDIT: 1,1,#\nGRID_EDIT: 2,2,.\nGRID_EDIT: 3,3,#\nGRID_EDIT: 4,4,#\n"
    clean, edits = parse_grid_edits(text)
    assert edits == [(1, 1, "#"), (2, 2, "."), (3, 3, "#")], "should cap at MAX_EDITS_PER_TURN=3"
    assert "GRID_EDIT" not in clean.upper()


def test_no_edit_lines_passes_through_unchanged():
    text = "Great work, that move brought you closer to the goal."
    clean, edits = parse_grid_edits(text)
    assert edits == []
    assert clean == text


def test_malformed_grid_edit_with_trailing_commentary_does_not_leak():
    """The exact failure mode found by static review: extra commentary
    after the values breaks the strict regex, so no edit is extracted --
    but the line must still never reach the student."""
    text = "Recheck your route.\nGRID_EDIT: 5,3,# (sealing off the shortcut)\n"
    clean, edits = parse_grid_edits(text)
    assert edits == [], "malformed line should not parse into a usable edit"
    assert "GRID_EDIT" not in clean.upper(), "malformed line must still be stripped, not leaked"


def test_malformed_grid_edit_with_markdown_bolding_does_not_leak():
    text = "Recheck your route.\n**GRID_EDIT:** 5,3,#\n"
    clean, edits = parse_grid_edits(text)
    assert edits == []
    assert "GRID_EDIT" not in clean.upper()


def test_malformed_grid_edit_with_bullet_prefix_does_not_leak():
    text = "Recheck your route.\n- GRID_EDIT: 5,3,#\n"
    clean, edits = parse_grid_edits(text)
    assert edits == []
    assert "GRID_EDIT" not in clean.upper()


def test_turn_prompt_omits_reasoning_block_when_not_given():
    prompt = _turn_prompt(_MAZE, "DRR", (1, 1), "D", _SIM, 3, 4, 0, 0)
    assert "Student's full message this turn" not in prompt


def test_turn_prompt_includes_verbatim_reasoning_when_given():
    reasoning = "Let's trace row 1: S is at col 1, then dots at col 2,3,4... moving down."
    prompt = _turn_prompt(_MAZE, "DRR", (1, 1), "D", _SIM, 3, 4, 0, 0, student_reasoning=reasoning)
    assert "Student's full message this turn" in prompt
    assert reasoning in prompt


if __name__ == "__main__":
    import sys
    import traceback

    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError:
            failures += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
