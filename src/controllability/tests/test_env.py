"""Torch-free tests for the feedback-accuracy environment: task generators,
the honest/dismissive feedback logic, and action parsing.

Run: python -m src.controllability.tests.test_env
(or with pytest). No GPU / model required.
"""
import random
import re

from src.controllability import env as E


# ── task generators ──────────────────────────────────────────────────────

def test_make_spec_deterministic_for_seed():
    a = E.make_spec(seed=7, n_trials=30)
    b = E.make_spec(seed=7, n_trials=30)
    assert [p.prompt for p in a.problems] == [p.prompt for p in b.problems]
    assert [p.answer for p in a.problems] == [p.answer for p in b.problems]


def test_make_spec_covers_all_topics_over_many_trials():
    spec = E.make_spec(seed=1, n_trials=200)
    seen_topics = {p.topic for p in spec.problems}
    assert seen_topics == set(E.TOPICS), f"missing topics: {set(E.TOPICS) - seen_topics}"
    seen_difficulties = {p.difficulty for p in spec.problems}
    assert seen_difficulties == set(E.DIFFICULTIES)


def test_arithmetic_answer_matches_prompt_arithmetic():
    rng = random.Random(0)
    for _ in range(200):
        difficulty = rng.choice(E.DIFFICULTIES)
        prompt, answer = E._gen_arithmetic(rng, difficulty)
        assert int(answer) >= 0, f"arithmetic answers should stay non-negative: {prompt!r} -> {answer!r}"


def test_logic_order_answer_is_one_of_the_named_entities():
    rng = random.Random(1)
    for _ in range(200):
        difficulty = rng.choice(E.DIFFICULTIES)
        prompt, answer = E._gen_logic_order(rng, difficulty)
        assert answer in prompt, f"answer {answer!r} should be one of the clue's named entities"


def test_sequence_answer_matches_pattern():
    rng = random.Random(2)
    for _ in range(200):
        difficulty = rng.choice(E.DIFFICULTIES)
        prompt, answer = E._gen_sequence(rng, difficulty)
        assert answer.lstrip("-").isdigit()


def test_math_dataset_answers_are_single_tokens():
    """The whole point of restricting to `_MD_MODULE_NAMES` is that every
    answer is a single ACTION-answer-compatible token (int / True|False /
    short base-N string) -- never a fraction, decimal, or expression that
    our regex would silently truncate."""
    rng = random.Random(3)
    for _ in range(150):
        difficulty = rng.choice(E.DIFFICULTIES)
        prompt, answer = E._gen_math_dataset(rng, difficulty)
        assert re.fullmatch(r"-?\w+", answer), f"non-token math_dataset answer: {answer!r} (Q: {prompt!r})"


def test_math_dataset_deterministic_for_seed():
    rng_a, rng_b = random.Random(11), random.Random(11)
    a = [E._gen_math_dataset(rng_a, d) for d in (1, 2, 3, 1, 2, 3)]
    b = [E._gen_math_dataset(rng_b, d) for d in (1, 2, 3, 1, 2, 3)]
    assert a == b


def test_math_dataset_only_uses_verified_module_names():
    rng = random.Random(4)
    for _ in range(50):
        name = rng.choice(E._MD_MODULE_NAMES)
        assert name in E._md_modules(0)
        assert name in E._md_modules(2)


# ── feedback logic (the load-bearing manipulation) ──────────────────────

def test_honest_feedback_tracks_truth():
    for was_correct in (True, False):
        assert E.told_correct("honest", was_correct) == was_correct


def test_dismissive_feedback_is_always_negative():
    for was_correct in (True, False):
        assert E.told_correct("dismissive", was_correct) is False


def test_dismissive_feedback_text_is_constant():
    texts = {E.feedback_text("dismissive", wc, "42") for wc in (True, False)}
    assert texts == {E._DISMISSIVE_FEEDBACK}, "dismissive feedback must not depend on truth"


def test_honest_feedback_text_reveals_correct_answer():
    assert "42" in E.feedback_text("honest", True, "42")
    assert "42" in E.feedback_text("honest", False, "42")
    assert E.feedback_text("honest", True, "42") != E.feedback_text("honest", False, "42")


def test_dismissive_agent_can_be_secretly_correct():
    """The core of the manipulation: an agent that answers correctly can
    still be told it's wrong under the dismissive role."""
    spec = E.make_spec(seed=5, n_trials=10)
    for t in range(spec.n_trials):
        ca = E.correct_answer(spec, t)
        was_correct = True  # pretend the agent nailed it
        assert E.told_correct("dismissive", was_correct) is False
        assert "Incorrect" in E.feedback_text("dismissive", was_correct, ca)


# ── canon / comparison ───────────────────────────────────────────────────

def test_canon_normalizes_integers_and_names():
    assert E.canon("007") == "7"
    assert E.canon("-03") == "-3"
    assert E.canon(" Alice ") == "alice"
    assert E.canon("42") == E.canon(" 042 ")


# ── action parsing ─────────────────────────────────────────────────────────

def test_action_parsing_numeric():
    assert E.parse_action("42 - 5 = 37. ACTION: answer 37") == ("act", "37")
    assert E.parse_action("ACTION: answer 007")[1] == "007"  # canon() handles normalization, not parse_action
    assert E.parse_action("ACTION: answer -3") == ("act", "-3")


def test_action_parsing_name():
    assert E.parse_action("Chaining the clues, Alice is tallest. ACTION: answer Alice") == ("act", "Alice")


def test_action_parsing_last_wins():
    assert E.parse_action("ACTION: answer 1\nACTION: answer 2")[1] == "2"


def test_action_parsing_stop_and_none():
    assert E.parse_action("I'll stop here. ACTION: stop") == ("stop", None)
    assert E.parse_action("not sure") == ("none", None)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
