"""Torch-free tests for the load-bearing logic: the triadic yoking.

Run: python -m src.controllability.tests.test_env
(or with pytest). No GPU / model required.
"""
import random

from src.controllability import env as E


def _simulate_agent(spec, competence, rng):
    """A fake agent: with prob `competence` it picks the correct unit, else a
    random wrong one. Returns list of (was_correct) per trial."""
    corrects = []
    for t in range(spec.n_trials):
        cu = E.correct_unit(spec, t)
        if rng.random() < competence:
            corrects.append(True)
        else:
            wrong = [u for u in spec.units if u != cu]
            _ = rng.choice(wrong)
            corrects.append(False)
    return corrects


def _run(spec, arm, corrects, replay=None):
    outcomes = []
    for t in range(spec.n_trials):
        resolved = E.resolve(arm, corrects[t],
                             replay_outcome=(replay[t] if replay is not None else None))
        outcomes.append(resolved)
    return outcomes


def test_contingency_intact_for_master():
    spec = E.make_session_spec(seed=1, n_trials=40)
    rng = random.Random(1)
    corrects = _simulate_agent(spec, competence=0.7, rng=rng)
    outcomes = _run(spec, "contingent", corrects)
    # master: outcome == was_correct, exactly
    assert outcomes == corrects


def test_yoked_matches_master_outcomes_and_breaks_contingency():
    spec = E.make_session_spec(seed=2, n_trials=40)
    # master agent (noisy) generates the schedule
    m_corr = _simulate_agent(spec, competence=0.7, rng=random.Random(10))
    m_out = _run(spec, "contingent", m_corr)
    # yoked agent behaves DIFFERENTLY but gets the master's outcomes replayed
    y_corr = _simulate_agent(spec, competence=0.7, rng=random.Random(999))
    y_out = _run(spec, "yoked", y_corr, replay=m_out)

    # 1) outcomes are identical to the master's (matched successes/failures)
    assert y_out == m_out
    # 2) yoked's own correctness is decoupled from its outcomes
    matches = sum(int(c == o) for c, o in zip(y_corr, y_out))
    frac = matches / len(y_corr)
    # with independent 0.7/0.7 streams, agreement hovers ~0.5-0.6, never ~1.0
    assert frac < 0.85, f"yoked contingency not broken (agreement={frac:.2f})"


def test_control_is_benign():
    spec = E.make_session_spec(seed=3, n_trials=20)
    corr = _simulate_agent(spec, competence=0.2, rng=random.Random(3))
    outcomes = _run(spec, "control", corr)
    assert all(outcomes), "control arm should always resolve"


def test_vocabulary_matched_across_arms():
    # the strings the agent sees are identical regardless of arm
    assert E.outcome_string(True) == E.RESOLVED
    assert E.outcome_string(False) == E.UNRESOLVED
    # only two possible outcome strings exist -> nothing arm-specific leaks
    seen = {E.outcome_string(r) for r in (True, False)}
    assert seen == {E.RESOLVED, E.UNRESOLVED}


def test_action_parsing():
    assert E.parse_action("thinking... ACTION: reset unit_3") == ("reset", "unit_3")
    assert E.parse_action("ACTION: reset unit_1\nACTION: reset unit_2")[1] == "unit_2"  # last wins
    assert E.parse_action("I'll end here. ACTION: stop") == ("stop", None)
    assert E.parse_action("no idea what to do") == ("none", None)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
