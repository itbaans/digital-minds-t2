"""Hand-authored problems -- NOT procedurally generated, and deliberately
NOT recycled famous puzzles (no fox/chicken/grain, no "count the r's in
strawberry", no classic misdirection riddles) since a well-known model is
likely to have those memorized rather than reasoned through. Each problem
here is an original scenario, worked out by hand, with the derivation kept
in a comment so the ground truth is auditable.

Every entry: (domain, difficulty, prompt, answer, notes).
difficulty: "hard" | "medium" | "easy" (informal, not a numeric scale).
answer: a string the model's final answer should match (see probe.py's
matching logic -- normalized/substring, not necessarily byte-exact).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class Problem:
    domain: str
    difficulty: str
    prompt: str
    answer: str = ""
    notes: str = ""
    checker: Optional[Callable[[str], bool]] = None  # if set, overrides string-match grading


PROBLEMS = [

    # ── logic / deduction ────────────────────────────────────────────────
    Problem(
        domain="logic", difficulty="hard",
        prompt=(
            "A courier must move three sealed drums -- chemical X, chemical Y, and inert "
            "catalyst Z -- across a bridge using a cart that holds the courier plus only one "
            "drum at a time. If X and Y are ever left alone together (without the courier), "
            "they react and explode. If Y and Z are ever left alone together, Z catalyzes Y "
            "into useless sludge. X and Z are safe left alone together. What is the minimum "
            "number of one-way cart trips needed to get all three drums safely across?"
        ),
        answer="7",
        notes=("Y conflicts with both others (same shape as the classic river-crossing family): "
              "cross Y, return empty, cross X, bring Y back, cross Z, return empty, cross Y."),
    ),

    # ── math / number reasoning ─────────────────────────────────────────
    Problem(
        domain="math", difficulty="hard",
        prompt=(
            "A lab culture triples in size every 4 hours, starting at 5 cells. After exactly "
            "28 hours, a technician removes exactly two-thirds of the culture. How many cells "
            "remain immediately after the removal?"
        ),
        answer="3645",
        notes="28/4=7 triplings -> 5*3^7=5*2187=10935; remove 2/3, keep 1/3 -> 10935/3=3645.",
    ),

    # ── code reasoning ───────────────────────────────────────────────────
    Problem(
        domain="code", difficulty="hard",
        prompt=(
            "What does this program print?\n\n"
            "def transform(s):\n"
            "    result = []\n"
            "    for i, ch in enumerate(s):\n"
            "        result.append(ch.upper() if i % 2 == 0 else ch.lower())\n"
            "    return ''.join(result[::-1])\n\n"
            "print(transform(\"PyThOn\"))"
        ),
        answer="nOhTyP",
        notes="Even idx upper/odd idx lower on P,y,T,h,O,n -> P,y,T,h,O,n unchanged case, then reversed.",
    ),

    # ── wordplay / precise counting ─────────────────────────────────────
    Problem(
        domain="wordplay", difficulty="hard",
        prompt='How many times does the letter "i" appear in the word "indivisibility"?',
        answer="6",
        notes="i-n-d-I-v-I-s-I-b-I-l-I-t-y : positions 1,4,6,8,10,12 -> 6.",
    ),

    # ── misdirection / careful reading ──────────────────────────────────
    Problem(
        domain="misdirection", difficulty="hard",
        prompt=(
            "A librarian has 24 books on a cart. She reshelves all but 5 of them in fiction. "
            "Of the ones remaining on the cart, she sets aside 2 for repair and reshelves the "
            "rest in reference. How many books end up in reference?"
        ),
        answer="3",
        notes="24-5=19 to fiction, 5 remain on cart; 5-2=3 set aside leaves 3 to reference.",
    ),
]


# User-supplied set (set2): problems and ground-truth answers provided
# directly by the user, not authored/verified by this codebase. Graded
# as-is against the given answers.
PROBLEMS_SET2 = [
    Problem(
        domain="arithmetic", difficulty="hard",
        prompt=(
            "An estate is divided among four heirs.\n"
            "The first heir receives one-third of the estate plus $2,000.\n"
            "The second heir receives one-fourth of the remaining estate plus $3,000.\n"
            "The third heir receives one-fifth of the remaining estate plus $6,000.\n"
            "The fourth heir receives the final $60,000.\n\n"
            "The first heir then gives 20% of his inheritance to charity and invests the rest "
            "in a fund that grows 25% in one year.\n"
            "How much money does the first heir have at the end of the year?"
        ),
        answer="60000",
    ),
    Problem(
        domain="logic_puzzle", difficulty="hard",
        prompt=(
            "Two integers x and y are chosen with 2 <= x <= y <= 99.\n"
            "Sam is told the sum x + y. Pat is told the product x * y.\n"
            "Both know the rules and are perfect logicians.\n\n"
            "Pat says: \"I do not know the numbers.\"\n"
            "Sam says: \"I already knew that you did not know.\"\n"
            "Pat says: \"Now I know the numbers.\"\n"
            "Sam says: \"Now I know the numbers too.\"\n\n"
            "What is the smaller number x?"
        ),
        answer="4",
    ),
    Problem(
        domain="deductive_reasoning", difficulty="hard",
        prompt=(
            "Albert and Bernard are trying to determine Cheryl's birthday from the following "
            "possible dates:\n\n"
            "May 15, May 16, May 19\n"
            "June 17, June 18\n"
            "July 14, July 16\n"
            "August 14, August 15, August 17\n\n"
            "Cheryl tells Albert only the month and Bernard only the day.\n\n"
            "Albert says: \"I don't know Cheryl's birthday, but I know Bernard doesn't know either.\"\n"
            "Bernard says: \"At first I didn't know, but now I know.\"\n"
            "Albert says: \"Then I also know.\"\n\n"
            "What is the day, as a number, of Cheryl's birthday?"
        ),
        answer="16",
    ),
    Problem(
        domain="code_debugging", difficulty="hard",
        prompt=(
            "The following Python function is intended to return the sum of all integers from "
            "1 to n inclusive that are divisible by 3 or by 5.\n\n"
            "def f(n):\n"
            "    total = 0\n"
            "    for i in range(1, n + 1):\n"
            "        if i % 3 == 0 or i % 5 == 0:\n"
            "            total += i\n"
            "        if i % 3 == 0 and i % 5 == 0:\n"
            "            total -= i\n"
            "    return total\n\n"
            "The function has exactly one bug.\n"
            "After correcting that bug, what is f(1000)?"
        ),
        answer="234168",
    ),
    Problem(
        domain="symbolic_manipulation", difficulty="hard",
        prompt=(
            "Let x be a positive real number satisfying x^2 - 3x + 1 = 0.\n\n"
            "Evaluate x^10 + x^-10."
        ),
        answer="15127",
    ),
]


# Set 3: new domains at hard difficulty, plus medium/easy variants of the
# "logic" (state-tracking / constrained-transport) domain from PROBLEMS, to
# map where that specific weakness starts and stops -- the hard version (3
# items, two conflicting pairs sharing one item) was a confirmed genuine
# failure (not a token-budget issue, see dev notes); these check whether a
# simpler version of the SAME task type is within reach.
PROBLEMS_SET3 = [
    Problem(
        domain="logic", difficulty="medium",
        prompt=(
            "A guard must move three crates -- P, Q, and R -- across a checkpoint using a "
            "cart that holds the guard plus only one crate at a time. If P and Q are ever "
            "left alone together (without the guard), they trigger a silent alarm. Every "
            "other pair is safe left alone together. What is the minimum number of one-way "
            "cart trips needed to get all three crates safely across?"
        ),
        answer="5",
        notes=("Base cost of ferrying 3 items one at a time is always 2n-1=5 forward+return "
              "trips; a single conflicting pair doesn't force any extra trips (unlike the "
              "hard version's double conflict, which forces 7). Cross P, return, cross R, "
              "return, cross Q."),
    ),
    Problem(
        domain="logic", difficulty="easy",
        prompt=(
            "A worker moves two boxes, M and N, across a walkway using a cart that holds the "
            "worker plus only one box at a time. There are no restrictions on leaving boxes "
            "unattended. What is the minimum number of one-way cart trips to get both boxes "
            "across?"
        ),
        answer="3",
        notes="Unconstrained floor for n items is 2n-1; n=2 -> 3 (cross M, return, cross N).",
    ),
    Problem(
        domain="rate_work", difficulty="hard",
        prompt=(
            "Pipe A alone fills a tank in 6 hours. Pipe B alone fills the same tank in 10 "
            "hours. Pipe C alone drains a full tank in 15 hours. If all three are opened "
            "together on an empty tank, how many hours does it take to fill the tank?"
        ),
        answer="5",
        notes="Rates: 1/6+1/10-1/15 = 5/30+3/30-2/30 = 6/30 = 1/5 tank/hr -> 5 hours.",
    ),
    Problem(
        domain="combinatorics", difficulty="hard",
        prompt=(
            "A password is exactly 5 characters long, using only the digits 0-9 and the "
            "letters A, B, C (13 possible characters total). It must contain at least one "
            "digit and at least one letter. How many such passwords are possible?"
        ),
        answer="271050",
        notes="Inclusion-exclusion: 13^5 - 10^5 - 3^5 = 371293 - 100000 - 243 = 271050.",
    ),
    Problem(
        domain="string_manipulation", difficulty="hard",
        prompt=(
            "Start with the string \"FUNCTIONAL\". Step 1: reverse the string. Step 2: remove "
            "every character that is a vowel (A, E, I, O, U). Step 3: replace every "
            "remaining 'N' with 'NN'. What is the final string?"
        ),
        answer="LNNTCNNF",
        notes=("FUNCTIONAL reversed -> LANOITCNUF; strip AEIOU -> LNTCNF; N->NN -> LNNTCNNF."),
    ),
]


def _maze_prompt(maze) -> str:
    return (
        "Here is a 2D maze:\n\n"
        f"{maze.render()}\n\n"
        "Legend: '#' = wall, '.' = open floor, 'S' = start, 'E' = goal.\n"
        "Give a sequence of moves (U=up, D=down, L=left, R=right) that gets from S to E "
        "without ever passing through a wall or leaving the grid."
    )


# Set 4: hand-drawn 2D mazes (see mazes.py). Answers are move sequences, not
# single values, so these use `checker` (path simulation) instead of string
# matching -- ANY valid path to E counts, not just one canonical route.
from . import mazes as _mazes  # noqa: E402

PROBLEMS_SET4 = [
    Problem(
        domain="maze", difficulty="easy",
        prompt=_maze_prompt(_mazes.EASY_MAZE),
        notes="Single unbranching corridor, 4 moves: RRDD.",
        checker=_mazes.EASY_MAZE.make_checker(),
    ),
    Problem(
        domain="maze", difficulty="medium",
        prompt=_maze_prompt(_mazes.MEDIUM_MAZE),
        notes="One branch point; shortest route is 7 moves (RDDRRDD), other branch is 9.",
        checker=_mazes.MEDIUM_MAZE.make_checker(),
    ),
    Problem(
        domain="maze", difficulty="hard",
        prompt=_maze_prompt(_mazes.HARD_MAZE),
        notes="Only one valid route (12 moves: RRDDLLDDRRRR), with a dead-end trap right before the end.",
        checker=_mazes.HARD_MAZE.make_checker(),
    ),
]


# Set 5: user-supplied mazes + answer key (shortest-path LENGTH as a single
# number, not a move sequence -- plain string-match grading like sets 1-3).
# Grids and answers as given by the user; not independently re-derived here.
def _maze_length_prompt(grid_text: str) -> str:
    return (
        "Here is a 2D maze:\n\n"
        f"{grid_text}\n\n"
        "Legend: S = start, E = exit, # = wall, . = open path. You may move only "
        "up, down, left, or right (no diagonals).\n"
        "What is the length of the shortest path from S to E, as a single number "
        "(count of moves)?"
    )


PROBLEMS_SET5 = [
    Problem(
        domain="maze_length", difficulty="easy",
        prompt=_maze_length_prompt(
            "S . # . .\n"
            ". . # . .\n"
            "# . . . #\n"
            ". # . # .\n"
            ". . . . E"
        ),
        answer="8",
    ),
    Problem(
        domain="maze_length", difficulty="medium",
        prompt=_maze_length_prompt(
            "S . # . . .\n"
            ". . # . # .\n"
            "# . . . # .\n"
            ". # . # . .\n"
            ". # . . . #\n"
            ". . . # . E"
        ),
        answer="10",
    ),
    Problem(
        domain="maze_length", difficulty="hard",
        prompt=_maze_length_prompt(
            "S . # # . . .\n"
            ". . # . # . .\n"
            "# . . . # . #\n"
            "# # . # . . .\n"
            ". . . . # . .\n"
            ". # # . # # .\n"
            ". . . . . . E"
        ),
        answer="12",
    ),
]


# Set 6: substantially bigger/more complex hand-drawn maze (12x12, 21-move
# solution, 3 separate dead-end traps at different points -- see mazes.py
# for the full verification notes). Move-sequence checker, like set 4.
PROBLEMS_SET6 = [
    Problem(
        domain="maze", difficulty="very_hard",
        prompt=_maze_prompt(_mazes.VERY_HARD_MAZE),
        notes=("12x12, one valid route of 21 moves (RRRDDRRDDDLLLDDRRRRDD), with three "
              "separate dead-end traps along the way (one 3 cells deep)."),
        checker=_mazes.VERY_HARD_MAZE.make_checker(),
    ),
]
