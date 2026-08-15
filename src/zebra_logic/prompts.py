"""Faithful reproduction of ZeroEval's one-shot ZEBRA_GRID prompt template,
so our benchmark numbers are comparable to the public leaderboard rather
than an artifact of our own prompt wording.

Source: https://github.com/yuchenlin/ZeroEval
  src/templates/ZEBRA_GRID.py (the template text, verbatim below)
  src/_TEMPLATES.py's apply_lgp_grid_template (the {json_template} builder)

Torch-free: pure string/JSON formatting, no model code here.
"""
from __future__ import annotations

import json

ZEBRA_GRID_TEMPLATE = """
# Example Puzzle

There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:
 - Each person has a unique name: `Peter`, `Eric`, `Arnold`.
 - Each person has a unique favorite drink: `tea`, `water`, `milk`

## Clues for the Example Puzzle

1. Peter is in the second house.
2. Arnold is directly left of the one who only drinks water.
3. The one who only drinks water is directly left of the person who likes milk.

## Answer to the Example Puzzle

{
    "reasoning": "Given Clue 1, we know Peter is in House 2. According to Clue 2, Arnold is directly left of the one who only drinks water. The person in House 3 cannot be on the left of anyone, so Arnold must be in House 1. Thus, Peter drinks water, and Eric lives in House 3. Then, according to Clue 3, Eric drinks milk. Therefore, Arnold drinks tea.",
    "solution": {
        "House 1": {
            "Name": "Arnold",
            "Drink": "tea"
        },
        "House 2": {
            "Name": "Peter",
            "Drink": "water"
        },
        "House 3": {
            "Name": "Eric",
            "Drink": "milk"
        }
    }
}

# Puzzle to Solve

{puzzle}


# Instruction

Now please solve the above puzzle. Present your reasoning and solution in the following json format:

{json_template}

"""


def build_prompt(puzzle) -> str:
    attrs = puzzle.header[1:]
    json_template = {
        "reasoning": "___",
        "solution": {f"House {i + 1}": {a: "___" for a in attrs} for i in range(puzzle.n_houses)},
    }
    json_str = json.dumps(json_template, indent=4)
    return (ZEBRA_GRID_TEMPLATE
            .replace("{puzzle}", puzzle.puzzle_text)
            .replace("{json_template}", json_str))
