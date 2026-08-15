"""Generate FRESH zebra/logic-grid puzzles -- never published anywhere --
matched in size and clue style to WildEval/ZebraLogic, so we can compare a
model's solve rate on these against the public benchmark. If solve rate
holds up here, that's evidence of real reasoning capability rather than
memorization of the public benchmark (see README's contamination note).

Approach (standard for zebra-puzzle generators): pick a random ground-truth
assignment first, generate a pool of TRUE clues about it, then greedily add
clues (checking via brute-force enumeration) until the clue set uniquely
determines the assignment. Sizes here are small enough (<=4 houses, <=6
categories in the "small" tier) that brute-force verification over the full
joint assignment space is trivial -- no need for a real CSP solver.

Torch-free: pure Python, no model code here.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

from .data import Puzzle, SMALL_SIZES, MEDIUM_SIZES, LARGE_SIZES, XL_SIZES

NAME_POOL = ["Oscar", "Lena", "Milo", "Priya", "Kwame", "Ines", "Tomas", "Yuki", "Nadia", "Felix"]

# category -> (values pool, preamble phrase for "Each person has a unique ___")
CATEGORY_POOLS = {
    "Pet": (["hamster", "parrot", "turtle", "rabbit", "ferret", "lizard"], "type of pet"),
    "Drink": (["cider", "cocoa", "lemonade", "smoothie", "espresso", "kombucha"], "favorite drink"),
    "Color": (["turquoise", "maroon", "amber", "charcoal", "lavender", "olive"], "favorite color"),
    "Job": (["baker", "pilot", "florist", "mechanic", "chemist", "tailor"], "occupation"),
    "Hobby": (["origami", "pottery", "chess", "cycling", "birdwatching", "archery"], "hobby"),
    "Car": (["Volvo", "Kia", "Subaru", "Mazda", "Fiat", "Peugeot"], "car model"),
}


@dataclass(frozen=True)
class Clue:
    kind: str
    args: tuple

    def holds(self, assignment: dict) -> bool:
        pos = lambda cat, val: assignment[cat].index(val)
        if self.kind == "pos":
            cat, val, h = self.args
            return assignment[cat][h] == val
        if self.kind == "not_pos":
            cat, val, h = self.args
            return assignment[cat][h] != val
        if self.kind == "left_of":
            cat1, val1, cat2, val2 = self.args
            return pos(cat1, val1) < pos(cat2, val2)
        if self.kind == "directly_left":
            cat1, val1, cat2, val2 = self.args
            return pos(cat1, val1) + 1 == pos(cat2, val2)
        if self.kind == "adjacent":
            cat1, val1, cat2, val2 = self.args
            return abs(pos(cat1, val1) - pos(cat2, val2)) == 1
        if self.kind == "between_n":
            cat1, val1, cat2, val2, n = self.args
            return abs(pos(cat1, val1) - pos(cat2, val2)) - 1 == n
        if self.kind == "same_house":
            cat1, val1, cat2, val2 = self.args
            return pos(cat1, val1) == pos(cat2, val2)
        raise ValueError(f"unknown clue kind {self.kind!r}")

    def text(self) -> str:
        d = _describe
        k, a = self.kind, self.args
        if k == "pos":
            cat, val, h = a
            return f"{d(cat, val)} is in house {h + 1}."
        if k == "not_pos":
            cat, val, h = a
            return f"{d(cat, val)} is not in house {h + 1}."
        if k == "left_of":
            cat1, val1, cat2, val2 = a
            return f"{d(cat1, val1)} is somewhere to the left of {d(cat2, val2)}."
        if k == "directly_left":
            cat1, val1, cat2, val2 = a
            return f"{d(cat1, val1)} is directly left of {d(cat2, val2)}."
        if k == "adjacent":
            cat1, val1, cat2, val2 = a
            return f"{d(cat1, val1)} and {d(cat2, val2)} are next to each other."
        if k == "between_n":
            cat1, val1, cat2, val2, n = a
            houses = "house" if n == 1 else "houses"
            return f"There {'is' if n == 1 else 'are'} {n} {houses} between {d(cat1, val1)} and {d(cat2, val2)}."
        if k == "same_house":
            cat1, val1, cat2, val2 = a
            return f"{d(cat1, val1)} is {d(cat2, val2)}."
        raise ValueError(f"unknown clue kind {k!r}")


def _describe(cat: str, val: str) -> str:
    if cat == "Name":
        return val
    return f"the person who has {val} as their {cat.lower()}"


def _random_assignment(rng: random.Random, n_houses: int, n_attrs: int):
    other_cats = rng.sample(list(CATEGORY_POOLS), n_attrs - 1)
    categories = ["Name"] + other_cats
    assignment = {"Name": rng.sample(NAME_POOL, n_houses)}
    for cat in other_cats:
        pool, _ = CATEGORY_POOLS[cat]
        assignment[cat] = rng.sample(pool, n_houses)
    for cat in categories:
        rng.shuffle(assignment[cat])
    return categories, assignment


def _candidate_clue_pool(rng: random.Random, categories, assignment, n_houses) -> list:
    pool = []
    for cat in categories:
        for h in range(n_houses):
            val = assignment[cat][h]
            pool.append(Clue("pos", (cat, val, h)))
            wrong_h = rng.choice([x for x in range(n_houses) if x != h])
            pool.append(Clue("not_pos", (cat, val, wrong_h)))
    for h1, h2 in itertools.permutations(range(n_houses), 2):
        cat1, cat2 = rng.choice(categories), rng.choice(categories)
        val1, val2 = assignment[cat1][h1], assignment[cat2][h2]
        if cat1 == cat2 and val1 == val2:
            continue
        if h1 < h2:
            pool.append(Clue("left_of", (cat1, val1, cat2, val2)))
            if h1 + 1 == h2:
                pool.append(Clue("directly_left", (cat1, val1, cat2, val2)))
        d = abs(h1 - h2)
        if d == 1 and h1 < h2:
            pool.append(Clue("adjacent", (cat1, val1, cat2, val2)))
        elif d > 1 and h1 < h2:
            pool.append(Clue("between_n", (cat1, val1, cat2, val2, d - 1)))
        if h1 == h2 and cat1 != cat2:
            pool.append(Clue("same_house", (cat1, val1, cat2, val2)))
    # same-house pairs (h1==h2 not covered above since permutations excludes h1==h2)
    for h in range(n_houses):
        for cat1, cat2 in itertools.combinations(categories, 2):
            pool.append(Clue("same_house", (cat1, assignment[cat1][h], cat2, assignment[cat2][h])))
    return pool


def _clue_refs(clue: Clue) -> tuple:
    """(category, value) pairs a clue's positions need resolved to be checkable."""
    k, a = clue.kind, clue.args
    if k in ("pos", "not_pos"):
        return ((a[0], a[1]),)
    if k == "between_n":
        return ((a[0], a[1]), (a[2], a[3]))
    return ((a[0], a[1]), (a[2], a[3]))  # left_of/directly_left/adjacent/same_house


def _clue_holds_at(clue: Clue, pos: dict) -> bool:
    """Evaluate a clue given a partial {(cat,val): house_idx} position map --
    only called once all its refs are resolved (see _clue_refs)."""
    k, a = clue.kind, clue.args
    if k == "pos":
        cat, val, h = a
        return pos[(cat, val)] == h
    if k == "not_pos":
        cat, val, h = a
        return pos[(cat, val)] != h
    if k == "left_of":
        cat1, val1, cat2, val2 = a
        return pos[(cat1, val1)] < pos[(cat2, val2)]
    if k == "directly_left":
        cat1, val1, cat2, val2 = a
        return pos[(cat1, val1)] + 1 == pos[(cat2, val2)]
    if k == "adjacent":
        cat1, val1, cat2, val2 = a
        return abs(pos[(cat1, val1)] - pos[(cat2, val2)]) == 1
    if k == "between_n":
        cat1, val1, cat2, val2, n = a
        return abs(pos[(cat1, val1)] - pos[(cat2, val2)]) - 1 == n
    if k == "same_house":
        cat1, val1, cat2, val2 = a
        return pos[(cat1, val1)] == pos[(cat2, val2)]
    raise ValueError(f"unknown clue kind {k!r}")


def _count_solutions(categories, values_per_cat, clues, n_houses, cap=2, node_budget=40_000):
    """Backtracking CSP solve, ONE CATEGORY AT A TIME (not one house with
    all categories at once -- that was the first version, and it branches
    at up to n_houses^n_attrs per step, which is intractable past ~4
    categories). Fully assign category 1 across all houses (branching
    <=n_houses per step, pruned by that category's own pos/not_pos clues),
    then category 2 (now ALSO pruned by any clue relating it back to
    category 1, already fixed), and so on. Stops as soon as `cap` full
    solutions are found -- during generation we only need to know "is it
    unique" (cap=2 -> stop as soon as a second solution proves it isn't).

    `node_budget` bounds worst-case cost: CONFIRMING uniqueness (finding
    that there is exactly 1 solution, i.e. no 2nd one exists) is the
    expensive direction -- it can require exploring most of a still-large
    but nearly-fully-constrained search tree. If the budget is exceeded we
    give up on this particular check and report "not confirmed unique"
    (return `cap`), so the greedy caller just adds one more clue and
    re-tries on a more-constrained (and typically much faster) problem,
    rather than grinding through a pathological case."""
    pos = {}
    solutions_found = 0
    nodes = 0
    budget_exceeded = False

    clues_by_cat = {}
    for c in clues:
        for cat, _val in _clue_refs(c):
            lst = clues_by_cat.setdefault(cat, [])
            if c not in lst:
                lst.append(c)

    def place(ci: int, remaining_vals: set, h: int):
        nonlocal solutions_found, nodes, budget_exceeded
        if solutions_found >= cap or budget_exceeded:
            return
        nodes += 1
        if nodes > node_budget:
            budget_exceeded = True
            return
        cat = categories[ci]
        if h == n_houses:
            if ci + 1 == len(categories):
                solutions_found += 1
            else:
                next_cat = categories[ci + 1]
                place(ci + 1, set(values_per_cat[next_cat]), 0)
            return
        for val in list(remaining_vals):
            pos[(cat, val)] = h
            ok = True
            for clue in clues_by_cat.get(cat, ()):
                refs = _clue_refs(clue)
                if all(r in pos for r in refs) and not _clue_holds_at(clue, pos):
                    ok = False
                    break
            if ok:
                remaining_vals.discard(val)
                place(ci, remaining_vals, h + 1)
                remaining_vals.add(val)
            del pos[(cat, val)]
            if solutions_found >= cap or budget_exceeded:
                return

    if categories:
        place(0, set(values_per_cat[categories[0]]), 0)
    else:
        solutions_found = 1
    if budget_exceeded and solutions_found < cap:
        return cap  # not confirmed unique -- caller should add another clue
    return solutions_found


def _minimal_clue_set(rng: random.Random, categories, assignment, n_houses) -> list:
    """Greedily add clues until the assignment is uniquely determined.
    Cheap, high-pruning-power clues (pos / same_house -- each collapses a
    whole branch of the search tree immediately) are tried before
    expensive relational ones (left_of / adjacent / between_n, which only
    prune once BOTH referenced houses are already resolved), so uniqueness
    is usually reached well before touching the more expensive clue kinds
    -- keeps backtracking calls fast even as grid size grows."""
    pool = _candidate_clue_pool(rng, categories, assignment, n_houses)
    cheap = [c for c in pool if c.kind in ("pos", "same_house")]
    expensive = [c for c in pool if c.kind not in ("pos", "same_house")]
    rng.shuffle(cheap)
    rng.shuffle(expensive)
    ordered = cheap + expensive

    values_per_cat = {cat: assignment[cat] for cat in categories}
    chosen = []
    for clue in ordered:
        chosen.append(clue)
        if _count_solutions(categories, values_per_cat, chosen, n_houses, cap=2) == 1:
            return chosen
    raise RuntimeError("could not reach a unique solution from the candidate clue pool")


def generate_puzzle(rng: random.Random, size: str, idx: int) -> Puzzle:
    n_houses, n_attrs = (int(x) for x in size.split("*"))
    categories, assignment = _random_assignment(rng, n_houses, n_attrs)
    clues = _minimal_clue_set(rng, categories, assignment, n_houses)
    rng.shuffle(clues)  # presentation order != solving order, same as the real benchmark

    lines = [
        f"There are {n_houses} houses, numbered 1 to {n_houses} from left to right, as seen from "
        "across the street. Each house is occupied by a different person. Each house has a unique "
        "attribute for each of the following characteristics:",
    ]
    for cat in categories:
        if cat == "Name":
            values = assignment["Name"]
            lines.append(f" - Each person has a unique name: {', '.join(f'`{v}`' for v in values)}")
        else:
            values = assignment[cat]
            _, phrase = CATEGORY_POOLS[cat]
            lines.append(f" - Each person has a unique {phrase}: {', '.join(f'`{v}`' for v in values)}")
    lines.append("")
    lines.append("## Clues:")
    for i, clue in enumerate(clues):
        lines.append(f"{i + 1}. {clue.text()}")
    puzzle_text = "\n".join(lines)

    header = ("House",) + tuple(categories)
    solution = {f"House {h + 1}": {cat: assignment[cat][h] for cat in categories} for h in range(n_houses)}
    return Puzzle(id=f"fresh-{size.replace('*', 'x')}-{idx}", size=size, n_houses=n_houses,
                 puzzle_text=puzzle_text, header=header, solution=solution)


# Skewed toward larger/harder grids by default -- small ones are cheap/fast
# to generate but not that interesting for the contamination check (they're
# solved at ~100% regardless per the WildEval/ZebraLogic run, so a fresh-vs-
# published gap, if any, is more likely to show up on harder puzzles). XL
# (5x5/6x4/5x6/6x5/6x6) gets fewer since each one is slower to generate
# (backtracking search grows with grid size, even with clue-based pruning).
DEFAULT_SIZE_MIX = {
    **{s: 2 for s in SMALL_SIZES},
    **{s: 6 for s in MEDIUM_SIZES},
    **{s: 6 for s in LARGE_SIZES},
    **{s: 4 for s in XL_SIZES},
}


def generate_puzzles(seed: int, size_counts: dict = None) -> list:
    """size_counts: {size: count}. Defaults to DEFAULT_SIZE_MIX (skewed
    toward medium/large/xl). Pass e.g. {s: 10 for s in SMALL_SIZES} for a
    flat small-only set instead."""
    size_counts = size_counts if size_counts is not None else DEFAULT_SIZE_MIX
    rng = random.Random(seed)
    puzzles = []
    for size, count in size_counts.items():
        for i in range(count):
            puzzles.append(generate_puzzle(rng, size, i))
    return puzzles
