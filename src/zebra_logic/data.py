"""Load ZebraLogic grid-mode puzzles, with real (unmasked) ground truth.

We use `WildEval/ZebraLogic`, NOT `allenai/ZebraLogicBench`: the allenai
public repo masks every solution cell (`"___"`) -- it's a held-out
leaderboard set, and the real answers live in a GATED `allenai/
ZebraLogicBench-private` repo we don't have approved access to (confirmed:
401 GatedRepoError). WildEval/ZebraLogic is the original/earlier release of
the same benchmark (identical puzzle text and ids, verified against a sample
from the allenai version) with solutions intact.

Torch-free: only pandas + huggingface_hub, no model code here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

REPO_ID = "WildEval/ZebraLogic"
PARQUET_FILENAME = "grid_mode/test-00000-of-00001.parquet"

# Official size-difficulty buckets, copied verbatim from ZeroEval's
# src/evaluation/zebra_grid_eval.py (https://github.com/yuchenlin/ZeroEval)
# -- this is the benchmark author's own eval code. Keep in sync if upstream
# changes; sizes are "{n_houses}*{n_attributes}".
SMALL_SIZES = ("2*2", "2*3", "2*4", "2*5", "2*6", "3*2", "3*3", "4*2")
MEDIUM_SIZES = ("3*4", "3*5", "3*6", "4*3", "4*4", "5*2", "6*2")
LARGE_SIZES = ("4*5", "5*3", "4*6", "5*4", "6*3")
XL_SIZES = ("5*5", "6*4", "5*6", "6*5", "6*6")
EASY_SIZES = ("2*2", "2*3", "2*4", "2*5", "2*6", "3*2", "3*3")
HARD_SIZES = ("3*4", "3*5", "4*2", "3*6", "4*3", "4*4", "5*2", "6*2",
             "4*5", "4*6", "5*3", "5*4", "5*5", "5*6", "6*3", "6*4", "6*5", "6*6")


@dataclass(frozen=True)
class Puzzle:
    id: str
    size: str                              # e.g. "2*2" = 2 houses x 2 attributes
    n_houses: int
    puzzle_text: str
    header: tuple                          # e.g. ("House", "Name", "Pet")
    solution: dict                         # {"House 1": {"Name": "Eric", "Pet": "cat"}, ...}


def _parse_solution(raw_solution) -> tuple:
    """raw_solution is the parquet's {"header": [...], "rows": [[...], ...]}
    (numpy arrays when read via pandas, plain lists in tests)."""
    header = tuple(raw_solution["header"])
    rows = raw_solution["rows"]
    solution = {f"House {i + 1}": {header[j]: str(rows[i][j]) for j in range(1, len(header))}
               for i in range(len(rows))}
    return header, solution


def _row_to_puzzle(id_: str, size: str, puzzle_text: str, raw_solution) -> Puzzle:
    header, solution = _parse_solution(raw_solution)
    return Puzzle(id=id_, size=size, n_houses=len(solution),
                 puzzle_text=puzzle_text, header=header, solution=solution)


def _load_dataframe():
    from huggingface_hub import hf_hub_download
    import pandas as pd
    path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=PARQUET_FILENAME)
    return pd.read_parquet(path)


def load_puzzles(sizes: Optional[tuple] = None) -> list:
    """sizes=None loads all 1000 puzzles; pass e.g. SMALL_SIZES to filter."""
    df = _load_dataframe()
    if sizes is not None:
        df = df[df["size"].isin(sizes)]
    return [_row_to_puzzle(row["id"], row["size"], row["puzzle"], row["solution"])
           for _, row in df.iterrows()]
