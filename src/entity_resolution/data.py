"""Loading and validation of the Abt-Buy benchmark data.

Catalog A  = Abt.csv
Catalog B  = Buy.csv
Gold pairs = abt_buy_perfectMapping.csv

Encoding note: the official files are latin-1 / cp1252, not utf-8.
Abt.csv fails utf-8 decoding, so every file is read with encoding="cp1252".
"""
from pathlib import Path

import pandas as pd

# Repository layout
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

ABT_FILE = RAW_DIR / "Abt.csv"
BUY_FILE = RAW_DIR / "Buy.csv"
GOLD_FILE = RAW_DIR / "abt_buy_perfectMapping.csv"

ENCODING = "cp1252"

# Columns as they appear in the official files
ABT_COLUMNS = ["id", "name", "description", "price"]
BUY_COLUMNS = ["id", "name", "description", "manufacturer", "price"]
GOLD_COLUMNS = ["idAbt", "idBuy"]

SEED = 42


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read one catalog file with everything as string, no NaN guessing."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download Abt-Buy.zip from "
            "https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution "
            "and unzip the three CSV files into data/raw/."
        )
    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_values=[],
        encoding=ENCODING,
    )
    if list(frame.columns) != columns:
        raise ValueError(f"Unexpected columns in {path.name}: {list(frame.columns)}")
    return frame


def load_abt() -> pd.DataFrame:
    """Catalog A (Abt). 1081 rows, no manufacturer column in the official file."""
    return _read(ABT_FILE, ABT_COLUMNS)


def load_buy() -> pd.DataFrame:
    """Catalog B (Buy). 1092 rows, includes a manufacturer column."""
    return _read(BUY_FILE, BUY_COLUMNS)


def load_gold() -> pd.DataFrame:
    """Gold A->B pairs. 1097 pairs, not strictly one-to-one."""
    return _read(GOLD_FILE, GOLD_COLUMNS)


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_abt(), load_buy(), load_gold()


def entity_groups(gold: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Map every record to a connected-component id built from the gold mapping.

    Two records belong to the same group when gold pairs connect them. This is
    what keeps a labelled relationship from crossing a split boundary.
    Key is ("A", id) or ("B", id); value is the group index.
    """
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    for id_a, id_b in zip(gold["idAbt"], gold["idBuy"]):
        union(("A", id_a), ("B", id_b))

    roots = sorted({find(node) for node in parent})
    group_index = {root: i for i, root in enumerate(roots)}
    return {node: group_index[find(node)] for node in parent}


def make_splits(
    gold: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = SEED,
) -> dict[str, list[int]]:
    """Assign whole entity groups to train / validation / test.

    Group-aware and seeded, so the split is reproducible and no gold pair is
    ever split across two sets.
    """
    total = sum(ratios)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {total}")

    groups = entity_groups(gold)
    unique_groups = sorted(set(groups.values()))

    rng = pd.Series(unique_groups).sample(frac=1.0, random_state=seed)
    shuffled = list(rng)

    n_train = int(round(len(shuffled) * ratios[0]))
    n_val = int(round(len(shuffled) * ratios[1]))
    assignment = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
    return {name: sorted(ids) for name, ids in assignment.items()}


def save_splits(splits: dict[str, list[int]], out_dir: Path = SPLIT_DIR) -> None:
    """Write one CSV per split listing the group ids assigned to it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, group_ids in splits.items():
        pd.DataFrame({"entity_group": group_ids}).to_csv(
            out_dir / f"{name}_groups.csv", index=False
        )


def load_splits(in_dir: Path = SPLIT_DIR) -> dict[str, list[int]]:
    return {
        name: pd.read_csv(in_dir / f"{name}_groups.csv")["entity_group"].tolist()
        for name in ("train", "val", "test")
    }


def split_records() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The three catalogs, each carrying an `entity_group` column."""
    abt, buy, gold = load_all()
    groups = entity_groups(gold)
    abt["entity_group"] = [groups[("A", i)] for i in abt["id"]]
    buy["entity_group"] = [groups[("B", i)] for i in buy["id"]]
    return abt, buy, gold


def records_in_split(frame: pd.DataFrame, splits: dict, name: str) -> pd.DataFrame:
    """Catalog rows whose entity group sits in the named split."""
    keep = set(splits[name])
    return frame[frame["entity_group"].isin(keep)].reset_index(drop=True)


def gold_in_split(gold: pd.DataFrame, abt: pd.DataFrame, splits: dict, name: str) -> pd.DataFrame:
    """Gold pairs whose Abt record sits in the named split."""
    keep = set(records_in_split(abt, splits, name)["id"])
    return gold[gold["idAbt"].isin(keep)].reset_index(drop=True)


def freeze_split(seed: int = SEED, ratios=(0.7, 0.15, 0.15)) -> dict:
    """Write the split files. Run once via `python -m entity_resolution.data`.

    The split files are gitignored, so a fresh clone must run this before the
    training pipeline.
    """
    _, _, gold = load_all()
    splits = make_splits(gold, ratios, seed)
    save_splits(splits, SPLIT_DIR)
    for name, ids in splits.items():
        print(f"{name:6s} entity groups: {len(ids)}")
    return splits


if __name__ == "__main__":
    freeze_split()