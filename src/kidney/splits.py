"""5-fold split, plan.md 1.2.

The plan asks for one `results/splits/folds.csv` (`image_id, fold`) read by every notebook.
The team already committed a leak-free split (`results/eda/folds.csv`) that keeps near-duplicate
groups (references.md, "Our findings") in the same fold. We adopt it as the canonical split, so
copies of one image never sit in train and test at once. If it is missing we fall back to
`StratifiedGroupKFold` / `StratifiedKFold` [S25] with random_state=42, stratified on Global diagnosis.

Per repetition k (as in [24]): test = fold k, validation = fold (k+1) % 5, train = the other three.
"""
from __future__ import annotations

import pandas as pd

from . import config as C


def build_folds(force: bool = False) -> pd.DataFrame:
    C.SPLITS.mkdir(parents=True, exist_ok=True)
    if C.FOLDS_CSV.exists() and not force:
        return read_folds()
    if C.TEAM_FOLDS_CSV.exists():
        t = pd.read_csv(C.TEAM_FOLDS_CSV)
        out = pd.DataFrame({
            "image_id": t["filename"].str.replace(".jpg", "", regex=False).astype(str),
            "fold": t["fold"].astype(int),
            "diagnosis": t["diagnosis"].astype(int),
            "group_id": t["group_id"].astype(str),
        })
    else:
        from sklearn.model_selection import StratifiedKFold

        from .data import list_dataset_a

        df = list_dataset_a()
        skf = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
        df["fold"] = -1
        for k, (_, te) in enumerate(skf.split(df, df["diagnosis"])):
            df.loc[te, "fold"] = k
        out = df[["image_id", "fold", "diagnosis"]].assign(group_id=df["image_id"])
    out.to_csv(C.FOLDS_CSV, index=False)
    return out


def read_folds() -> pd.DataFrame:
    if not C.FOLDS_CSV.exists():
        return build_folds()
    return pd.read_csv(C.FOLDS_CSV, dtype={"image_id": str, "group_id": str})


def fold_roles(test_fold: int, n_folds: int = C.N_FOLDS) -> dict:
    val = (test_fold + 1) % n_folds
    train = [f for f in range(n_folds) if f not in (test_fold, val)]
    return {"test": test_fold, "val": val, "train": train}


def split_ids(test_fold: int, folds: pd.DataFrame | None = None) -> dict:
    folds = read_folds() if folds is None else folds
    r = fold_roles(test_fold)
    return {
        "train": folds.loc[folds.fold.isin(r["train"]), "image_id"].tolist(),
        "val": folds.loc[folds.fold == r["val"], "image_id"].tolist(),
        "test": folds.loc[folds.fold == r["test"], "image_id"].tolist(),
    }


if __name__ == "__main__":
    f = build_folds(force=True)
    print(f.groupby("fold").diagnosis.agg(["size", "mean"]))
    # Leak check: no group in two folds
    assert (f.groupby("group_id").fold.nunique() == 1).all(), "a duplicate group spans two folds"
    print("saved", C.FOLDS_CSV)
