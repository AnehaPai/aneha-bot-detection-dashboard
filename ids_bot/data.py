"""Loading, schema validation, cleaning, de-duplication and splitting.

Nothing in this module learns from data: every step here is either a
deterministic schema fix or a row filter that does not depend on the labels of
other rows, so it is safe to apply before the train/test split.
"""
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_raw(path=C.RAW_CSV, verify_hash=True):
    """Read the unmodified CSV and strip the stray spaces from column names."""
    if verify_hash:
        digest = sha256_file(path)
        if digest != C.RAW_SHA256:
            raise ValueError(f"SHA-256 mismatch for {path}: {digest}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    validate_schema(df)
    return df


def validate_schema(df):
    if df.shape[1] != 79:
        raise ValueError(f"expected 79 columns, found {df.shape[1]}")
    if C.LABEL not in df.columns or C.PORT not in df.columns:
        raise ValueError("Label or Destination Port column missing")
    labels = set(df[C.LABEL].unique())
    if labels != {C.NEGATIVE, C.POSITIVE}:
        raise ValueError(f"unexpected labels: {labels}")
    non_numeric = [c for c in df.columns if c != C.LABEL and not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"non-numeric feature columns: {non_numeric}")


def clean(df):
    """Drop the duplicated header column and rows containing NaN or +/-inf.

    Returns the cleaned frame and a dict describing what was removed.
    """
    report = {}
    dup_col_identical = bool((df[C.DUPLICATE_COLUMN] == df["Fwd Header Length"]).all())
    report["duplicate_column_identical"] = dup_col_identical
    if dup_col_identical:
        df = df.drop(columns=[C.DUPLICATE_COLUMN])

    feats = df.drop(columns=[C.LABEL])
    nan_mask = feats.isna().any(axis=1)
    inf_mask = np.isinf(feats).any(axis=1)
    bad = nan_mask | inf_mask
    report["rows_with_nan"] = int(nan_mask.sum())
    report["rows_with_inf"] = int(inf_mask.sum())
    report["rows_with_nan_or_inf"] = int(bad.sum())
    report["nan_or_inf_rows_by_class"] = df.loc[bad, C.LABEL].value_counts().to_dict()
    report["columns_with_nan"] = feats.columns[feats.isna().any()].tolist()
    report["columns_with_inf"] = feats.columns[np.isinf(feats).any()].tolist()
    df = df.loc[~bad].copy()
    report["rows_after_clean"] = len(df)
    report["class_counts_after_clean"] = df[C.LABEL].value_counts().to_dict()
    return df, report


def deduplicate(df):
    """Keep the first occurrence of every exact (features + label) duplicate."""
    dup = df.duplicated(keep="first")
    report = {
        "duplicate_rows_removed": int(dup.sum()),
        "duplicate_rows_removed_by_class": df.loc[dup, C.LABEL].value_counts().to_dict(),
    }
    out = df.loc[~dup].copy()
    report["rows_after_dedup"] = len(out)
    report["class_counts_after_dedup"] = out[C.LABEL].value_counts().to_dict()
    return out, report


def feature_columns(df):
    return [c for c in df.columns if c != C.LABEL]


def to_xy(df, columns=None):
    columns = columns or feature_columns(df)
    X = df[columns].astype(np.float32)
    y = (df[C.LABEL] == C.POSITIVE).astype(np.int8).to_numpy()
    return X, y


def split_indices(df, seed=C.SEED, test_size=C.TEST_SIZE, val_size=C.VAL_SIZE, ordered=False):
    """Stratified train/test split, then a stratified validation slice of train.

    With ordered=True the split is by file position instead (first rows train,
    last rows test, validation = last part of train). The CSV has no timestamp
    column, so file order is only a proxy for capture time.

    Returns positional index arrays: train (full), fit (train minus val), val, test.
    """
    pos = np.arange(len(df))
    if ordered:
        cut = int(round(len(df) * (1 - test_size)))
        vcut = int(round(cut * (1 - val_size)))
        return {"train": pos[:cut], "fit": pos[:vcut], "val": pos[vcut:cut], "test": pos[cut:]}
    y = df[C.LABEL].to_numpy()
    train, test = train_test_split(pos, test_size=test_size, stratify=y, random_state=seed)
    fit, val = train_test_split(train, test_size=val_size, stratify=y[train], random_state=seed)
    return {"train": np.sort(train), "fit": np.sort(fit), "val": np.sort(val), "test": np.sort(test)}


def prepare(dedup=True):
    """Full deterministic data preparation used by every experiment."""
    raw = load_raw()
    cleaned, clean_report = clean(raw)
    if dedup:
        cleaned, dedup_report = deduplicate(cleaned)
    else:
        dedup_report = {"duplicate_rows_removed": 0}
    return cleaned.reset_index(drop=True), {**clean_report, **dedup_report}
