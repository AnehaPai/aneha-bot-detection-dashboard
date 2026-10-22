"""Shared loaders and helpers for the dashboard pages.

Every number shown in the dashboard comes from one of these canonical artifacts:
  reports/data_audit.json      dataset statistics (written by ids_bot.audit)
  results/metrics.csv          all recorded experiment runs (ids_bot.run_experiments)
  models/selected_model.json   selected model record (ids_bot.select_model)
  results/verification.json    re-evaluated selected model + file-order refits (ids_bot.verify)
  results/run_manifest.json    package versions and seeds
or is computed on load from the hash-checked raw CSV with the project's own data functions.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ids_bot import config as C  # noqa: E402
from ids_bot.data import prepare, split_indices  # noqa: E402

FIGURES = ROOT / "reports" / "figures"
MODEL_NAMES = {"logreg": "Logistic Regression", "naive_bayes": "Gaussian NB", "decision_tree": "Decision Tree",
               "knn": "KNN (k=5)", "isolation_forest": "Isolation Forest", "random_forest": "Random Forest",
               "xgboost": "XGBoost"}
IMBALANCE_NAMES = {"none": "none", "class_weight": "class weights", "smote": "SMOTE",
                   "smote_cw": "SMOTE + class weights"}
FEATURE_SET_NAMES = {"flow_only": "flow-only", "with_port": "flow + port", "with_port_freq": "flow + port + port frequency",
                     "port_only": "port + port frequency only"}
SCORE_COLS = ["precision_bot", "recall_bot", "f1_bot", "pr_auc_bot", "accuracy"]


@st.cache_data
def load_json(rel):
    p = ROOT / rel
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def load_metrics():
    p = ROOT / "results" / "metrics.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource(max_entries=2)
def load_model(name):
    p = ROOT / "models" / f"{name}.joblib"
    return joblib.load(p) if p.exists() else None


DEMO = ROOT / "results" / "demo"
# The hosted deployment ships no raw CSV; it falls back to the small exported sample
# written by `python -m ids_bot.export_demo_sample`. Locally the full raw file is used.
RAW_AVAILABLE = C.RAW_CSV.exists()


@st.cache_data(max_entries=1)
def load_clean_data():
    df, _ = prepare(dedup=True)
    return df


@st.cache_data(max_entries=1)
def load_test_rows():
    if not RAW_AVAILABLE:
        return pd.read_csv(DEMO / "test_sample.csv")
    df = load_clean_data()
    positions = np.load(ROOT / "results" / "test_positions_primary.npy")
    return df.iloc[positions].reset_index(drop=True)


def test_rows_source():
    if RAW_AVAILABLE:
        return "the full primary test partition"
    m = json.loads((DEMO / "sample_manifest.json").read_text())
    return (f"a {m['rows']:,}-row sample of the primary test partition (all {m['bot_rows']} Bot rows and "
            f"{m['benign_rows']:,} of {m['benign_rows_in_full_test_partition']:,} BENIGN rows); the raw dataset "
            "is not included in this hosted demonstration")


@st.cache_data(max_entries=1)
def late_pattern_counts():
    """Bot flows on port 8080 with Init_Win_bytes_backward == 0, per file-order segment."""
    if not RAW_AVAILABLE:
        return json.loads((DEMO / "late_pattern_counts.json").read_text())
    df = load_clean_data()
    o = split_indices(df, ordered=True)
    out = {}
    for k in ("fit", "val", "test"):
        part = df.iloc[o[k]]
        bot = part[part[C.LABEL] == C.POSITIVE]
        out[k] = int(((bot[C.PORT] == 8080) & (bot["Init_Win_bytes_backward"] == 0)).sum())
    return out


def require(*artifacts):
    missing = [a for a in artifacts if a is None]
    if missing:
        st.warning("Some result files are missing. Run the pipeline first (see the Reproducibility page).")
        st.stop()


def pretty(df):
    out = df.copy()
    if "model" in out:
        out["model"] = out["model"].map(MODEL_NAMES).fillna(out["model"])
    if "imbalance" in out:
        out["imbalance"] = out["imbalance"].map(IMBALANCE_NAMES).fillna(out["imbalance"])
    if "feature_set" in out:
        out["feature_set"] = out["feature_set"].map(FEATURE_SET_NAMES).fillna(out["feature_set"])
    return out


def metric_table(df, cols=("model", "imbalance", "feature_set", "precision_bot", "recall_bot", "f1_bot",
                            "pr_auc_bot", "accuracy", "tp", "fn", "fp", "tn")):
    shown = pretty(df[[c for c in cols if c in df.columns]])
    config = {c: st.column_config.NumberColumn(format="%.4f") for c in SCORE_COLS}
    config.update({
        "precision_bot": st.column_config.NumberColumn("Bot precision", format="%.4f"),
        "recall_bot": st.column_config.NumberColumn("Bot recall", format="%.4f"),
        "f1_bot": st.column_config.NumberColumn("Bot F1", format="%.4f"),
        "pr_auc_bot": st.column_config.NumberColumn("Bot PR-AUC", format="%.4f"),
        "accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f"),
        "tp": st.column_config.NumberColumn("TP", format="%d"), "fn": st.column_config.NumberColumn("FN", format="%d"),
        "fp": st.column_config.NumberColumn("FP", format="%d"), "tn": st.column_config.NumberColumn("TN", format="%d"),
        "feature_set": "Features", "imbalance": "Imbalance handling", "model": "Model",
    })
    st.dataframe(shown, hide_index=True, column_config=config)


def figure(name, caption):
    p = FIGURES / name
    if p.exists():
        st.image(str(p), caption=caption)
    else:
        st.info(f"Figure `{name}` not generated yet: run `python reports/figures/make_figures.py`.")


def confusion_frame(cm):
    return pd.DataFrame({"Predicted BENIGN": [cm["tn"], cm["fn"]], "Predicted Bot": [cm["fp"], cm["tp"]]},
                        index=["True BENIGN", "True Bot"])
