import streamlit as st

from common import load_json, load_metrics, require

audit = load_json("reports/data_audit.json")
metrics = load_metrics()
ver = load_json("results/verification.json")
require(audit, metrics, ver)

st.title("Minority Bot traffic detection using imbalance-aware machine learning")
st.markdown("An experimental study of network-flow classification, evaluation reliability, and "
            "minority-class detection · **Aneha P.**")

with st.container(border=True):
    st.markdown(
        "**Research question.** How effectively can data preparation and imbalance-aware machine-learning "
        "methods detect minority Bot traffic in a labeled network-flow dataset, and how reliable are those "
        "results under different evaluation conditions?")

raw = audit["class_counts_raw"]
sel = ver["selected_model"]
f = metrics[(metrics.split == "test") & (metrics.group == "F_file_order_holdout") & (metrics.model == "xgboost")
            & (metrics.feature_set == "flow_only")].iloc[0]
sens = [r for r in ver["file_order_holdout"]["sensitivity_train_on_full_first_67pct"]
        if r["model"] == "xgboost" and r["feature_set"] == "flow_only"][0]
with st.container(horizontal=True):
    st.metric("Flows in the dataset file", f"{audit['schema']['rows']:,}", border=True)
    st.metric("Bot share of flows", f"{raw['Bot'] / (raw['Bot'] + raw['BENIGN']):.2%}", border=True)
    st.metric("Selected model: Bot recall (random split)", f"{sel['recall_bot']:.4f}", border=True)
    st.metric("Selected model: Bot precision (random split)", f"{sel['precision_bot']:.4f}", border=True)
    st.metric("Unseen Bot pattern detected (file order)",
              f"{int(f.tp)} / {int(f.tp + f.fn)} → {sens['tp']} / {sens['tp'] + sens['fn']}", border=True,
              help="XGBoost flow-only, trained on the first 53.6 % of rows vs the first 67 % of rows.")

col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.subheader("Problem statement")
        st.markdown(
            "Network intrusion detection datasets are dominated by benign traffic. In the file studied here, "
            f"only {raw['Bot']:,} of {audit['schema']['rows']:,} flows are labelled Bot. A classifier can "
            "therefore reach very high accuracy while missing most attacks, and an evaluation that looks "
            "excellent on a random split may say little about traffic patterns the model has not seen.")
        st.subheader("Why minority-class detection matters")
        st.markdown(
            "- A missed Bot flow (false negative) is an undetected compromise.\n"
            "- A false alarm (false positive) costs analyst time; thousands per day make a detector unusable.\n"
            "- Accuracy hides both: predicting *always benign* would already be about 99 % accurate here.")
with col2:
    with st.container(border=True):
        st.subheader("Sub-questions")
        st.markdown(
            "- **RQ1** How well do the evaluated models identify Bot flows compared with benign flows?\n"
            "- **RQ2** How do imbalance-handling strategies (class weights, SMOTE) change Bot recall, "
            "precision, F1 and the false-positive burden?\n"
            "- **RQ3** How sensitive are the findings to the distribution and ordering of the records?\n"
            "- **RQ4** What risks arise from dataset-specific features such as the destination port, and "
            "from evaluating on a single dataset?")
        st.subheader("Objectives and scope")
        st.markdown(
            "- Build a leakage-safe, reproducible pipeline: cleaning, de-duplication, training-only "
            "preprocessing, validation-based model selection, one held-out test evaluation.\n"
            "- Compare standard classifiers and imbalance-handling strategies under identical conditions.\n"
            "- Test how far the results hold under alternative evaluation conditions.\n"
            "- **Out of scope:** live traffic capture, real-time detection, blocking, other datasets. "
            "This dashboard replays recorded results and demonstrates the saved model on stored records.")

st.caption("Positive class throughout: **Bot**. Negative class: **BENIGN**. Values on this page are read from "
           "reports/data_audit.json, results/metrics.csv and results/verification.json.")
