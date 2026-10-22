import pandas as pd
import streamlit as st

from common import figure, load_json, load_metrics, require

metrics = load_metrics()
manifest = load_json("results/run_manifest.json")
require(metrics, manifest)

st.title("Methodology")
figure("fig1_pipeline.png",
       "Figure 1. Experimental pipeline. Colours mark which partition each step may use; no learned step "
       "sees validation or test rows.")

col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.subheader("Leakage safeguards (implemented and unit-tested)")
        st.markdown(
            "- Cleaning and de-duplication are deterministic row filters applied before the split; nothing is "
            "learned from the data in these steps.\n"
            "- Every learned step sits inside one `imblearn` pipeline that is fitted on training rows only: "
            "the destination-port frequency table, the constant-column filter, the Min–Max scaler, SMOTE and "
            "the classifier.\n"
            "- **SMOTE** (Synthetic Minority Over-sampling Technique) creates synthetic Bot rows by "
            "interpolating between neighbouring Bot rows. It runs only while fitting; validation and test rows "
            "are never resampled.\n"
            "- Validation rows are used only to choose the final model; the test partition is scored once per "
            "run and never used for selection.\n"
            "- The raw file's SHA-256 is verified every time it is loaded.")
with col2:
    with st.container(border=True):
        st.subheader("Methods actually implemented")
        st.dataframe(pd.DataFrame([
            ("Logistic Regression", "lbfgs, max_iter 2000"),
            ("Gaussian Naive Bayes", "defaults"),
            ("Decision Tree", "defaults"),
            ("K-nearest neighbours", "k = 5, brute force"),
            ("Isolation Forest", "200 trees, contamination = training Bot rate"),
            ("Random Forest", "200 trees"),
            ("XGBoost", "300 trees, depth 6, learning rate 0.1, hist"),
        ], columns=["Model", "Settings"]), hide_index=True)
        st.markdown(
            "**Imbalance handling:** none · class weights (`balanced` / `scale_pos_weight`) · SMOTE to 1:1 · "
            "SMOTE to 10 % plus class weights.  \n"
            "**Feature sets:** flow-only (no port information) · flow + port · flow + port + port frequency · "
            "port and port frequency only.  \n"
            f"**Seeds:** {manifest['seeds']['primary']} (primary), {manifest['seeds']['robustness']} (robustness). "
            f"**Threshold:** 0.5 for every model.")

with st.container(border=True):
    st.subheader("Experiment groups")
    counts = metrics.drop_duplicates("run_id").group.value_counts()
    groups = pd.DataFrame([
        ("A", "A_baselines", "Baselines; LR with class weights and SMOTE", "primary protocol", "planned (2 LR runs added later)"),
        ("B", "B_imbalance", "RF / XGBoost × four imbalance strategies; selection pool", "primary protocol", "planned"),
        ("C", "C_port_shortcut", "Destination-port feature sets with SMOTE", "primary protocol", "planned"),
        ("D", "D_paper_aligned", "Duplicates retained, models fitted on the full 67 % training split", "alternative protocol", "planned"),
        ("E", "E_seed_robustness", "Five split/model seeds", "primary protocol", "planned"),
        ("F", "F_file_order_holdout", "Split by file order instead of at random", "file-order proxy", "added after initial results"),
    ], columns=["Group", "key", "Purpose", "Protocol", "Status"])
    groups["Runs"] = groups.key.map(counts).fillna(0).astype(int)
    st.dataframe(groups.drop(columns="key"), hide_index=True)
    st.caption("Primary protocol: NaN/inf and exact duplicates removed, stratified 67/33 train/test split "
               "(seed 42), stratified 20 % of training rows held out for validation. A further three "
               "sensitivity refits for group F are stored in results/verification.json.")

with st.container(border=True):
    st.subheader("Not implemented (possible future work)")
    st.markdown("Threshold tuning, hyper-parameter search, evaluation on other days or datasets, host- or "
                "session-grouped splits, and deep-learning models were **not** carried out in this study.")
