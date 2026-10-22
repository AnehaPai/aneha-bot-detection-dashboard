import streamlit as st

from common import confusion_frame, figure, load_json, load_metrics, metric_table, require

metrics = load_metrics()
sel = load_json("models/selected_model.json")
ver = load_json("results/verification.json")
require(metrics, sel, ver)

st.title("Selected model")
v = ver["selected_model"]
cm = v["confusion_matrix"]

with st.container(border=True):
    st.subheader("How the model was selected")
    st.markdown(
        "Among the eight group-B configurations (Random Forest and XGBoost, each with four imbalance strategies, "
        "flow-only features), the model with the **highest validation Bot F1** was chosen, with validation "
        "PR-AUC as tie-breaker. Only the 24,662 validation rows were used; test rows played no part in the "
        "choice. The selected configuration was then refitted with the same seed and its test confusion matrix "
        "was checked against the recorded run.")
    val = metrics[(metrics.group == "B_imbalance") & (metrics.split == "val")].sort_values("f1_bot", ascending=False)
    metric_table(val, cols=("model", "imbalance", "precision_bot", "recall_bot", "f1_bot", "pr_auc_bot"))
    st.caption("Validation metrics (not test metrics). Process note: the exact ranking rule was written into "
               "the code after test metrics had appeared in the run log; it uses validation rows only, and the "
               "validation winner is also the best test configuration.")

st.subheader(f"Held-out test performance: XGBoost, {sel['imbalance'].replace('none', 'no resampling')}, "
             f"{sel['feature_set'].replace('_', '-')} features")
with st.container(horizontal=True):
    st.metric("Bot precision", f"{v['precision_bot']:.4f}", border=True)
    st.metric("Bot recall", f"{v['recall_bot']:.4f}", border=True)
    st.metric("Bot F1", f"{v['f1_bot']:.4f}", border=True)
    st.metric("Bot PR-AUC", f"{sel['test_metrics']['pr_auc_bot']:.4f}", border=True)
    st.metric("Matches recorded run", "yes" if v["matches_metrics_csv"] and v["matches_model_json"] else "NO",
              border=True)

c1, c2 = st.columns([1, 1.2])
with c1:
    figure("fig4_confusion_matrix.png", "Figure 4. Confusion matrix of the selected model on the test partition.")
with c2:
    with st.container(border=True):
        st.markdown("**Reading the confusion matrix** (rows = true class, columns = predicted class)")
        st.dataframe(confusion_frame(cm))
        st.markdown(
            f"- **True positives (TP) = {cm['tp']}**: Bot flows correctly flagged.\n"
            f"- **False negatives (FN) = {cm['fn']}**: Bot flows missed.\n"
            f"- **False positives (FP) = {cm['fp']}**: benign flows wrongly flagged.\n"
            f"- **True negatives (TN) = {cm['tn']:,}**: benign flows correctly passed.\n\n"
            f"Precision = TP / (TP + FP) = {cm['tp']} / {cm['tp'] + cm['fp']} = {v['precision_bot']:.4f}.  \n"
            f"Recall = TP / (TP + FN) = {cm['tp']} / {cm['tp'] + cm['fn']} = {v['recall_bot']:.4f}.")
st.info("These figures come from a random stratified split of one capture. They show how well the model "
        "recognises Bot behaviour already represented in training, not how it would perform on another network. "
        "See *Robustness and limitations*.", icon=":material/info:")
