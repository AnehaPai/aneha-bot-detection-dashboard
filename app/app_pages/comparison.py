import streamlit as st

from common import IMBALANCE_NAMES, MODEL_NAMES, confusion_frame, figure, load_metrics, metric_table, require

metrics = load_metrics()
require(metrics)
test = metrics[metrics.split == "test"]

st.title("Model comparison")
st.markdown("All scores refer to the **Bot class as the positive class**, on held-out test rows. Each view "
            "below compares runs that share the same split, features and seed, so the comparison is fair.")

def sizes(group):
    r = test[test.group == group].iloc[0]
    return f"{int(r.support_benign + r.support_bot):,} test flows, {int(r.support_bot)} Bot."


views = {
    "Classifiers and imbalance handling": "Primary protocol, flow-only features, seed 42 (groups A and B). "
                                          + sizes("B_imbalance"),
    "Feature sets": "Primary protocol, SMOTE, seed 42 (groups B and C).",
    "Duplicates retained": "Alternative protocol: exact duplicates kept, models fitted on the full 67 % "
                           "training split (group D). " + sizes("D_paper_aligned"),
    "Seed variability": "Primary protocol repeated with split/model seeds 0–4 (group E).",
}
view = st.segmented_control("Evaluation condition", list(views), default=list(views)[0])
view = view or list(views)[0]
st.caption(views[view])

if view == "Classifiers and imbalance handling":
    rows = test[(test.feature_set == "flow_only") & (test.seed == 42)
                & test.group.isin(["A_baselines", "B_imbalance"])].sort_values("f1_bot", ascending=False)
    metric_table(rows)
    figure("fig3_model_comparison.png", "Figure 3. Bot-class precision, recall and F1 (a) and false positives (b) "
           "for every classifier and imbalance strategy under identical conditions.")
elif view == "Feature sets":
    rows = test[(test.group == "C_port_shortcut") | ((test.group == "B_imbalance") & (test.imbalance == "smote"))]
    metric_table(rows.sort_values(["model", "feature_set"]))
    figure("fig6_port_shortcut.png", "Figure 6. Effect of destination-port features and stand-alone strength of "
           "the port.")
elif view == "Duplicates retained":
    d = test[test.group == "D_paper_aligned"]
    metric_table(d)
    prim = test[test.group.isin(["A_baselines", "B_imbalance", "C_port_shortcut"]) & (test.seed == 42)]
    diffs = [abs(r.f1_bot - prim[(prim.model == r.model) & (prim.imbalance == r.imbalance)
                                 & (prim.feature_set == r.feature_set)].f1_bot.iloc[0])
             for r in d.itertuples()
             if len(prim[(prim.model == r.model) & (prim.imbalance == r.imbalance) & (prim.feature_set == r.feature_set)])]
    st.caption(f"Largest Bot-F1 difference to the matching primary-protocol run: {max(diffs):.4f}. Only a handful "
               "of Bot test rows have an exact training duplicate (see Dataset page).")
else:
    e = test[test.group == "E_seed_robustness"]
    agg = e.groupby(["model", "feature_set"])[["precision_bot", "recall_bot", "f1_bot", "fn", "fp"]].agg(["mean", "std"])
    agg.columns = [f"{a} ({b})" for a, b in agg.columns]
    st.dataframe(agg.reset_index().round(4), hide_index=True)

with st.container(border=True):
    st.subheader("Why accuracy alone is misleading")
    lr = test[(test.group == "A_baselines") & (test.model == "logreg") & (test.imbalance == "none")
              & (test.feature_set == "flow_only")].iloc[0]
    st.markdown(
        f"Logistic Regression without imbalance handling reaches accuracy **{lr.accuracy:.4f}** but detects only "
        f"**{int(lr.tp)} of {int(lr.tp + lr.fn)}** Bot flows (recall {lr.recall_bot:.4f}). Because "
        f"{lr.support_benign / (lr.support_benign + lr.support_bot):.1%} of test flows are benign, a model that "
        "never flags anything is already about that accurate. Bot recall, precision, F1, PR-AUC and the raw "
        "false-positive count are therefore the primary measures.")

with st.container(border=True):
    st.subheader("Inspect one run")
    pool = test[test.group.isin(["A_baselines", "B_imbalance", "C_port_shortcut", "D_paper_aligned"])]
    labels = {r.run_id: f"{MODEL_NAMES[r.model]} · {IMBALANCE_NAMES[r.imbalance]} · {r.feature_set} · group {r.group[0]}"
              for r in pool.itertuples()}
    rid = st.selectbox("Run", list(labels), format_func=labels.get)
    r = pool[pool.run_id == rid].iloc[0]
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.dataframe(confusion_frame({k: int(r[k]) for k in ("tn", "fp", "fn", "tp")}))
    with c2:
        st.markdown(f"Bot precision **{r.precision_bot:.4f}** · recall **{r.recall_bot:.4f}** · "
                    f"F1 **{r.f1_bot:.4f}** · PR-AUC **{r.pr_auc_bot:.4f}**  \n"
                    f"Missed Bot flows (FN): **{int(r.fn)}** · false alarms (FP): **{int(r.fp):,}** "
                    f"out of {int(r.tn + r.fp):,} benign test flows.")
