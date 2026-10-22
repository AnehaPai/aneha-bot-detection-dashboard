import pandas as pd
import streamlit as st

from common import figure, late_pattern_counts, load_json, load_metrics, metric_table, require

metrics = load_metrics()
ver = load_json("results/verification.json")
audit = load_json("reports/data_audit.json")
require(metrics, ver, audit)
test = metrics[metrics.split == "test"]

st.title("Robustness and limitations")
st.markdown("High scores on a random split can hide fragile behaviour. This page reports the checks that "
            "probe it, including the result that changed after verification.")

seg = ver["file_order_holdout"]["segments"]
late = late_pattern_counts()
f_rows = test[test.group == "F_file_order_holdout"]
flow_tp = int(f_rows[f_rows.feature_set == "flow_only"].tp.max())
st.subheader("1. File-order sensitivity analysis")
st.markdown(
    "The CSV has no timestamps, so the rows were split **by their order in the file** as a proxy for time: "
    f"the first {seg['fit']['rows'] + seg['val']['rows']:,} rows (67 %) for training and the last "
    f"{seg['test']['rows']:,} rows (33 %) for testing. This is **not a genuine temporal evaluation**.")
c1, c2 = st.columns([1.2, 1])
with c1:
    figure("fig5_file_order_sensitivity.png",
           "Figure 5. Bot recall on the last third of the file for two training windows (all models use SMOTE).")
with c2:
    with st.container(border=True):
        st.markdown("**What happened**")
        st.markdown(
            f"- In the original group-F runs, models were fitted on the fit part only (first 53.6 %, "
            f"{seg['fit']['bot']} Bot flows); the last {seg['val']['rows']:,} training rows served as validation.\n"
            f"- The test third contains a Bot pattern (port 8080, `Init_Win_bytes_backward` = 0, long flows) that "
            f"occurs **{late['fit']}** times in the fit part, **{late['val']}** times in the validation slice and "
            f"**{late['test']}** times in the test part.\n"
            f"- Flow-only models fitted on the fit part detected **{flow_tp} of {seg['test']['bot']}** test Bot "
            "flows. This was initially read as a failure on later traffic.\n"
            "- **Corrected:** refitted on the full first 67 % (including those "
            f"{late['val']} flows), the models detected "
            + ", ".join(f"**{r['tp']} of {r['tp'] + r['fn']}** ({r['model'].replace('_', ' ')}, "
                        f"{r['feature_set'].replace('_', ' ')})"
                        for r in ver["file_order_holdout"]["sensitivity_train_on_full_first_67pct"]) + ".")
        st.markdown("**Interpretation:** detection depends on whether a Bot behaviour is represented in "
                    "training at all. The zero-recall result reflects an incomplete training window, not proof "
                    "that later traffic is undetectable; the recovery shows that a few examples suffice for this "
                    "pattern, not that the models generalise to new botnets.")

with st.expander("File-order runs in detail"):
    st.markdown("**Recorded runs (fitted on the first 53.6 % of rows)**")
    metric_table(test[test.group == "F_file_order_holdout"])
    st.markdown("**Sensitivity refits (fitted on the first 67 %)**, from results/verification.json")
    metric_table(pd.DataFrame(ver["file_order_holdout"]["sensitivity_train_on_full_first_67pct"])
                 .assign(imbalance="smote"))

st.subheader("2. Destination-port shortcut")
dp = audit["destination_port"]
port = test[(test.group == "C_port_shortcut") | ((test.group == "B_imbalance") & (test.imbalance == "smote"))]
po = port[port.feature_set == "port_only"]
st.markdown(
    f"{dp['bot_rows_on_8080']:,} of {dp['bot_rows_total']:,} Bot flows target port 8080 and no benign flow does. "
    f"Under the random split the port alone is a weak detector (Bot F1 {po.f1_bot.min():.4f}–{po.f1_bot.max():.4f}), "
    "because the remaining Bot flows use ephemeral ports shared with benign traffic. In the file-order test, "
    "however, port features caught every late Bot flow, purely because the command-and-control port never "
    "changed. A different botnet or port would remove this signal.")

st.subheader("3. Seed variability")
e = test[test.group == "E_seed_robustness"].groupby(["model", "feature_set"]).f1_bot.agg(["mean", "std"])
st.markdown("Bot F1 over five split/model seeds: " + "; ".join(
    f"{m.replace('_', ' ')} {fs.replace('_', ' ')} {r['mean']:.4f} ± {r['std']:.4f}" for (m, fs), r in e.iterrows())
    + ". Differences between the strong models are of the same order as this variation.")

with st.container(border=True):
    st.subheader("Threats to validity")
    st.markdown(
        "- Single dataset file, single capture morning, single botnet: no external-dataset evaluation.\n"
        "- File order is only a proxy for time; no host- or session-grouped split was possible.\n"
        "- Random splits can place flows of the same bot sessions in both training and test (session identifiers "
        "are not available to check this), so they mainly measure recognition of known patterns.\n"
        "- Published CIC-IDS2017 labels contain known errors (Engelen et al., 2021).\n"
        "- Fixed hyper-parameters and a fixed 0.5 threshold; no tuning was performed.\n"
        "- Group F and the sensitivity refits were added after initial results and are exploratory.")
