import pandas as pd
import streamlit as st

from common import figure, load_json, require

audit = load_json("reports/data_audit.json")
sel = load_json("models/selected_model.json")
ver = load_json("results/verification.json")
require(audit, sel, ver)

st.title("Dataset")
st.markdown("CIC-IDS2017, file `Friday-WorkingHours-Morning.pcap_ISCX.csv` (Canadian Institute for "
            "Cybersecurity, University of New Brunswick; Sharafaldin et al., ICISSP 2018).")

prov = audit["provenance"]
raw, clean, dup = audit["class_counts_raw"], audit["cleaning"], audit["duplicates"]
after = dup["class_counts_after_dedup"]
lk = ver["leakage_checks"]

with st.container(horizontal=True):
    st.metric("Raw flows", f"{audit['schema']['rows']:,}", border=True)
    st.metric("Flows after cleaning", f"{dup['rows_after_dedup']:,}", border=True)
    st.metric("Imbalance ratio after cleaning", f"{after['BENIGN'] / after['Bot']:.1f} : 1",
              help="BENIGN flows per Bot flow", border=True)
    st.metric("Numeric features in file", audit["schema"]["feature_columns"], border=True)
    st.metric("Features used by selected model", lk["n_model_features_after_constant_filter"], border=True)

col1, col2 = st.columns([1.1, 1])
with col1:
    with st.container(border=True):
        st.subheader("Class counts by preparation stage")
        stages = pd.DataFrame({
            "Stage": ["Raw file", "After NaN/inf removal", "After duplicate removal"],
            "BENIGN": [raw["BENIGN"], clean["class_counts_after_clean"]["BENIGN"], after["BENIGN"]],
            "Bot": [raw["Bot"], clean["class_counts_after_clean"]["Bot"], after["Bot"]],
        })
        stages["Bot share"] = stages.Bot / (stages.Bot + stages.BENIGN)
        st.dataframe(stages, hide_index=True,
                     column_config={"Bot share": st.column_config.NumberColumn(format="percent")})
        figure("fig2_class_distribution.png", "Figure 2. Class counts per stage (log scale).")
with col2:
    with st.container(border=True):
        st.subheader("Provenance")
        st.markdown(
            f"- Size: {prov['size_bytes']:,} bytes; SHA-256 `{prov['sha256'][:16]}…` (checked on every load)\n"
            "- Obtained from a public Hugging Face mirror pinned to one commit, because the official "
            "download page requires a personal registration form; the hash matches two independent "
            "mirrors and the original `MachineLearningCSV.zip` archive entry.\n"
            "- Full record: `data/raw/PROVENANCE.md`, `reports/DATA_CARD.md`.")
    with st.container(border=True):
        st.subheader("Cleaning and duplicate decisions")
        cb = dup["cross_boundary_if_not_deduplicated"][0]
        st.markdown(
            f"- {clean['rows_with_nan_or_inf']} rows with NaN or ±inf removed "
            f"(BENIGN {clean['nan_or_inf_rows_by_class'].get('BENIGN', 0)}, "
            f"Bot {clean['nan_or_inf_rows_by_class'].get('Bot', 0)}), all in "
            f"{', '.join(clean['columns_with_inf'])}.\n"
            "- `Fwd Header Length.1` is an exact copy of `Fwd Header Length` and is dropped.\n"
            f"- {len(audit['schema']['constant_columns'])} constant columns are removed by a filter fitted on "
            "training rows.\n"
            f"- {dup['duplicate_rows_removed']:,} exact duplicate rows removed before splitting "
            f"(BENIGN {dup['duplicate_rows_removed_by_class']['BENIGN']:,}, "
            f"Bot {dup['duplicate_rows_removed_by_class']['Bot']}); no duplicate carries conflicting labels.\n"
            f"- Without de-duplication, {cb['test_rows_with_copy_in_train']:,} of {cb['test_rows']:,} test rows "
            f"would have an exact copy in training (Bot: {cb['by_class']['Bot']}).")

with st.container(border=True):
    st.subheader("Limitations and capture-specific risks")
    dp = audit["destination_port"]
    st.markdown(
        f"- **Destination port shortcut:** {dp['bot_rows_on_8080']:,} of {dp['bot_rows_total']:,} Bot flows go to "
        f"port 8080 and {dp['benign_rows_on_8080']} BENIGN flows do. This is the botnet's command-and-control "
        "port in this lab capture, not a general property of bots.\n"
        "- **One capture, one morning, one botnet**; results cannot be assumed to transfer to other networks.\n"
        "- **No timestamps or IP addresses** in this CSV release, so time-based or host-grouped splits are not "
        "possible; file order is the only available proxy.\n"
        "- **Label quality:** independent work has reported labelling and flow-construction errors in "
        "CIC-IDS2017 (Engelen et al., 2021); labels are used as published.")
