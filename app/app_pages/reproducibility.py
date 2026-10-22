import streamlit as st

from common import load_json, require

manifest = load_json("results/run_manifest.json")
sel = load_json("models/selected_model.json")
audit = load_json("reports/data_audit.json")
require(manifest, sel, audit)

st.title("Reproducibility and references")

c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.subheader("Reproduce the results (Windows PowerShell, project root)")
        st.code(
            ".venv\\Scripts\\Activate.ps1\n"
            "$env:OPENBLAS_NUM_THREADS = \"1\"\n"
            "python -m pytest -q\n"
            "python -m ids_bot.audit\n"
            "python -m ids_bot.run_experiments --groups A B C D E F\n"
            "python -m ids_bot.select_model\n"
            "python -m ids_bot.verify\n"
            "python -m ids_bot.summarize\n"
            "python reports\\figures\\make_figures.py\n"
            "streamlit run app\\streamlit_app.py", language="powershell")
        st.caption("`run_experiments` skips runs already recorded in results/metrics.csv. "
                   "`select_model` and `verify` alone re-check the saved model in about a minute.")
with c2:
    with st.container(border=True):
        st.subheader("Implementation details")
        pk = manifest["python_packages"]
        st.markdown(
            f"- Python {sel['versions']['python']}; scikit-learn {pk['sklearn']}, imbalanced-learn {pk['imblearn']}, "
            f"XGBoost {pk['xgboost']}, pandas {pk['pandas']}, NumPy {pk['numpy']} (full lock: requirements-lock.txt)\n"
            f"- Seeds: {manifest['seeds']['primary']} (primary), {manifest['seeds']['robustness']} (robustness)\n"
            f"- Test fraction {manifest['test_size']}, validation fraction {manifest['val_size']} of training rows\n"
            f"- Raw data SHA-256 `{audit['provenance']['sha256']}`\n"
            "- Artifacts: results/metrics.csv, results/run_log.txt, results/verification.json, "
            "models/selected_model.joblib + .json, reports/data_audit.json")

with st.container(border=True):
    st.subheader("Relationship to earlier work")
    st.markdown(
        "This investigation re-examines the pipeline proposed in an earlier manuscript by the same author "
        "(*Improving Minority Cyberattack Detection in Network Traffic Using Data Engineering and Imbalance-Aware "
        "Machine Learning*). The present study re-implements the data preparation and models with a leakage-safe "
        "protocol and adds robustness analyses; its findings are reported in "
        "`reports/ANEHA_P_RESEARCH_PAPER.md`.")

with st.container(border=True):
    st.subheader("Key references")
    st.markdown(
        "1. I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani. Toward Generating a New Intrusion Detection Dataset and "
        "Intrusion Traffic Characterization. *Proc. ICISSP 2018*, pp. 108–116. doi:10.5220/0006639801080116 "
        "(dataset citation).\n"
        "2. N. V. Chawla, K. W. Bowyer, L. O. Hall, W. P. Kegelmeyer. SMOTE: Synthetic Minority Over-sampling "
        "Technique. *JAIR* 16 (2002) 321–357. doi:10.1613/jair.953\n"
        "3. T. Chen, C. Guestrin. XGBoost: A Scalable Tree Boosting System. *KDD 2016*, 785–794. "
        "doi:10.1145/2939672.2939785\n"
        "4. G. Engelen, V. Rimmer, W. Joosen. Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case "
        "Study. *IEEE SPW 2021*, 7–12. doi:10.1109/SPW53761.2021.00009\n"
        "5. D. Arp et al. Dos and Don'ts of Machine Learning in Computer Security. *USENIX Security 2022*, "
        "3971–3988.\n\n"
        "The full reference list is in the research paper.")
