# Minority Bot Traffic Detection Using Imbalance-Aware Machine Learning: research dashboard

Academic research demonstration by **Aneha P.** It presents an experimental study of minority Bot-traffic
detection on the CIC-IDS2017 Friday-morning network-flow capture: dataset, methodology, recorded
experiment results, robustness analyses and limitations. It also demonstrates the saved model on stored
flow records.

**This is not an intrusion-detection or network-monitoring system.** It does not capture, monitor or block
live network traffic.

## Contents

| Path | Purpose |
|---|---|
| `app/` | Streamlit dashboard; entry point `app/streamlit_app.py` |
| `ids_bot/` | Project modules needed to load the saved pipelines |
| `models/` | Selected model and comparison model (scikit-learn / imbalanced-learn / XGBoost pipelines) with metadata |
| `results/metrics.csv` | Recorded metrics of all experiment runs |
| `results/verification.json` | Re-evaluation of the selected model and file-order sensitivity refits |
| `results/demo/` | 2,000-row sample of the held-out test partition (all 643 Bot rows plus 1,357 benign rows) for the prediction demonstration |
| `reports/` | Dataset audit statistics and the figures shown in the dashboard |

The full raw dataset is not included. The sample rows are taken unchanged from the CIC-IDS2017 file
`Friday-WorkingHours-Morning.pcap_ISCX.csv`, published by the Canadian Institute for Cybersecurity for
research use.

## Run locally

```bash
pip install -r requirements.txt
```

```bash
streamlit run app/streamlit_app.py
```

## Citation

Dataset: I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, "Toward Generating a New Intrusion Detection
Dataset and Intrusion Traffic Characterization," in *Proc. ICISSP 2018*, pp. 108–116,
doi:10.5220/0006639801080116.
