import numpy as np
import pandas as pd
import streamlit as st

from common import C, load_json, load_model, load_test_rows, require, test_rows_source

sel = load_json("models/selected_model.json")
model = load_model("selected_model")
port_model = load_model("port_feature_model")
require(sel, model)

st.title("Prediction demonstration")
st.info("This page applies the saved model to **stored or uploaded flow records** for demonstration. It does "
        "not capture or monitor live network traffic and cannot block anything.", icon=":material/info:")
st.markdown(
    f"Model: `models/selected_model.joblib` (XGBoost, flow-only features). The saved pipeline contains the "
    f"training-fitted preprocessing, so it takes the {len(sel['input_columns'])} original numeric CIC-IDS2017 "
    f"columns as input. Decision threshold: P(Bot) ≥ {sel['decision_threshold']}.")

st.subheader("Score a held-out test flow")
st.caption(f"Rows come from {test_rows_source()}. The test partition was used neither for training nor for "
           "model selection.")
rows = load_test_rows()
cls = st.segmented_control("True class of the example", ["Bot", "BENIGN"], default="Bot")
cls = cls or "Bot"
subset = rows[rows[C.LABEL] == cls]
i = st.number_input(f"Example number (0 to {len(subset) - 1:,})", 0, len(subset) - 1, 0, step=1)
row = subset.iloc[[int(i)]]
X = row[sel["input_columns"]].astype(np.float32)
prob = float(model.predict_proba(X)[0, 1])
pred = "Bot" if prob >= sel["decision_threshold"] else "BENIGN"
with st.container(horizontal=True):
    st.metric("True label", row[C.LABEL].iloc[0], border=True)
    st.metric("Predicted class", pred, border=True)
    st.metric("Estimated P(Bot)", f"{prob:.4f}", border=True,
              help="Model score after the logistic link; not a calibrated probability.")
    if port_model is not None:
        st.metric("P(Bot), comparison model with port features", f"{float(port_model.predict_proba(X)[0, 1]):.4f}",
                  border=True)
    st.metric("Destination port", int(row[C.PORT].iloc[0]), border=True)
st.caption("Correct" if pred == row[C.LABEL].iloc[0] else "Misclassified example: this is one of the model's errors.")
with st.expander("Feature values of this flow"):
    st.dataframe(row.drop(columns=[C.LABEL]).T.rename(columns={row.index[0]: "value"}))

st.divider()
st.subheader("Score uploaded records")
st.markdown("Upload a CSV with the original CIC-IDS2017 column names (leading spaces are stripped). Rows with "
            "NaN or infinite values are skipped, as in training. A `Label` column, if present, is shown for "
            "comparison only.")
up = st.file_uploader("Flow records (CSV)", type="csv")
if up is not None:
    new = pd.read_csv(up)
    new.columns = [c.strip() for c in new.columns]
    missing = [c for c in sel["input_columns"] if c not in new.columns]
    if missing:
        st.error(f"{len(missing)} required columns are missing, e.g. {missing[:5]}")
    else:
        Xn = new[sel["input_columns"]].replace([np.inf, -np.inf], np.nan)
        ok = Xn.notna().all(axis=1)
        out = new.loc[ok].copy()
        out["P(Bot)"] = model.predict_proba(Xn.loc[ok].astype(np.float32))[:, 1]
        out["predicted"] = np.where(out["P(Bot)"] >= sel["decision_threshold"], "Bot", "BENIGN")
        st.caption(f"Scored {int(ok.sum()):,} rows; skipped {int((~ok).sum()):,} rows with NaN/inf.")
        st.dataframe(out[[c for c in ("Label", "predicted", "P(Bot)", C.PORT) if c in out.columns]].head(500),
                     column_config={"P(Bot)": st.column_config.NumberColumn(format="%.4f")})
