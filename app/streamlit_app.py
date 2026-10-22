"""Research demonstration dashboard (entry point).

Run from the project root:  .venv\\Scripts\\streamlit run app/streamlit_app.py
Pages live in app/app_pages/; shared loaders in app/common.py. The dashboard shows recorded
experiment results and demonstrates the saved model on stored or uploaded flow records. It is not a
live network monitor or an intrusion-prevention system.
"""
import streamlit as st

st.set_page_config(page_title="Minority Bot traffic detection", page_icon=":material/hub:", layout="wide")

pages = {
    "Study": [
        st.Page("app_pages/overview.py", title="Research overview", icon=":material/menu_book:", default=True),
        st.Page("app_pages/dataset.py", title="Dataset", icon=":material/database:"),
        st.Page("app_pages/methodology.py", title="Methodology", icon=":material/account_tree:"),
    ],
    "Evidence": [
        st.Page("app_pages/comparison.py", title="Model comparison", icon=":material/leaderboard:"),
        st.Page("app_pages/selected_model.py", title="Selected model", icon=":material/verified:"),
        st.Page("app_pages/robustness.py", title="Robustness and limitations", icon=":material/rule:"),
    ],
    "Use": [
        st.Page("app_pages/predict.py", title="Prediction demonstration", icon=":material/target:"),
        st.Page("app_pages/reproducibility.py", title="Reproducibility and references", icon=":material/library_books:"),
    ],
}
st.navigation(pages).run()
