import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
from agents.orchestrator import ADAOrchestrator

# Prevent double runs
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False

# ─── Page Config ──────────────────────────────────────
st.set_page_config(
    page_title="ADA — Autonomous Data Analysis Agent",
    page_icon="🤖",
    layout="wide"
)

# ─── Header ───────────────────────────────────────────
st.title("🤖 ADA — Autonomous Data Analysis Agent")
st.markdown(
    "Upload any CSV dataset and ADA will autonomously "
    "analyze, clean, model, and explain it — just like a junior data scientist."
)
st.divider()

# ─── Sidebar ──────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    uploaded_file = st.file_uploader(
        "Upload your CSV dataset",
        type=["csv"]
    )

    target_col_input = st.text_input(
        "Target column (optional)",
        placeholder="e.g. Survived, Price, Churn",
        help="Leave empty and ADA will use the last column"
    )

    run_button = st.button(
        "▶ Run ADA Pipeline",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None or st.session_state.pipeline_running
    )

    st.divider()
    st.markdown("**Pipeline stages**")
    st.markdown("1. 🔍 Exploratory Analysis")
    st.markdown("2. 🧹 Data Cleaning")
    st.markdown("3. 🤖 Model Training")
    st.markdown("4. 💡 SHAP Explanation")
    st.markdown("5. 📄 Final Report")

# ─── Main Area ────────────────────────────────────────
if uploaded_file is None:
    st.info("👈 Upload a CSV file from the sidebar to get started.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🏥 Healthcare")
        st.markdown("Predict patient outcomes, identify risk factors")
    with col2:
        st.markdown("### 💰 Finance")
        st.markdown("Detect fraud, predict loan defaults")
    with col3:
        st.markdown("### 🛒 Retail")
        st.markdown("Predict churn, forecast sales")

elif not run_button:
    df_preview = pd.read_csv(uploaded_file)
    uploaded_file.seek(0)

    st.subheader("📋 Dataset Preview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df_preview.shape[0])
    col2.metric("Columns", df_preview.shape[1])
    col3.metric("Missing Values", int(df_preview.isnull().sum().sum()))
    col4.metric("Numeric Columns",
                len(df_preview.select_dtypes(include="number").columns))

    st.dataframe(df_preview.head(20), use_container_width=True)
    st.caption("Showing first 20 rows. Press 'Run ADA Pipeline' to start analysis.")

else:
    temp_path = f"data/uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    target = target_col_input.strip() if target_col_input.strip() else None

    st.session_state.pipeline_running = True
    progress = st.progress(0)
    status = st.status("🚀 ADA pipeline starting...", expanded=True)

    try:
        with status:
            st.write("🔍 Running Exploratory Data Analysis...")
            progress.progress(10)

            ada = ADAOrchestrator(filepath=temp_path, target_col=target)

            st.write("🧹 Cleaning data...")
            progress.progress(30)

            st.write("🤖 Training ML models...")
            progress.progress(50)

            st.write("💡 Computing SHAP explanations...")
            progress.progress(70)

            st.write("📄 Generating final report...")
            progress.progress(90)

            results = ada.run_pipeline()
            progress.progress(100)

        status.update(label="✅ ADA Pipeline Complete!", state="complete")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 EDA", "🧹 Cleaning", "🤖 ML Results",
            "💡 Explanation", "📄 Final Report"
        ])

        # Tab 1 — EDA
        with tab1:
            st.subheader("Exploratory Data Analysis")
            eda_stats = results.get("eda_stats", {})

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Dataset Shape**")
                shape = eda_stats.get("shape", {})
                st.metric("Rows", shape.get("rows", "N/A"))
                st.metric("Columns", shape.get("columns", "N/A"))
            with col2:
                st.markdown("**Missing Values**")
                missing = eda_stats.get("missing_values", {})
                if missing:
                    missing_df = pd.DataFrame(
                        list(missing.items()),
                        columns=["Column", "Missing Count"]
                    )
                    st.dataframe(missing_df, use_container_width=True)
                else:
                    st.success("No missing values found!")

            st.markdown("**AI EDA Report**")
            st.markdown(results.get("eda_report", ""))

        # Tab 2 — Cleaning
        with tab2:
            st.subheader("Data Cleaning")
            strategy = results.get("cleaning_strategy", {})

            st.markdown("**AI Reasoning**")
            st.info(strategy.get("reasoning", "N/A"))

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Original Shape", str(results.get("raw_shape", "N/A")))
            with col2:
                st.metric("Cleaned Shape", str(results.get("cleaned_shape", "N/A")))

            if strategy.get("columns_to_drop"):
                st.markdown("**Columns Dropped**")
                st.write(strategy.get("columns_to_drop"))

            if strategy.get("missing_value_strategies"):
                st.markdown("**Imputation Strategies Applied**")
                strat_df = pd.DataFrame(
                    list(strategy["missing_value_strategies"].items()),
                    columns=["Column", "Strategy"]
                )
                st.dataframe(strat_df, use_container_width=True)

        # Tab 3 — ML Results
        with tab3:
            st.subheader("Machine Learning Results")
            ml = results.get("ml_results", {})
            interp = ml.get("interpretation", {})

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Problem Type",
                          ml.get("problem_type", "N/A").capitalize())
                st.metric("Best Model", interp.get("best_model", "N/A"))
            with col2:
                st.markdown("**AI Reasoning**")
                st.info(interp.get("reasoning", "N/A"))

            st.markdown("**Model Comparison**")
            model_results = ml.get("model_results", {})
            if model_results:
                model_df = pd.DataFrame(model_results).T
                st.dataframe(model_df, use_container_width=True)

            if interp.get("concerns"):
                st.warning(f"⚠️ {interp.get('concerns')}")

            if interp.get("recommendation"):
                st.markdown("**Next Steps**")
                st.success(interp.get("recommendation"))

        # Tab 4 — Explanation
        with tab4:
            st.subheader("Model Explanation (SHAP)")
            explain = results.get("explain_results", {})
            exp_interp = explain.get("interpretation", {})

            st.markdown("**Plain English Summary**")
            st.info(exp_interp.get("plain_english_summary", "N/A"))

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top 3 Most Influential Features**")
                top_feats = exp_interp.get("top_3_features", [])
                for i, feat in enumerate(top_feats, 1):
                    st.markdown(
                        f"**{i}. {feat.get('feature', '')}** — "
                        f"{feat.get('explanation', '')}"
                    )
            with col2:
                st.markdown("**Feature Importance (SHAP)**")
                importance = explain.get("feature_importance", {})
                if importance:
                    imp_df = pd.DataFrame(
                        list(importance.items())[:10],
                        columns=["Feature", "SHAP Importance"]
                    )
                    st.dataframe(imp_df, use_container_width=True)

            shap_path = "outputs/shap_summary.png"
            if os.path.exists(shap_path):
                st.markdown("**SHAP Summary Plot**")
                st.image(shap_path, use_container_width=True)

            if exp_interp.get("business_insight"):
                st.markdown("**Business Insight**")
                st.success(exp_interp.get("business_insight"))

        # Tab 5 — Final Report
        with tab5:
            st.subheader("Final Report")
            report = results.get("final_report", "")
            st.markdown(report)

            st.download_button(
                label="⬇️ Download Report",
                data=report,
                file_name=f"ADA_report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

    except Exception as e:
        status.update(label="❌ Pipeline failed", state="error")
        st.error(f"Error: {str(e)}")
        st.exception(e)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        st.session_state.pipeline_running = False
