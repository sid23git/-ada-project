import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — saves files instead of showing popups
import matplotlib.pyplot as plt

load_dotenv()
client = OpenAI()


def train_best_model(df: pd.DataFrame, target_col: str,
                     problem_type: str, best_model_name: str):
    """
    Retrains the best model identified by the ML agent.
    We need the actual trained model object for SHAP.
    """

    df = df.copy()
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Drop high cardinality columns
    cols_to_drop = [col for col in X.columns
                    if X[col].dtype == "object" and X[col].nunique() > 50]
    X = X.drop(columns=cols_to_drop)

    # Encode categoricals
    le = LabelEncoder()
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = le.fit_transform(X[col].astype(str))

    if y.dtype == "object":
        y = le.fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the best model
    if best_model_name == "Random Forest":
        if problem_type == "classification":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        if problem_type == "classification":
            model = xgb.XGBClassifier(random_state=42,
                                       eval_metric='logloss',
                                       verbosity=0)
        else:
            model = xgb.XGBRegressor(random_state=42, verbosity=0)

    model.fit(X_train, y_train)

    return model, X_train, X_test, X.columns.tolist()


def compute_shap_values(model, X_train, X_test):
    """
    Computes SHAP values using TreeExplainer.
    TreeExplainer works for both Random Forest and XGBoost
    and is the fastest and most accurate SHAP method for
    tree-based models.
    """

    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # For binary classification, Random Forest returns
    # shap values for both classes — we take class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return explainer, shap_values


def get_feature_importance_summary(shap_values, feature_names: list) -> dict:
    """
    Converts raw SHAP values into a clean summary
    that can be sent to the AI for interpretation.
    """

    # Handle all possible SHAP value shapes
    if isinstance(shap_values, list):
        # Random Forest binary classification returns list of 2 arrays
        shap_array = np.array(shap_values[1])
    else:
        shap_array = np.array(shap_values)

    # If 3D array (samples, features, classes), take first class
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 0]

    # Now safely compute mean absolute importance per feature
    mean_shap = np.abs(shap_array).mean(axis=0).flatten()

    importance_dict = {
        feature: round(float(importance), 4)
        for feature, importance in zip(feature_names, mean_shap)
    }

    # Sort by importance descending
    importance_dict = dict(
        sorted(importance_dict.items(),
               key=lambda x: x[1], reverse=True)
    )

    return importance_dict

def save_shap_plot(shap_values, X_test, feature_names: list):
    """
    Saves a SHAP summary plot to the outputs folder.
    """

    # Handle all possible SHAP value shapes
    if isinstance(shap_values, list):
        shap_array = np.array(shap_values[1])
    else:
        shap_array = np.array(shap_values)

    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 0]

    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_array,
        X_test_df,
        show=False,
        plot_size=None
    )
    plt.tight_layout()
    plt.savefig("outputs/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP plot saved to outputs/shap_summary.png")


def interpret_shap_with_ai(importance_dict: dict,
                            problem_type: str,
                            target_col: str) -> dict:
    """
    Sends SHAP feature importance to GPT and asks it
    to explain the results in plain English.
    This is what makes the explanation truly useful —
    not just numbers, but meaning.
    """

    prompt = f"""
You are an expert data scientist explaining model predictions to a non-technical audience.

TARGET VARIABLE: {target_col}
PROBLEM TYPE: {problem_type}

FEATURE IMPORTANCE (SHAP values — higher means more influential):
{json.dumps(importance_dict, indent=2)}

Please respond ONLY with a JSON object in this format:
{{
    "plain_english_summary": "2-3 sentence explanation of what drives predictions",
    "top_3_features": [
        {{
            "feature": "feature name",
            "impact": "positive or negative",
            "explanation": "what this means in plain English"
        }}
    ],
    "surprising_findings": "anything unexpected or interesting",
    "business_insight": "practical actionable insight from these findings"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an AI explainability expert. Make complex ML results understandable. Respond with valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())


def run_explain_agent(df: pd.DataFrame,
                      target_col: str,
                      problem_type: str,
                      best_model_name: str) -> dict:
    """
    Full explanation agent pipeline.
    Step 1 — Retrain best model
    Step 2 — Compute SHAP values
    Step 3 — Extract feature importance summary
    Step 4 — Save SHAP plot
    Step 5 — AI interprets results in plain English
    """

    print("\nExplanation Agent starting...")

    # Step 1 — Retrain best model
    print(f"Retraining {best_model_name} for explanation...")
    model, X_train, X_test, feature_names = train_best_model(
        df, target_col, problem_type, best_model_name
    )

    # Step 2 — Compute SHAP values
    explainer, shap_values = compute_shap_values(model, X_train, X_test)

    # Step 3 — Get feature importance summary
    importance_dict = get_feature_importance_summary(
        shap_values, feature_names
    )
    print("\nFeature importance (SHAP):")
    for feat, val in list(importance_dict.items())[:5]:
        print(f"  {feat}: {val}")

    # Step 4 — Save SHAP plot
    save_shap_plot(shap_values, X_test, feature_names)

    # Step 5 — AI interprets in plain English
    print("\nAsking AI to interpret SHAP results...")
    interpretation = interpret_shap_with_ai(
        importance_dict, problem_type, target_col
    )

    print("\n--- Explanation Summary ---")
    print(f"Summary: {interpretation['plain_english_summary']}")
    print(f"\nTop features:")
    for feat in interpretation['top_3_features']:
        print(f"  - {feat['feature']}: {feat['explanation']}")
    print(f"\nSurprising finding: {interpretation['surprising_findings']}")
    print(f"Business insight:   {interpretation['business_insight']}")

    return {
        "feature_importance": importance_dict,
        "interpretation": interpretation,
        "shap_plot_path": "outputs/shap_summary.png"
    }