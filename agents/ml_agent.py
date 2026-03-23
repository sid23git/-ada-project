import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_squared_error, r2_score)
import xgboost as xgb

load_dotenv()
client = OpenAI()


def decide_problem_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Asks the AI to decide whether this is a
    classification or regression problem.
    This mimics how a data scientist looks at
    the target column and makes a judgement call.
    """

    target_info = {
        "column_name": target_col,
        "unique_values": int(df[target_col].nunique()),
        "sample_values": df[target_col].dropna().head(10).tolist(),
        "dtype": str(df[target_col].dtype)
    }

    prompt = f"""
You are an expert data scientist.
Look at this target column information and decide the problem type.

TARGET COLUMN INFO:
{json.dumps(target_info, indent=2)}

Respond ONLY with a JSON object in this exact format:
{{
    "problem_type": "classification" or "regression",
    "reasoning": "one sentence explanation"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a machine learning expert. Respond with valid JSON only."
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

    result = json.loads(raw.strip())
    return result


def prepare_features(df: pd.DataFrame, target_col: str):
    """
    Prepares X (features) and y (target) for ML.
    Handles encoding of categorical variables automatically.
    """

    df = df.copy()

    # Separate target
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Drop non-useful columns (IDs, names, tickets)
    cols_to_drop = []
    for col in X.columns:
        if X[col].dtype == "object" and X[col].nunique() > 50:
            cols_to_drop.append(col)
    X = X.drop(columns=cols_to_drop)
    if cols_to_drop:
        print(f"Dropped high-cardinality columns: {cols_to_drop}")

    # Encode categorical columns
    le = LabelEncoder()
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = le.fit_transform(X[col].astype(str))

    # Encode target if classification
    if y.dtype == "object":
        y = le.fit_transform(y.astype(str))

    return X, y, cols_to_drop


def train_and_evaluate(X, y, problem_type: str) -> dict:
    """
    Trains multiple models and compares them.
    Returns results for all models so the AI
    can reason about which one performed best.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    if problem_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": xgb.XGBClassifier(random_state=42,
                                          eval_metric='logloss',
                                          verbosity=0)
        }

        for name, model in models.items():
            if name == "Logistic Regression":
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X, y, cv=5)

            results[name] = {
                "accuracy": round(float(accuracy), 4),
                "cv_mean": round(float(cv_scores.mean()), 4),
                "cv_std": round(float(cv_scores.std()), 4)
            }
            print(f"{name}: Accuracy={accuracy:.4f}, CV={cv_scores.mean():.4f}")

    else:  # regression
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": xgb.XGBRegressor(random_state=42, verbosity=0)
        }

        for name, model in models.items():
            if name == "Linear Regression":
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            results[name] = {
                "r2_score": round(float(r2), 4),
                "rmse": round(float(rmse), 4)
            }
            print(f"{name}: R2={r2:.4f}, RMSE={rmse:.4f}")

    return results


def interpret_results(results: dict, problem_type: str) -> dict:
    """
    Asks AI to interpret the model results and
    recommend the best model with reasoning.
    This is the key research contribution —
    the agent doesn't just run models, it
    reasons about which one to use and why.
    """

    prompt = f"""
You are an expert data scientist reviewing ML model results.

PROBLEM TYPE: {problem_type}
MODEL RESULTS: {json.dumps(results, indent=2)}

Please analyze these results and respond ONLY with JSON:
{{
    "best_model": "model name",
    "reasoning": "why this model is best",
    "performance_summary": "one sentence on overall performance",
    "concerns": "any concerns about the results (overfitting, etc)",
    "recommendation": "what to try next to improve"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an ML expert. Respond with valid JSON only."
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


def run_ml_agent(df: pd.DataFrame, target_col: str = None) -> dict:
    """
    The full ML agent pipeline.
    Step 1 — AI decides problem type
    Step 2 — Prepare features
    Step 3 — Train and evaluate multiple models
    Step 4 — AI interprets results and recommends best model
    """

    print("\nML Agent starting...")

    # Step 1 — Decide target column
    if target_col is None:
        target_col = df.columns[-1]
        print(f"No target column specified. Using last column: '{target_col}'")

    # Step 2 — AI decides problem type
    print(f"Deciding problem type for target: '{target_col}'...")
    problem_info = decide_problem_type(df, target_col)
    problem_type = problem_info["problem_type"]
    print(f"Problem type: {problem_type}")
    print(f"Reasoning: {problem_info['reasoning']}")

    # Step 3 — Prepare features
    print("\nPreparing features...")
    X, y, dropped = prepare_features(df, target_col)
    print(f"Features shape: {X.shape}")

    # Step 4 — Train models
    print(f"\nTraining models for {problem_type}...")
    results = train_and_evaluate(X, y, problem_type)

    # Step 5 — AI interprets results
    print("\nAsking AI to interpret results...")
    interpretation = interpret_results(results, problem_type)

    print(f"\n--- ML Summary ---")
    print(f"Best model:  {interpretation['best_model']}")
    print(f"Reasoning:   {interpretation['reasoning']}")
    print(f"Performance: {interpretation['performance_summary']}")
    print(f"Concerns:    {interpretation['concerns']}")
    print(f"Next steps:  {interpretation['recommendation']}")

    return {
        "problem_type": problem_type,
        "target_column": target_col,
        "model_results": results,
        "interpretation": interpretation
    }