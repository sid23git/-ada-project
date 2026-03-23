import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()


def get_cleaning_strategy(stats: dict) -> dict:
    """
    Asks GPT to decide the best cleaning strategy
    based on the EDA statistics.
    This is the 'reasoning' step — AI decides WHAT to do.
    """

    prompt = f"""
You are an expert data scientist deciding how to clean a dataset.
Based on the following dataset statistics, provide a cleaning strategy.

DATASET STATISTICS:
{json.dumps(stats, indent=2)}

Respond ONLY with a valid JSON object in this exact format:
{{
    "missing_value_strategies": {{
        "column_name": "strategy"
    }},
    "columns_to_drop": ["col1", "col2"],
    "reasoning": "brief explanation of your decisions"
}}

Valid strategies for missing values are:
- "mean" (for numeric columns with normal distribution)
- "median" (for numeric columns with skew or outliers)
- "mode" (for categorical columns)
- "drop_rows" (if very few rows affected, under 5%)
- "drop_column" (if more than 60% values are missing)
- "constant_unknown" (for categorical where unknown is meaningful)

Be decisive and practical.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a data cleaning expert. Always respond with valid JSON only. No explanation text outside the JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    raw = response.choices[0].message.content

    # Strip markdown code fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    strategy = json.loads(raw.strip())
    return strategy


def apply_cleaning_strategy(df: pd.DataFrame, strategy: dict) -> pd.DataFrame:
    """
    Actually applies the cleaning strategy to the dataframe.
    This is the 'execution' step — code does WHAT AI decided.
    Notice: AI reasons, Python executes. They have separate jobs.
    """

    df = df.copy()  # never modify original data — best practice
    # Drop auto-generated index columns — common in exported CSVs
    unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        print(f"Dropped auto-index columns: {unnamed_cols}")

    print("\n--- Applying cleaning strategy ---")

    # Drop columns first
    cols_to_drop = strategy.get("columns_to_drop", [])
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Dropped column: {col}")

    # Handle missing values
    strategies = strategy.get("missing_value_strategies", {})
    for col, method in strategies.items():
        if col not in df.columns:
            continue

        missing_count = df[col].isnull().sum()
        if missing_count == 0:
            continue

        if method == "mean":
            fill_val = df[col].mean()
            df[col] = df[col].fillna(fill_val)
            print(f"Filled '{col}' with mean ({fill_val:.2f})")

        elif method == "median":
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            print(f"Filled '{col}' with median ({fill_val:.2f})")

        elif method == "mode":
            fill_val = df[col].mode()[0]
            df[col] = df[col].fillna(fill_val)
            print(f"Filled '{col}' with mode ({fill_val})")

        elif method == "drop_rows":
            before = len(df)
            df = df.dropna(subset=[col])
            print(f"Dropped {before - len(df)} rows with missing '{col}'")

        elif method == "drop_column":
            df = df.drop(columns=[col])
            print(f"Dropped column '{col}' (too many missing values)")

        elif method == "constant_unknown":
            df[col] = df[col].fillna("Unknown")
            print(f"Filled '{col}' with 'Unknown'")

    # Remove duplicate rows — always a good practice
    before = len(df)
    df = df.drop_duplicates()
    duplicates_removed = before - len(df)
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate rows")

    return df


def run_cleaning_agent(df: pd.DataFrame, eda_stats: dict) -> tuple:
    """
    The full cleaning agent pipeline.
    Returns both the cleaned dataframe AND the strategy used.
    Returning the strategy is important for the audit trail
    and for your research paper — you can show exactly what
    decisions the AI made and why.
    """

    print("\nCleaning Agent starting...")

    # Step 1 — AI reasons about what cleaning is needed
    print("Asking AI for cleaning strategy...")
    strategy = get_cleaning_strategy(eda_stats)

    print(f"\nAI Reasoning: {strategy.get('reasoning', 'N/A')}")

    # Step 2 — Python applies the strategy
    cleaned_df = apply_cleaning_strategy(df, strategy)

    # Step 3 — Report what changed
    print(f"\n--- Cleaning Summary ---")
    print(f"Original shape:           {df.shape}")
    print(f"Cleaned shape:            {cleaned_df.shape}")
    print(f"Rows removed:             {df.shape[0] - cleaned_df.shape[0]}")
    print(f"Columns removed:          {df.shape[1] - cleaned_df.shape[1]}")
    print(f"Missing values remaining: {cleaned_df.isnull().sum().sum()}")

    return cleaned_df, strategy
