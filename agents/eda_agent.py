import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def analyze_dataframe(df: pd.DataFrame) -> dict:
    """
    Extracts raw statistics from the dataframe.
    This is pure Python/Pandas — no AI involved yet.
    Think of this as gathering evidence before asking
    the AI to interpret it.
    """
    analysis = {}

    # Basic shape
    analysis["shape"] = {
        "rows": df.shape[0],
        "columns": df.shape[1]
    }

    # Column types
    analysis["column_types"] = df.dtypes.astype(str).to_dict()

    # Missing values
    missing = df.isnull().sum()
    analysis["missing_values"] = {
        col: int(count)
        for col, count in missing.items()
        if count > 0
    }

    # Basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        analysis["numeric_summary"] = df[numeric_cols].describe().to_dict()

    # Unique value counts for categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    analysis["categorical_summary"] = {
        col: df[col].nunique()
        for col in categorical_cols
    }

    # Detect potential target column (last column heuristic)
    analysis["potential_target"] = df.columns[-1]

    return analysis


def run_eda_agent(filepath: str) -> str:
    """
    The actual EDA Agent.
    Step 1 — Load data
    Step 2 — Extract statistics using Pandas
    Step 3 — Send stats to GPT and ask it to reason about them
    Step 4 — Return the AI generated EDA report
    """

    # Step 1 — Load the dataset
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Step 2 — Extract raw statistics
    print("Extracting statistics...")
    stats = analyze_dataframe(df)

    # Step 3 — Ask GPT to reason about the statistics
    print("Running AI analysis...")

    prompt = f"""
You are an expert data scientist performing Exploratory Data Analysis (EDA).
You have been given the following statistics about a dataset.
Your job is to analyze these statistics and produce a professional EDA report.

DATASET STATISTICS:
{stats}

Please provide:
1. Dataset overview (what kind of data this appears to be)
2. Data quality assessment (missing values, potential issues)
3. Key observations about the numeric columns
4. Key observations about the categorical columns  
5. Hypothesis — what problem is this dataset likely trying to solve?
6. Recommended next steps for data cleaning

Be specific, insightful, and professional. Format your response clearly.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert data scientist. You analyze datasets and provide professional, actionable insights."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    report = response.choices[0].message.content

    # Step 4 — Return the report
    return report