import json
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
client = OpenAI()


def generate_hypotheses(df: pd.DataFrame, target_col: str = None) -> dict:
    """
    Generates testable hypotheses BEFORE running any analysis.
    This is the key research contribution of v3.0 —
    the agent reasons about what it EXPECTS to find,
    then later we check if it was right.

    Why this matters for the paper:
    This mimics the scientific method — form a hypothesis,
    then test it. No existing AutoML tool does this.
    """

    # Give the AI just enough to form hypotheses
    # without revealing the actual data values
    column_info = {
        col: {
            "dtype": str(df[col].dtype),
            "unique_values": int(df[col].nunique()),
            "sample_values": df[col].dropna().head(5).tolist()
        }
        for col in df.columns
    }

    target_info = f"Target column: '{target_col}'" if target_col else \
        f"Likely target: '{df.columns[-1]}' (last column)"

    prompt = f"""
You are an expert data scientist about to analyze a dataset.
Before running any analysis, form 5 testable hypotheses
about what you expect to find.

DATASET COLUMN INFORMATION:
{json.dumps(column_info, indent=2)}

{target_info}

Based ONLY on the column names, types, and sample values
(not actual statistics), generate 5 hypotheses.

Respond ONLY with a valid JSON object:
{{
    "dataset_type": "one sentence describing what this dataset is about",
    "hypotheses": [
        {{
            "id": "H1",
            "hypothesis": "clear testable statement",
            "reasoning": "why you think this based on column names/types",
            "expected_evidence": "what finding would confirm this",
            "confidence": "high/medium/low"
        }},
        ...5 total hypotheses...
    ],
    "most_important_hypothesis": "H1/H2/H3/H4/H5",
    "analysis_strategy": "brief note on what to focus on"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a data science expert. Form clear, testable hypotheses. Respond with valid JSON only."
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

    # Print hypotheses clearly
    print(f"\nDataset identified as: {result['dataset_type']}")
    print(f"\nGenerated {len(result['hypotheses'])} hypotheses:")
    for h in result["hypotheses"]:
        print(f"  {h['id']} [{h['confidence']}]: {h['hypothesis']}")
    print(f"\nMost important: {result['most_important_hypothesis']}")
    print(f"Strategy: {result['analysis_strategy']}")

    return result


def run_hypothesis_agent(df: pd.DataFrame,
                         target_col: str = None) -> dict:
    """
    Entry point for the hypothesis agent.
    Called before EDA so hypotheses are formed
    without bias from seeing the actual statistics.
    """
    print("\nHypothesis Agent starting...")
    print("Forming hypotheses before analysis...")

    hypotheses = generate_hypotheses(df, target_col)

    print("\nHypothesis Agent complete.")
    return hypotheses