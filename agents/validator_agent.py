import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def validate_hypotheses(hypotheses: dict,
                        eda_stats: dict,
                        ml_results: dict,
                        explain_results: dict) -> dict:
    """
    Tests each hypothesis against actual findings.
    This closes the scientific loop — we formed hypotheses
    before analysis, now we check which ones were right.

    This is the most novel part of ADA v3.0.
    No existing AutoML tool does hypothesis validation.
    """

    # Build a summary of findings to send to the AI
    findings = {
        "eda": {
            "missing_values": eda_stats.get("missing_values", {}),
            "numeric_summary": list(
                eda_stats.get("numeric_summary", {}).keys()
            ),
            "categorical_summary": eda_stats.get(
                "categorical_summary", {}
            )
        },
        "ml": {
            "problem_type": ml_results.get("problem_type", ""),
            "best_model": ml_results.get(
                "interpretation", {}
            ).get("best_model", ""),
            "model_results": ml_results.get("model_results", {}),
            "concerns": ml_results.get(
                "interpretation", {}
            ).get("concerns", "")
        },
        "explanation": {
            "top_features": list(
                explain_results.get(
                    "feature_importance", {}
                ).keys()
            )[:5],
            "plain_english": explain_results.get(
                "interpretation", {}
            ).get("plain_english_summary", "")
        }
    }

    prompt = f"""
You are an expert data scientist validating hypotheses
against actual analysis findings.

ORIGINAL HYPOTHESES (formed before analysis):
{json.dumps(hypotheses.get('hypotheses', []), indent=2)}

ACTUAL FINDINGS FROM ANALYSIS:
{json.dumps(findings, indent=2)}

For each hypothesis, determine if it was CONFIRMED,
REJECTED, or INCONCLUSIVE based on the findings.

Respond ONLY with valid JSON:
{{
    "validation_results": [
        {{
            "id": "H1",
            "hypothesis": "original hypothesis text",
            "verdict": "CONFIRMED/REJECTED/INCONCLUSIVE",
            "evidence": "specific finding that supports this verdict",
            "insight": "what this tells us about the data"
        }},
        ...one entry per hypothesis...
    ],
    "overall_summary": "2-3 sentence summary of what was learned",
    "most_surprising": "which hypothesis result was most unexpected",
    "scientific_contribution": "what these findings add to understanding"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a data science expert validating hypotheses. Be rigorous and evidence-based. Respond with valid JSON only."
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

    # Print validation results clearly
    print("\n--- Hypothesis Validation Results ---")
    confirmed = 0
    rejected = 0
    inconclusive = 0

    for v in result["validation_results"]:
        icon = "✅" if v["verdict"] == "CONFIRMED" else \
               "❌" if v["verdict"] == "REJECTED" else "⚠️"
        print(f"{icon} {v['id']}: {v['verdict']}")
        print(f"   Evidence: {v['evidence'][:100]}...")

        if v["verdict"] == "CONFIRMED":
            confirmed += 1
        elif v["verdict"] == "REJECTED":
            rejected += 1
        else:
            inconclusive += 1

    print(f"\nResults: {confirmed} confirmed, "
          f"{rejected} rejected, "
          f"{inconclusive} inconclusive")
    print(f"\nSummary: {result['overall_summary']}")

    return result


def run_validator_agent(hypotheses: dict,
                        eda_stats: dict,
                        ml_results: dict,
                        explain_results: dict) -> dict:
    """
    Entry point for the validator agent.
    Called after all analysis is complete.
    """
    print("\nValidator Agent starting...")
    print("Testing hypotheses against findings...")

    if not hypotheses or not hypotheses.get("hypotheses"):
        print("No hypotheses to validate — skipping.")
        return {}

    validation = validate_hypotheses(
        hypotheses, eda_stats, ml_results, explain_results
    )

    print("\nValidator Agent complete.")
    return validation