import pandas as pd
import json
import os
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

from agents.eda_agent import analyze_dataframe, run_eda_agent
from agents.cleaning_agent import run_cleaning_agent
from agents.ml_agent import run_ml_agent
from agents.explain_agent import run_explain_agent

load_dotenv()
client = OpenAI()


class ADAOrchestrator:
    """
    The Orchestrator is the central controller of ADA.
    It manages the full pipeline, tracks state across
    all agents, and assembles the final report.

    Using a class here instead of just functions is
    intentional — it lets us store the pipeline state
    (audit trail, results, timestamps) as the run
    progresses. This is important for the research paper
    because you can show exactly what happened at each step.
    """

    def __init__(self, filepath: str, target_col: str = None):
        self.filepath = filepath
        self.target_col = target_col
        self.audit_trail = []  # every decision logged here
        self.results = {}      # all agent outputs stored here
        self.start_time = datetime.now()

        print("=" * 55)
        print("   ADA — Autonomous Data Analysis Agent")
        print("=" * 55)
        print(f"Dataset:   {filepath}")
        print(f"Started:   {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 55)

    def log(self, agent: str, action: str, detail: str = ""):
        """
        Logs every decision made during the pipeline.
        This audit trail is a key research contribution —
        you can show reviewers exactly how the agent
        reasoned at each step.
        """
        entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "agent": agent,
            "action": action,
            "detail": detail
        }
        self.audit_trail.append(entry)
        print(f"\n[{entry['timestamp']}] [{agent}] {action}")
        if detail:
            print(f"           {detail}")

    def run_pipeline(self) -> dict:
        """
        Runs the full ADA pipeline in order.
        Each stage feeds its output into the next.
        If any stage fails, the error is logged
        and the pipeline stops gracefully.
        """

        try:
            # ─── Stage 1: Load Data ───────────────────────────
            self.log("Orchestrator", "Loading dataset...")
            df = pd.read_csv(self.filepath)
            self.results["raw_shape"] = df.shape
            self.log("Orchestrator", "Dataset loaded",
                     f"{df.shape[0]} rows × {df.shape[1]} columns")

            # ─── Stage 2: EDA ─────────────────────────────────
            self.log("EDA Agent", "Starting exploratory analysis...")
            eda_stats = analyze_dataframe(df)
            eda_report = run_eda_agent(self.filepath)
            self.results["eda_stats"] = eda_stats
            self.results["eda_report"] = eda_report
            self.log("EDA Agent", "EDA complete",
                     f"Found {len(eda_stats.get('missing_values', {}))} columns with missing values")

            # ─── Stage 3: Cleaning ────────────────────────────
            self.log("Cleaning Agent", "Starting data cleaning...")
            cleaned_df, cleaning_strategy = run_cleaning_agent(df, eda_stats)
            self.results["cleaned_shape"] = cleaned_df.shape
            self.results["cleaning_strategy"] = cleaning_strategy
            self.log("Cleaning Agent", "Cleaning complete",
                     f"Shape: {df.shape} → {cleaned_df.shape}")

            # Save cleaned data
            cleaned_path = self.filepath.replace(".csv", "_cleaned.csv")
            cleaned_df.to_csv(cleaned_path, index=False)

            # ─── Stage 4: ML ──────────────────────────────────
            self.log("ML Agent", "Starting model training...")
            ml_results = run_ml_agent(cleaned_df, target_col=self.target_col)
            self.results["ml_results"] = ml_results
            best_model = ml_results["interpretation"]["best_model"]
            problem_type = ml_results["problem_type"]
            self.log("ML Agent", "Training complete",
                     f"Best model: {best_model}")

            # ─── Stage 5: Explanation ─────────────────────────
            self.log("Explain Agent", "Starting SHAP analysis...")
            explain_results = run_explain_agent(
                df=cleaned_df,
                target_col=ml_results["target_column"],
                problem_type=problem_type,
                best_model_name=best_model
            )
            self.results["explain_results"] = explain_results
            self.log("Explain Agent", "Explanation complete",
                     f"Top feature: {list(explain_results['feature_importance'].keys())[0]}")

            # ─── Stage 6: Final Report ────────────────────────
            self.log("Orchestrator", "Generating final report...")
            report = self.generate_final_report()
            self.results["final_report"] = report

            # Save report
            report_path = f"outputs/ADA_report_{self.start_time.strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_path, "w") as f:
                f.write(report)

            # Save audit trail
            audit_path = f"outputs/audit_trail_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(audit_path, "w") as f:
                json.dump(self.audit_trail, f, indent=2)

            end_time = datetime.now()
            duration = (end_time - self.start_time).seconds

            print("\n" + "=" * 55)
            print("   ADA Pipeline Complete!")
            print("=" * 55)
            print(f"Duration:     {duration} seconds")
            print(f"Report saved: {report_path}")
            print(f"Audit trail:  {audit_path}")
            print(f"SHAP plot:    outputs/shap_summary.png")
            print("=" * 55)

            return self.results

        except Exception as e:
            self.log("Orchestrator", "PIPELINE ERROR", str(e))
            raise e

    def generate_final_report(self) -> str:
        """
        Asks GPT to synthesize ALL agent findings into
        one coherent professional report.
        This is the final deliverable — what a junior
        data scientist would hand to their manager.
        """

        summary = {
            "dataset": {
                "original_shape": str(self.results.get("raw_shape")),
                "cleaned_shape": str(self.results.get("cleaned_shape"))
            },
            "cleaning": {
                "reasoning": self.results.get(
                    "cleaning_strategy", {}).get("reasoning", "")
            },
            "ml": {
                "problem_type": self.results.get(
                    "ml_results", {}).get("problem_type", ""),
                "best_model": self.results.get(
                    "ml_results", {}).get(
                    "interpretation", {}).get("best_model", ""),
                "model_results": self.results.get(
                    "ml_results", {}).get("model_results", {}),
                "performance_summary": self.results.get(
                    "ml_results", {}).get(
                    "interpretation", {}).get("performance_summary", ""),
                "concerns": self.results.get(
                    "ml_results", {}).get(
                    "interpretation", {}).get("concerns", "")
            },
            "explanation": {
                "plain_english": self.results.get(
                    "explain_results", {}).get(
                    "interpretation", {}).get("plain_english_summary", ""),
                "top_features": self.results.get(
                    "explain_results", {}).get(
                    "interpretation", {}).get("top_3_features", []),
                "business_insight": self.results.get(
                    "explain_results", {}).get(
                    "interpretation", {}).get("business_insight", "")
            }
        }

        prompt = f"""
You are a senior data scientist writing a professional analysis report.
Synthesize these findings from an automated analysis pipeline into a
clear, well-structured report suitable for both technical and
non-technical audiences.

PIPELINE FINDINGS:
{json.dumps(summary, indent=2)}

Write a complete report with these sections:
1. Executive Summary (3-4 sentences, non-technical)
2. Dataset Overview
3. Data Quality & Cleaning
4. Model Performance
5. Key Insights & Explainability
6. Recommendations & Next Steps

Be professional, specific, and insightful.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior data scientist writing professional analysis reports."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1500
        )

        return response.choices[0].message.content