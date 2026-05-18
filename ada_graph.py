import pandas as pd
from datetime import datetime
from typing import Literal
from langgraph.graph import StateGraph, END
from utils.ada_state import ADAState
from agents.eda_agent import analyze_dataframe, run_eda_agent
from agents.cleaning_agent import run_cleaning_agent
from agents.ml_agent import run_ml_agent
from agents.explain_agent import run_explain_agent
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()


# ─────────────────────────────────────────────────
# HELPER — Logging
# ─────────────────────────────────────────────────

def log(state: ADAState, agent: str, action: str, detail: str = "") -> list:
    """
    Appends a log entry to the audit trail.
    Returns the updated audit trail list.
    Every node calls this to record what it did.
    """
    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "agent": agent,
        "action": action,
        "detail": detail
    }
    print(f"[{entry['timestamp']}] [{agent}] {action}")
    if detail:
        print(f"           {detail}")

    trail = state.get("audit_trail", [])
    return trail + [entry]


# ─────────────────────────────────────────────────
# NODE 1 — Load Data
# ─────────────────────────────────────────────────

def load_data_node(state: ADAState) -> ADAState:
    """
    Loads the CSV file into a dataframe.
    Critical node — if this fails, nothing else can run.
    """
    try:
        trail = log(state, "LoadData", "Loading dataset...",
                    state["filepath"])

        df = pd.read_csv(state["filepath"])

        trail = log({"audit_trail": trail}, "LoadData",
                    "Dataset loaded",
                    f"{df.shape[0]} rows × {df.shape[1]} columns")

        return {
            **state,
            "raw_df": df,
            "current_node": "load_data",
            "audit_trail": trail,
            "error": None,
            "retry_count": 0
        }

    except Exception as e:
        trail = log(state, "LoadData", "FAILED", str(e))
        return {
            **state,
            "current_node": "load_data",
            "audit_trail": trail,
            "error": f"load_data: {str(e)}"
        }


# ─────────────────────────────────────────────────
# NODE 2 — EDA
# ─────────────────────────────────────────────────

def eda_node(state: ADAState) -> ADAState:
    """
    Runs exploratory data analysis.
    Uses the raw_df from state — doesn't reload the file.
    """
    try:
        trail = log(state, "EDA", "Starting exploratory analysis...")

        df = state["raw_df"]
        eda_stats = analyze_dataframe(df)
        eda_report = run_eda_agent(state["filepath"])

        missing_count = len(eda_stats.get("missing_values", {}))
        trail = log({"audit_trail": trail}, "EDA",
                    "EDA complete",
                    f"Found {missing_count} columns with missing values")

        return {
            **state,
            "eda_stats": eda_stats,
            "eda_report": eda_report,
            "current_node": "eda",
            "audit_trail": trail,
            "error": None
        }

    except Exception as e:
        trail = log(state, "EDA", "FAILED", str(e))
        return {
            **state,
            "current_node": "eda",
            "audit_trail": trail,
            "error": f"eda: {str(e)}"
        }


# ─────────────────────────────────────────────────
# NODE 3 — Cleaning
# ─────────────────────────────────────────────────

def cleaning_node(state: ADAState) -> ADAState:
    """
    Cleans the raw dataframe based on EDA findings.
    Reads raw_df and eda_stats from state.
    Writes cleaned_df and cleaning_strategy back to state.
    """
    try:
        trail = log(state, "Cleaning", "Starting data cleaning...")

        df = state["raw_df"]
        eda_stats = state["eda_stats"]

        cleaned_df, strategy = run_cleaning_agent(df, eda_stats)

        trail = log({"audit_trail": trail}, "Cleaning",
                    "Cleaning complete",
                    f"Shape: {df.shape} → {cleaned_df.shape}")

        return {
            **state,
            "cleaned_df": cleaned_df,
            "cleaning_strategy": strategy,
            "current_node": "cleaning",
            "audit_trail": trail,
            "error": None
        }

    except Exception as e:
        trail = log(state, "Cleaning", "FAILED", str(e))
        return {
            **state,
            "current_node": "cleaning",
            "audit_trail": trail,
            "error": f"cleaning: {str(e)}"
        }


# ─────────────────────────────────────────────────
# NODE 4 — ML
# ─────────────────────────────────────────────────

def ml_node(state: ADAState) -> ADAState:
    """
    Trains and evaluates ML models.
    Reads cleaned_df from state.
    Writes ml_results back to state.
    """
    try:
        trail = log(state, "ML", "Starting model training...")

        cleaned_df = state["cleaned_df"]
        target_col = state.get("target_col")

        ml_results = run_ml_agent(cleaned_df, target_col=target_col)

        best_model = ml_results["interpretation"]["best_model"]
        trail = log({"audit_trail": trail}, "ML",
                    "Training complete",
                    f"Best model: {best_model}")

        return {
            **state,
            "ml_results": ml_results,
            "current_node": "ml",
            "audit_trail": trail,
            "error": None
        }

    except Exception as e:
        trail = log(state, "ML", "FAILED", str(e))
        return {
            **state,
            "current_node": "ml",
            "audit_trail": trail,
            "error": f"ml: {str(e)}"
        }


# ─────────────────────────────────────────────────
# NODE 5 — Explanation
# ─────────────────────────────────────────────────

def explain_node(state: ADAState) -> ADAState:
    """
    Computes SHAP values and generates plain English explanations.
    Non-critical — if this fails, pipeline continues to report.
    """
    try:
        trail = log(state, "Explain", "Starting SHAP analysis...")

        cleaned_df = state["cleaned_df"]
        ml_results = state["ml_results"]

        explain_results = run_explain_agent(
            df=cleaned_df,
            target_col=ml_results["target_column"],
            problem_type=ml_results["problem_type"],
            best_model_name=ml_results["interpretation"]["best_model"]
        )

        top_feature = list(
            explain_results["feature_importance"].keys()
        )[0]
        trail = log({"audit_trail": trail}, "Explain",
                    "Explanation complete",
                    f"Top feature: {top_feature}")

        return {
            **state,
            "explain_results": explain_results,
            "current_node": "explain",
            "audit_trail": trail,
            "error": None
        }

    except Exception as e:
        trail = log(state, "Explain", "FAILED — skipping", str(e))
        return {
            **state,
            "explain_results": {},
            "current_node": "explain",
            "audit_trail": trail,
            "error": None  # non-critical — don't stop pipeline
        }


# ─────────────────────────────────────────────────
# NODE 6 — Report
# ─────────────────────────────────────────────────

def report_node(state: ADAState) -> ADAState:
    """
    Generates the final report and saves outputs.
    Last node in the pipeline.
    """
    try:
        trail = log(state, "Report", "Generating final report...")

        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build summary for GPT
        summary = {
            "eda": state.get("eda_report", "")[:500],
            "cleaning": state.get("cleaning_strategy", {}),
            "ml": state.get("ml_results", {}).get("interpretation", {}),
            "explanation": state.get("explain_results", {}).get(
                "interpretation", {})
        }

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior data scientist writing professional reports."
                },
                {
                    "role": "user",
                    "content": f"Write a professional data analysis report based on: {json.dumps(summary)}"
                }
            ],
            max_tokens=1000
        )

        final_report = response.choices[0].message.content

        # Save outputs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"outputs/ADA_v2_report_{timestamp}.txt"
        audit_path = f"outputs/ADA_v2_audit_{timestamp}.json"

        with open(report_path, "w") as f:
            f.write(final_report)

        trail_data = state.get("audit_trail", [])
        with open(audit_path, "w") as f:
            json.dump(trail_data, f, indent=2)

        trail = log({"audit_trail": trail}, "Report",
                    "Pipeline complete!",
                    f"Report saved to {report_path}")

        return {
            **state,
            "final_report": final_report,
            "end_time": end_time,
            "current_node": "report",
            "audit_trail": trail,
            "error": None
        }

    except Exception as e:
        trail = log(state, "Report", "FAILED", str(e))
        return {
            **state,
            "current_node": "report",
            "audit_trail": trail,
            "error": f"report: {str(e)}"
        }


# ─────────────────────────────────────────────────
# NODE 7 — Error Handler
# ─────────────────────────────────────────────────

def error_handler_node(state: ADAState) -> ADAState:
    """
    Handles errors from any node.
    Decides whether to retry or stop.
    This is the self-correction mechanism —
    your key research contribution.
    """
    error = state.get("error", "")
    retry_count = state.get("retry_count", 0)
    current_node = state.get("current_node", "")

    trail = log(state, "ErrorHandler",
                f"Handling error in {current_node}",
                error)

    # Critical nodes — stop the pipeline
    critical_nodes = ["load_data"]
    if current_node in critical_nodes:
        trail = log({"audit_trail": trail}, "ErrorHandler",
                    "Critical node failed — stopping pipeline")
        return {
            **state,
            "audit_trail": trail,
            "error": f"CRITICAL: {error}"
        }

    # Max retries reached — skip this node
    if retry_count >= 2:
        trail = log({"audit_trail": trail}, "ErrorHandler",
                    f"Max retries reached for {current_node} — skipping")
        return {
            **state,
            "audit_trail": trail,
            "retry_count": 0,
            "error": None  # clear error and continue
        }

    # Retry the node
    trail = log({"audit_trail": trail}, "ErrorHandler",
                f"Retrying {current_node}",
                f"Attempt {retry_count + 1} of 2")
    return {
        **state,
        "audit_trail": trail,
        "retry_count": retry_count + 1,
        "error": None
    }


# ─────────────────────────────────────────────────
# CONDITIONAL EDGES
# ─────────────────────────────────────────────────

def check_error(state: ADAState) -> Literal[
    "error_handler", "eda", "cleaning", "ml",
    "explain", "report", "end"
]:
    """
    Called after every node.
    Routes to error_handler if there's an error,
    otherwise continues to the next node.
    This is what makes every node self-correcting.
    """
    if state.get("error"):
        return "error_handler"

    node = state.get("current_node", "")
    routing = {
        "load_data": "eda",
        "eda": "cleaning",
        "cleaning": "ml",
        "ml": "explain",
        "explain": "report",
        "report": "end"
    }
    return routing.get(node, "end")


# ─────────────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────────────

def build_ada_graph():
    """
    Assembles all nodes and edges into the LangGraph.
    This is where the magic happens — we define the
    structure of the pipeline declaratively.
    """

    graph = StateGraph(ADAState)

    # Add all nodes
    graph.add_node("load_data", load_data_node)
    graph.add_node("eda", eda_node)
    graph.add_node("cleaning", cleaning_node)
    graph.add_node("ml", ml_node)
    graph.add_node("explain", explain_node)
    graph.add_node("report", report_node)
    graph.add_node("error_handler", error_handler_node)

    # Entry point
    graph.set_entry_point("load_data")

    # Conditional edges after every node
    for node in ["load_data", "eda", "cleaning", "ml", "explain", "report"]:
        graph.add_conditional_edges(node, check_error, {
            "error_handler": "error_handler",
            "eda": "eda",
            "cleaning": "cleaning",
            "ml": "ml",
            "explain": "explain",
            "report": "report",
            "end": END
        })

    # Error handler routes back to the failed node or ends
    graph.add_conditional_edges("error_handler", check_error, {
        "error_handler": "error_handler",
        "eda": "eda",
        "cleaning": "cleaning",
        "ml": "ml",
        "explain": "explain",
        "report": "report",
        "end": END
    })

    return graph.compile()


# ─────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────

def run_ada_v2(filepath: str, target_col: str = None) -> dict:
    """
    Entry point for ADA v2.0.
    Builds the graph and runs it with initial state.
    """

    print("=" * 55)
    print("   ADA v2.0 — LangGraph Multi-Agent System")
    print("=" * 55)

    # Build the graph
    ada_graph = build_ada_graph()

    # Initial state — everything starts empty except inputs
    initial_state: ADAState = {
        "filepath": filepath,
        "target_col": target_col,
        "raw_df": None,
        "cleaned_df": None,
        "eda_stats": None,
        "eda_report": None,
        "cleaning_strategy": None,
        "ml_results": None,
        "explain_results": None,
        "final_report": None,
        "current_node": None,
        "error": None,
        "retry_count": 0,
        "audit_trail": [],
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": None
    }

    # Run the graph
    final_state = ada_graph.invoke(initial_state)

    print("\n" + "=" * 55)
    print("   ADA v2.0 Complete!")
    print("=" * 55)

    return final_state