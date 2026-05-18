from typing import TypedDict, Optional, Any
import pandas as pd


class ADAState(TypedDict):
    """
    The central shared state for the entire ADA pipeline.

    This is the 'whiteboard' that every agent reads from
    and writes to. LangGraph passes this state between
    nodes automatically.

    Using TypedDict here is important — it gives us
    type safety (Python knows what fields exist and
    what types they should be) while keeping the
    structure simple and serializable.
    """

    # ── Input ────────────────────────────────────────
    filepath: str                    # path to the CSV file
    target_col: Optional[str]        # target column name

    # ── Data ─────────────────────────────────────────
    raw_df: Optional[Any]            # original dataframe
    cleaned_df: Optional[Any]        # cleaned dataframe

    # ── Agent outputs ─────────────────────────────────
    eda_stats: Optional[dict]        # raw statistics from EDA
    eda_report: Optional[str]        # AI-generated EDA report
    cleaning_strategy: Optional[dict]# decisions made by cleaning agent
    ml_results: Optional[dict]       # model training results
    explain_results: Optional[dict]  # SHAP explanation results
    final_report: Optional[str]      # final synthesized report

    # ── Pipeline control ──────────────────────────────
    current_node: Optional[str]      # which node is running now
    error: Optional[str]             # error message if something failed
    retry_count: int                 # how many times we've retried
    audit_trail: list                # every decision logged here

    # ── Metadata ──────────────────────────────────────
    start_time: Optional[str]        # when pipeline started
    end_time: Optional[str]          # when pipeline finished