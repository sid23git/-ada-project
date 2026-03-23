from agents.orchestrator import ADAOrchestrator

# This is now the entire entry point for ADA.
# One object, one method call — the orchestrator
# handles everything else automatically.
ada = ADAOrchestrator(
    filepath="data/sample.csv",
    target_col="Survived"
)

results = ada.run_pipeline()