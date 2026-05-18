from ada_graph import run_ada_v2

results = run_ada_v2(
    filepath="data/sample.csv",
    target_col="Survived"
)

print("\nFinal report preview:")
print(results["final_report"][:500])
print(f"\nAudit trail entries: {len(results['audit_trail'])}")