# Generated Lab Reports

Run:

```powershell
uv run --extra dev safety-lab report
```

This creates:

- `reports/attack-defense-results.json`
- `reports/observability-demo.json`
- `reports/langsmith-local-traces.jsonl`

The generated files are ignored by Git because they may contain classroom test prompts, latency measurements, and intentionally vulnerable raw outputs.

