# Example Anomaly Report

This is simulated example output from the portfolio project. It illustrates how
the structured inference JSON can be rendered into an operations-friendly
handoff without implying production readiness.

## Summary

- `rack_id`: `rack-demo-001`
- `severity`: `warning`
- `likely_pattern`: `localized_thermal_hotspot`
- `anomaly_score`: `0.82`

## Recommended Action

Inspect rack airflow path and verify localized heat sources.

## Structured JSON Report

```json
{
  "rack_id": "rack-demo-001",
  "anomaly_score": 0.82,
  "severity": "warning",
  "likely_pattern": "localized_thermal_hotspot",
  "contributing_signals": [
    "thermal_hotspot_score",
    "thermal_gradient_score",
    "hotspot_persistence_seconds"
  ],
  "recommended_action": "Inspect rack airflow path and verify localized heat sources."
}
```

## Interpretation

The report identifies a localized thermal hotspot pattern because the simulated
window shows elevated thermal-derived features and hotspot persistence. Those
contributing signals point to heat concentration over time rather than a single
isolated reading.

The anomaly score and severity summarize the model scoring layer, while the
likely pattern and contributing signals provide a practical explanation that an
operator or downstream workflow can inspect.

## Operational Value

This kind of structured report can feed a dashboard, ticketing system, runbook
workflow, or agentic operations handoff. The value is not autonomous control; it
is a clear, reviewable signal that connects model output to human triage or
assisted operations workflows.
