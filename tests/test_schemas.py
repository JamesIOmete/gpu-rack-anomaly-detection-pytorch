import json
from pathlib import Path

import pytest

from gpu_rack_anomaly.schemas import AnomalyReport, TelemetrySample, TelemetryWindow
from gpu_rack_anomaly.simulate_telemetry import simulate_window


def test_example_windows_are_schema_valid() -> None:
    examples_dir = Path(__file__).resolve().parents[1] / "examples"

    for path in examples_dir.glob("*_window.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        window = TelemetryWindow.from_dict(payload)
        assert window.rack_id


def test_telemetry_window_round_trips_through_dict() -> None:
    window = simulate_window("normal", window_size=3, seed=123)

    restored = TelemetryWindow.from_dict(window.to_dict())

    assert restored == window
    assert restored.samples[0].rack_id == "rack-demo-001"


def test_sample_rejects_invalid_thermal_scores() -> None:
    sample = simulate_window("normal", window_size=1, seed=123).samples[0].to_dict()
    sample["thermal_hotspot_score"] = 1.5

    with pytest.raises(ValueError, match="thermal_hotspot_score"):
        TelemetrySample.from_dict(sample)


def test_window_rejects_mismatched_rack_ids() -> None:
    sample = simulate_window("normal", rack_id="rack-a", window_size=1, seed=123).samples[0]

    with pytest.raises(ValueError, match="window rack_id"):
        TelemetryWindow(
            rack_id="rack-b",
            scenario="normal",
            sample_period_seconds=10,
            samples=(sample,),
        )


def test_anomaly_report_schema_matches_future_output_contract() -> None:
    report = AnomalyReport(
        rack_id="rack-demo-001",
        anomaly_score=0.82,
        severity="high",
        likely_pattern="airflow_obstruction",
        contributing_signals=("airflow_cfm", "gpu_temperature_c"),
        recommended_action="Inspect front-of-rack airflow path.",
    )

    assert report.to_dict() == {
        "rack_id": "rack-demo-001",
        "anomaly_score": 0.82,
        "severity": "high",
        "likely_pattern": "airflow_obstruction",
        "contributing_signals": ["airflow_cfm", "gpu_temperature_c"],
        "recommended_action": "Inspect front-of-rack airflow path.",
    }
