"""Inference CLI for operational-style anomaly reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gpu_rack_anomaly.evaluate import (
    derive_thresholds,
    load_telemetry_window,
    score_telemetry_window,
)
from gpu_rack_anomaly.features import FEATURE_COLUMNS, FeatureMatrix
from gpu_rack_anomaly.model import load_model_artifact
from gpu_rack_anomaly.schemas import AnomalyReport


def infer_file(
    *,
    model_path: str | Path,
    input_path: str | Path,
    output_path: str | Path | None = None,
    stride: int | None = None,
) -> dict[str, Any]:
    loaded = load_model_artifact(model_path)
    telemetry = load_telemetry_window(input_path)
    scored, features = score_telemetry_window(telemetry=telemetry, loaded_model=loaded, stride=stride)
    worst = max(scored, key=lambda item: item.reconstruction_error)
    thresholds = derive_thresholds(loaded.metrics)
    report = build_anomaly_report(
        rack_id=telemetry.rack_id,
        feature_matrix=features,
        anomaly_score=worst.anomaly_score,
        severity=worst.severity,
        reconstruction_error=worst.reconstruction_error,
        critical_threshold=thresholds.critical,
    )
    payload = report.to_dict()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def build_anomaly_report(
    *,
    rack_id: str,
    feature_matrix: FeatureMatrix,
    anomaly_score: float,
    severity: str,
    reconstruction_error: float,
    critical_threshold: float,
) -> AnomalyReport:
    pattern, signals, action = classify_likely_pattern(feature_matrix)
    if severity == "normal" and reconstruction_error >= critical_threshold * 0.35:
        severity = "watch"
    return AnomalyReport(
        rack_id=rack_id,
        anomaly_score=anomaly_score,
        severity=severity,
        likely_pattern=pattern,
        contributing_signals=tuple(signals),
        recommended_action=action,
    )


def classify_likely_pattern(feature_matrix: FeatureMatrix) -> tuple[str, list[str], str]:
    """Explainable signal heuristics layered on reconstruction error."""

    stats = _feature_stats(feature_matrix)
    delta_t = stats["outlet_temperature_c"]["mean"] - stats["inlet_temperature_c"]["mean"]
    hotspot = stats["thermal_hotspot_score"]["mean"]
    gradient = stats["thermal_gradient_score"]["mean"]
    persistence = stats["hotspot_persistence_seconds"]["max"]
    airflow = stats["airflow_cfm"]["mean"]
    humidity = stats["relative_humidity_pct"]["mean"]
    coolant_swing = (
        stats["coolant_return_temperature_c"]["max"] - stats["coolant_return_temperature_c"]["min"]
    )

    if hotspot >= 0.45 or gradient >= 0.42 or persistence >= 120.0:
        return (
            "localized_thermal_hotspot",
            ["thermal_hotspot_score", "thermal_gradient_score", "hotspot_persistence_seconds"],
            "Inspect rack airflow path and verify localized heat sources.",
        )
    if airflow <= 9500.0 and delta_t >= 11.0:
        return (
            "airflow_obstruction",
            ["airflow_cfm", "delta_t_c", "outlet_temp_c"],
            "Inspect rack airflow path and remove possible intake or exhaust obstruction.",
        )
    if coolant_swing >= 3.0:
        return (
            "coolant_loop_instability",
            ["coolant_supply_temperature_c", "coolant_return_temperature_c", "delta_t_c"],
            "Verify cooling loop stability and review coolant temperature control.",
        )
    if humidity <= 38.0 and stats["inlet_temperature_c"]["mean"] >= 24.0:
        return (
            "door_open_or_containment_event",
            ["relative_humidity_pct", "inlet_temperature_c", "airflow_cfm"],
            "Check containment state and confirm rack doors or aisle barriers are closed.",
        )
    if stats["gpu_temperature_c"]["max"] - stats["gpu_temperature_c"]["min"] >= 5.0:
        return (
            "sensor_drift_or_cooling_degradation",
            ["gpu_temperature_c", "inlet_temperature_c", "outlet_temp_c"],
            "Compare sensor readings against adjacent telemetry and inspect cooling trend.",
        )
    return (
        "normal_operating_variation",
        ["reconstruction_error", "rack_power_kw"],
        "Continue monitoring normal telemetry trends.",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run autoencoder inference on telemetry JSON.")
    parser.add_argument("--model", type=Path, required=True, help="Path to autoencoder.pt artifact.")
    parser.add_argument("--input", type=Path, required=True, help="TelemetryWindow JSON file.")
    parser.add_argument("--output", type=Path, help="Optional anomaly report JSON output path.")
    parser.add_argument("--stride", type=int, help="Override inference stride.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = infer_file(
        model_path=args.model,
        input_path=args.input,
        output_path=args.output,
        stride=args.stride,
    )
    print(json.dumps(report, indent=2))
    return 0


def _feature_stats(feature_matrix: FeatureMatrix) -> dict[str, dict[str, float]]:
    stats = {}
    for column_index, column_name in enumerate(feature_matrix.feature_columns):
        values = [row[column_index] for row in feature_matrix.values]
        stats[column_name] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    for column in FEATURE_COLUMNS:
        stats.setdefault(column, {"mean": 0.0, "min": 0.0, "max": 0.0})
    return stats


if __name__ == "__main__":
    raise SystemExit(main())
