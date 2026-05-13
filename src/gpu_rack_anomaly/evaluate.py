"""Evaluate a trained autoencoder against telemetry window JSON."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from gpu_rack_anomaly.features import (
    FEATURE_COLUMNS,
    FeatureMatrix,
    make_sliding_windows,
    normalize_features,
    telemetry_window_to_feature_matrix,
)
from gpu_rack_anomaly.model import LoadedModel, load_model_artifact
from gpu_rack_anomaly.schemas import TelemetryWindow


@dataclass(frozen=True)
class Thresholds:
    """Deterministic severity thresholds derived from training metadata."""

    watch: float
    warning: float
    critical: float

    def to_dict(self) -> dict[str, float]:
        return {"watch": self.watch, "warning": self.warning, "critical": self.critical}


@dataclass(frozen=True)
class ScoredWindow:
    """One reconstruction-error score for a sliding telemetry window."""

    start_index: int
    reconstruction_error: float
    severity: str
    anomaly_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_index": self.start_index,
            "reconstruction_error": self.reconstruction_error,
            "severity": self.severity,
            "anomaly_score": self.anomaly_score,
        }


def load_telemetry_window(path: str | Path) -> TelemetryWindow:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return TelemetryWindow.from_dict(payload)


def score_telemetry_window(
    *,
    telemetry: TelemetryWindow,
    loaded_model: LoadedModel,
    stride: int | None = None,
) -> tuple[list[ScoredWindow], FeatureMatrix]:
    """Score telemetry windows with a trained model and saved scaler."""

    model = loaded_model.model
    feature_matrix = telemetry_window_to_feature_matrix(telemetry, loaded_model.feature_columns)
    normalized = normalize_features(feature_matrix, loaded_model.scaler)
    window_stride = stride or int(loaded_model.metrics.get("stride", 1))
    windowed = make_sliding_windows(
        normalized,
        window_size=model.config.window_size,
        stride=window_stride,
    )
    inputs = torch.tensor(windowed.windows, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        errors = nn.functional.mse_loss(outputs, inputs, reduction="none").mean(dim=(1, 2))

    thresholds = derive_thresholds(loaded_model.metrics)
    scored = [
        ScoredWindow(
            start_index=start_index,
            reconstruction_error=float(error),
            severity=severity_for_error(float(error), thresholds),
            anomaly_score=anomaly_score_for_error(float(error), thresholds),
        )
        for start_index, error in zip(windowed.start_indices, errors, strict=True)
    ]
    return scored, feature_matrix


def evaluate_file(
    *,
    model_path: str | Path,
    input_path: str | Path,
    output_path: str | Path | None = None,
    stride: int | None = None,
) -> dict[str, Any]:
    loaded = load_model_artifact(model_path)
    telemetry = load_telemetry_window(input_path)
    scored, features = score_telemetry_window(telemetry=telemetry, loaded_model=loaded, stride=stride)
    thresholds = derive_thresholds(loaded.metrics)
    errors = [item.reconstruction_error for item in scored]
    severities = Counter(item.severity for item in scored)
    summary = {
        "model_path": str(model_path),
        "input_path": str(input_path),
        "rack_id": telemetry.rack_id,
        "scenario": telemetry.scenario,
        "feature_columns": list(features.feature_columns),
        "window_size": loaded.model.config.window_size,
        "stride": stride or int(loaded.metrics.get("stride", 1)),
        "window_count": len(scored),
        "mean_reconstruction_error": _mean(errors),
        "max_reconstruction_error": max(errors),
        "p95_reconstruction_error": _percentile(errors, 95.0),
        "thresholds": thresholds.to_dict(),
        "severity_counts": dict(sorted(severities.items())),
        "windows": [item.to_dict() for item in scored],
    }
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def derive_thresholds(metrics: dict[str, Any]) -> Thresholds:
    """Derive simple thresholds from training loss metadata."""

    baseline = float(metrics.get("final_train_loss") or metrics.get("final_test_loss") or 1.0)
    baseline = max(baseline, 1e-6)
    return Thresholds(watch=baseline * 1.5, warning=baseline * 2.5, critical=baseline * 4.0)


def severity_for_error(error: float, thresholds: Thresholds) -> str:
    if error >= thresholds.critical:
        return "critical"
    if error >= thresholds.warning:
        return "warning"
    if error >= thresholds.watch:
        return "watch"
    return "normal"


def anomaly_score_for_error(error: float, thresholds: Thresholds) -> float:
    return round(min(max(error / thresholds.critical, 0.0), 1.0), 4)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained telemetry autoencoder.")
    parser.add_argument("--model", type=Path, required=True, help="Path to autoencoder.pt artifact.")
    parser.add_argument("--input", type=Path, required=True, help="TelemetryWindow JSON file.")
    parser.add_argument("--output", type=Path, help="Optional metrics JSON output path.")
    parser.add_argument("--stride", type=int, help="Override evaluation stride.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    metrics = evaluate_file(
        model_path=args.model,
        input_path=args.input,
        output_path=args.output,
        stride=args.stride,
    )
    print(json.dumps(metrics, indent=2))
    return 0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


if __name__ == "__main__":
    raise SystemExit(main())
