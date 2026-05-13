"""Public-safe schemas for simulated GPU rack telemetry.

The project intentionally models derived rack-level signals rather than raw
datacenter feeds or thermal imagery. These schemas are dependency-light so the
Phase 1 simulator and tests can run in a minimal local environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class AnomalyScenario(str, Enum):
    """Supported synthetic operating patterns."""

    NORMAL = "normal"
    COOLING_DEGRADATION = "cooling_degradation"
    AIRFLOW_OBSTRUCTION = "airflow_obstruction"
    LOCALIZED_HOTSPOT = "localized_hotspot"
    COOLANT_LOOP_INSTABILITY = "coolant_loop_instability"
    DOOR_OPEN_CONTAINMENT_EVENT = "door_open_containment_event"
    SENSOR_DRIFT = "sensor_drift"


@dataclass(frozen=True)
class TelemetrySample:
    """One timestamped rack telemetry observation."""

    timestamp: str
    rack_id: str
    gpu_temperature_c: float
    inlet_temperature_c: float
    outlet_temperature_c: float
    airflow_cfm: float
    relative_humidity_pct: float
    rack_power_kw: float
    coolant_supply_temperature_c: float
    coolant_return_temperature_c: float
    vibration_mm_s: float
    thermal_hotspot_score: float
    thermal_gradient_score: float
    hotspot_persistence_seconds: float
    scenario: str = AnomalyScenario.NORMAL.value

    def __post_init__(self) -> None:
        _validate_timestamp(self.timestamp)
        _require_non_empty("rack_id", self.rack_id)
        _require_enum_value("scenario", self.scenario, AnomalyScenario)
        _require_range("gpu_temperature_c", self.gpu_temperature_c, 0.0, 120.0)
        _require_range("inlet_temperature_c", self.inlet_temperature_c, -10.0, 60.0)
        _require_range("outlet_temperature_c", self.outlet_temperature_c, -10.0, 80.0)
        _require_range("airflow_cfm", self.airflow_cfm, 0.0, 30000.0)
        _require_range("relative_humidity_pct", self.relative_humidity_pct, 0.0, 100.0)
        _require_range("rack_power_kw", self.rack_power_kw, 0.0, 250.0)
        _require_range(
            "coolant_supply_temperature_c", self.coolant_supply_temperature_c, -5.0, 50.0
        )
        _require_range(
            "coolant_return_temperature_c", self.coolant_return_temperature_c, -5.0, 70.0
        )
        _require_range("vibration_mm_s", self.vibration_mm_s, 0.0, 50.0)
        _require_range("thermal_hotspot_score", self.thermal_hotspot_score, 0.0, 1.0)
        _require_range("thermal_gradient_score", self.thermal_gradient_score, 0.0, 1.0)
        _require_range(
            "hotspot_persistence_seconds", self.hotspot_persistence_seconds, 0.0, 3600.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "rack_id": self.rack_id,
            "gpu_temperature_c": self.gpu_temperature_c,
            "inlet_temperature_c": self.inlet_temperature_c,
            "outlet_temperature_c": self.outlet_temperature_c,
            "airflow_cfm": self.airflow_cfm,
            "relative_humidity_pct": self.relative_humidity_pct,
            "rack_power_kw": self.rack_power_kw,
            "coolant_supply_temperature_c": self.coolant_supply_temperature_c,
            "coolant_return_temperature_c": self.coolant_return_temperature_c,
            "vibration_mm_s": self.vibration_mm_s,
            "thermal_hotspot_score": self.thermal_hotspot_score,
            "thermal_gradient_score": self.thermal_gradient_score,
            "hotspot_persistence_seconds": self.hotspot_persistence_seconds,
            "scenario": self.scenario,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TelemetrySample":
        return cls(**payload)


@dataclass(frozen=True)
class TelemetryWindow:
    """A fixed-length telemetry sequence used by future model stages."""

    rack_id: str
    scenario: str
    sample_period_seconds: int
    samples: tuple[TelemetrySample, ...]

    def __post_init__(self) -> None:
        _require_non_empty("rack_id", self.rack_id)
        _require_enum_value("scenario", self.scenario, AnomalyScenario)
        if self.sample_period_seconds <= 0:
            raise ValueError("sample_period_seconds must be positive")
        if not self.samples:
            raise ValueError("samples must not be empty")
        for sample in self.samples:
            if sample.rack_id != self.rack_id:
                raise ValueError("all samples must use the window rack_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rack_id": self.rack_id,
            "scenario": self.scenario,
            "sample_period_seconds": self.sample_period_seconds,
            "samples": [sample.to_dict() for sample in self.samples],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TelemetryWindow":
        samples = tuple(TelemetrySample.from_dict(item) for item in payload["samples"])
        return cls(
            rack_id=payload["rack_id"],
            scenario=payload["scenario"],
            sample_period_seconds=payload["sample_period_seconds"],
            samples=samples,
        )


@dataclass(frozen=True)
class AnomalyReport:
    """Future inference output schema.

    Phase 1 does not implement inference, but keeping the report contract here
    anchors downstream model and CLI work.
    """

    rack_id: str
    anomaly_score: float
    severity: str
    likely_pattern: str
    contributing_signals: tuple[str, ...]
    recommended_action: str

    def __post_init__(self) -> None:
        _require_non_empty("rack_id", self.rack_id)
        _require_range("anomaly_score", self.anomaly_score, 0.0, 1.0)
        _require_non_empty("severity", self.severity)
        _require_non_empty("likely_pattern", self.likely_pattern)
        if not self.contributing_signals:
            raise ValueError("contributing_signals must not be empty")
        _require_non_empty("recommended_action", self.recommended_action)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rack_id": self.rack_id,
            "anomaly_score": self.anomaly_score,
            "severity": self.severity,
            "likely_pattern": self.likely_pattern,
            "contributing_signals": list(self.contributing_signals),
            "recommended_action": self.recommended_action,
        }


def _validate_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be ISO-8601 formatted") from exc
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include timezone information")


def isoformat_utc(value: datetime) -> str:
    """Return a compact UTC ISO-8601 timestamp with a Z suffix."""

    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _require_non_empty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_range(name: str, value: float, lower: float, upper: float) -> None:
    if not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    if value < lower or value > upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")


def _require_enum_value(name: str, value: str, enum_type: type[Enum]) -> None:
    valid_values = {item.value for item in enum_type}
    if value not in valid_values:
        raise ValueError(f"{name} must be one of {sorted(valid_values)}")
