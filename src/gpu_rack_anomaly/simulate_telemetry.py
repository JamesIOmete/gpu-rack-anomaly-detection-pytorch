"""Synthetic GPU rack telemetry simulator.

The simulator creates public-safe rack-level telemetry windows with derived
thermal-camera features. It is deterministic when a seed is provided so tests
and examples remain stable.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from gpu_rack_anomaly.schemas import AnomalyScenario, TelemetrySample, TelemetryWindow, isoformat_utc


DEFAULT_RACK_ID = "rack-demo-001"
DEFAULT_WINDOW_SIZE = 60
DEFAULT_SAMPLE_PERIOD_SECONDS = 10


def simulate_window(
    scenario: AnomalyScenario | str = AnomalyScenario.NORMAL,
    *,
    rack_id: str = DEFAULT_RACK_ID,
    window_size: int = DEFAULT_WINDOW_SIZE,
    sample_period_seconds: int = DEFAULT_SAMPLE_PERIOD_SECONDS,
    seed: int | None = None,
    start_time: datetime | None = None,
) -> TelemetryWindow:
    """Generate one telemetry window for a normal or anomalous pattern."""

    scenario_value = _coerce_scenario(scenario)
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if sample_period_seconds <= 0:
        raise ValueError("sample_period_seconds must be positive")

    rng = random.Random(seed)
    start = start_time or datetime(2026, 1, 1, tzinfo=timezone.utc)

    samples = []
    for index in range(window_size):
        progress = index / max(window_size - 1, 1)
        timestamp = start + timedelta(seconds=index * sample_period_seconds)
        base = _normal_sample_values(rng, progress)
        adjusted = _apply_scenario(base, scenario_value, rng, progress, index, sample_period_seconds)
        samples.append(
            TelemetrySample(
                timestamp=isoformat_utc(timestamp),
                rack_id=rack_id,
                scenario=scenario_value.value,
                **{name: round(value, 3) for name, value in adjusted.items()},
            )
        )

    return TelemetryWindow(
        rack_id=rack_id,
        scenario=scenario_value.value,
        sample_period_seconds=sample_period_seconds,
        samples=tuple(samples),
    )


def write_window_json(window: TelemetryWindow, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(window.to_dict(), indent=2) + "\n", encoding="utf-8")


def _normal_sample_values(rng: random.Random, progress: float) -> dict[str, float]:
    daily_wave = math.sin(progress * math.tau)
    power = 82.0 + 2.2 * daily_wave + rng.gauss(0.0, 0.7)
    inlet = 22.0 + 0.5 * daily_wave + rng.gauss(0.0, 0.15)
    airflow = 12200.0 + rng.gauss(0.0, 120.0)
    coolant_supply = 18.0 + rng.gauss(0.0, 0.08)
    coolant_return = 23.0 + 0.025 * power + rng.gauss(0.0, 0.12)
    gpu_temp = 54.0 + 0.18 * power + 0.08 * (inlet - 22.0) + rng.gauss(0.0, 0.45)
    outlet = inlet + 9.0 + 0.035 * power + rng.gauss(0.0, 0.2)

    return {
        "gpu_temperature_c": gpu_temp,
        "inlet_temperature_c": inlet,
        "outlet_temperature_c": outlet,
        "airflow_cfm": airflow,
        "relative_humidity_pct": 43.0 + rng.gauss(0.0, 1.0),
        "rack_power_kw": power,
        "coolant_supply_temperature_c": coolant_supply,
        "coolant_return_temperature_c": coolant_return,
        "vibration_mm_s": 1.2 + rng.gauss(0.0, 0.08),
        "thermal_hotspot_score": _clip(0.12 + rng.gauss(0.0, 0.025), 0.0, 1.0),
        "thermal_gradient_score": _clip(0.16 + rng.gauss(0.0, 0.03), 0.0, 1.0),
        "hotspot_persistence_seconds": max(0.0, rng.gauss(8.0, 3.0)),
    }


def _apply_scenario(
    values: dict[str, float],
    scenario: AnomalyScenario,
    rng: random.Random,
    progress: float,
    index: int,
    sample_period_seconds: int,
) -> dict[str, float]:
    adjusted = dict(values)
    if scenario is AnomalyScenario.NORMAL:
        return _bounded(adjusted)

    if scenario is AnomalyScenario.COOLING_DEGRADATION:
        ramp = 1.0 + 7.0 * progress
        adjusted["gpu_temperature_c"] += ramp
        adjusted["outlet_temperature_c"] += ramp * 0.75
        adjusted["coolant_return_temperature_c"] += ramp * 0.45
        adjusted["thermal_gradient_score"] += 0.08 + 0.22 * progress

    elif scenario is AnomalyScenario.AIRFLOW_OBSTRUCTION:
        obstruction = 0.22 + 0.2 * progress
        adjusted["airflow_cfm"] *= 1.0 - obstruction
        adjusted["gpu_temperature_c"] += 4.0 + 4.0 * progress
        adjusted["outlet_temperature_c"] += 3.0 + 3.0 * progress
        adjusted["thermal_gradient_score"] += 0.18 + 0.18 * progress

    elif scenario is AnomalyScenario.LOCALIZED_HOTSPOT:
        adjusted["thermal_hotspot_score"] += 0.52 + rng.uniform(0.0, 0.1)
        adjusted["thermal_gradient_score"] += 0.34 + rng.uniform(0.0, 0.08)
        adjusted["hotspot_persistence_seconds"] = (index + 1) * sample_period_seconds
        adjusted["gpu_temperature_c"] += 2.0 + 2.5 * progress

    elif scenario is AnomalyScenario.COOLANT_LOOP_INSTABILITY:
        oscillation = math.sin(progress * math.tau * 4)
        adjusted["coolant_supply_temperature_c"] += 1.8 * oscillation
        adjusted["coolant_return_temperature_c"] += 2.4 * oscillation
        adjusted["gpu_temperature_c"] += 1.2 * max(oscillation, 0.0)
        adjusted["thermal_gradient_score"] += abs(oscillation) * 0.16

    elif scenario is AnomalyScenario.DOOR_OPEN_CONTAINMENT_EVENT:
        event = 1.0 if 0.35 <= progress <= 0.85 else 0.25
        adjusted["inlet_temperature_c"] += 3.5 * event
        adjusted["relative_humidity_pct"] -= 7.0 * event
        adjusted["airflow_cfm"] += 900.0 * event
        adjusted["gpu_temperature_c"] += 2.2 * event
        adjusted["thermal_gradient_score"] += 0.12 * event

    elif scenario is AnomalyScenario.SENSOR_DRIFT:
        drift = 6.0 * progress
        adjusted["gpu_temperature_c"] += drift
        adjusted["inlet_temperature_c"] += drift * 0.45
        adjusted["outlet_temperature_c"] += drift * 0.45
        adjusted["thermal_hotspot_score"] += 0.04 * progress

    return _bounded(adjusted)


def _bounded(values: dict[str, float]) -> dict[str, float]:
    bounds = {
        "gpu_temperature_c": (0.0, 120.0),
        "inlet_temperature_c": (-10.0, 60.0),
        "outlet_temperature_c": (-10.0, 80.0),
        "airflow_cfm": (0.0, 30000.0),
        "relative_humidity_pct": (0.0, 100.0),
        "rack_power_kw": (0.0, 250.0),
        "coolant_supply_temperature_c": (-5.0, 50.0),
        "coolant_return_temperature_c": (-5.0, 70.0),
        "vibration_mm_s": (0.0, 50.0),
        "thermal_hotspot_score": (0.0, 1.0),
        "thermal_gradient_score": (0.0, 1.0),
        "hotspot_persistence_seconds": (0.0, 3600.0),
    }
    return {name: _clip(value, *bounds[name]) for name, value in values.items()}


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _coerce_scenario(value: AnomalyScenario | str) -> AnomalyScenario:
    if isinstance(value, AnomalyScenario):
        return value
    return AnomalyScenario(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic GPU rack telemetry JSON.")
    parser.add_argument("--scenario", choices=[item.value for item in AnomalyScenario], default="normal")
    parser.add_argument("--rack-id", default=DEFAULT_RACK_ID)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--sample-period-seconds", type=int, default=DEFAULT_SAMPLE_PERIOD_SECONDS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    window = simulate_window(
        args.scenario,
        rack_id=args.rack_id,
        window_size=args.window_size,
        sample_period_seconds=args.sample_period_seconds,
        seed=args.seed,
    )
    write_window_json(window, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
