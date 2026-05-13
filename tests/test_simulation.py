import json

from gpu_rack_anomaly.schemas import AnomalyScenario, TelemetryWindow
from gpu_rack_anomaly.simulate_telemetry import main, simulate_window


def test_simulation_is_deterministic_with_seed() -> None:
    first = simulate_window("normal", window_size=5, seed=42)
    second = simulate_window("normal", window_size=5, seed=42)

    assert first == second


def test_simulation_supports_all_scenarios() -> None:
    for scenario in AnomalyScenario:
        window = simulate_window(scenario, window_size=6, seed=42)

        assert window.scenario == scenario.value
        assert len(window.samples) == 6
        assert {sample.scenario for sample in window.samples} == {scenario.value}


def test_airflow_obstruction_reduces_airflow_vs_normal() -> None:
    normal = simulate_window("normal", window_size=20, seed=9)
    obstructed = simulate_window("airflow_obstruction", window_size=20, seed=9)

    normal_airflow = _mean(sample.airflow_cfm for sample in normal.samples)
    obstructed_airflow = _mean(sample.airflow_cfm for sample in obstructed.samples)

    assert obstructed_airflow < normal_airflow * 0.75


def test_localized_hotspot_increases_thermal_features() -> None:
    normal = simulate_window("normal", window_size=20, seed=9)
    hotspot = simulate_window("localized_hotspot", window_size=20, seed=9)

    assert _mean(sample.thermal_hotspot_score for sample in hotspot.samples) > 0.6
    assert _mean(sample.thermal_gradient_score for sample in hotspot.samples) > _mean(
        sample.thermal_gradient_score for sample in normal.samples
    )
    assert hotspot.samples[-1].hotspot_persistence_seconds == 200


def test_cli_writes_valid_window_json(tmp_path) -> None:
    output = tmp_path / "window.json"

    exit_code = main(
        [
            "--scenario",
            "cooling_degradation",
            "--window-size",
            "4",
            "--seed",
            "5",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    window = TelemetryWindow.from_dict(payload)
    assert window.scenario == "cooling_degradation"
    assert len(window.samples) == 4


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)
