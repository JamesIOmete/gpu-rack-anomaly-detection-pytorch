import json

from gpu_rack_anomaly.features import telemetry_window_to_feature_matrix
from gpu_rack_anomaly.infer import classify_likely_pattern, infer_file, main
from gpu_rack_anomaly.simulate_telemetry import simulate_window, write_window_json
from gpu_rack_anomaly.train import train_autoencoder


def test_classify_likely_pattern_identifies_localized_hotspot() -> None:
    telemetry = simulate_window("localized_hotspot", window_size=24, seed=41)
    features = telemetry_window_to_feature_matrix(telemetry)

    pattern, signals, action = classify_likely_pattern(features)

    assert pattern == "localized_thermal_hotspot"
    assert "thermal_hotspot_score" in signals
    assert action


def test_infer_file_returns_operational_anomaly_report(tmp_path) -> None:
    model_path, telemetry_path = _train_and_write_hotspot_input(tmp_path)
    output_path = tmp_path / "report.json"

    report = infer_file(
        model_path=model_path,
        input_path=telemetry_path,
        output_path=output_path,
        stride=4,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved == report
    assert report["rack_id"] == "rack-demo-001"
    assert 0.0 <= report["anomaly_score"] <= 1.0
    assert report["severity"] in {"normal", "watch", "warning", "critical"}
    assert report["likely_pattern"] == "localized_thermal_hotspot"
    assert "thermal_hotspot_score" in report["contributing_signals"]
    assert report["recommended_action"]


def test_infer_cli_writes_json_output(tmp_path) -> None:
    model_path, telemetry_path = _train_and_write_hotspot_input(tmp_path)
    output_path = tmp_path / "cli-report.json"

    exit_code = main(
        [
            "--model",
            str(model_path),
            "--input",
            str(telemetry_path),
            "--output",
            str(output_path),
            "--stride",
            "4",
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["likely_pattern"] == "localized_thermal_hotspot"


def _train_and_write_hotspot_input(tmp_path):
    normal = simulate_window("normal", window_size=80, seed=42)
    train_autoencoder(
        feature_matrix=telemetry_window_to_feature_matrix(normal),
        output_dir=tmp_path / "artifacts",
        window_size=8,
        stride=4,
        epochs=2,
        batch_size=8,
        hidden_dim=16,
        latent_dim=4,
        seed=42,
    )
    telemetry_path = tmp_path / "hotspot.json"
    write_window_json(simulate_window("localized_hotspot", window_size=32, seed=43), telemetry_path)
    return tmp_path / "artifacts" / "autoencoder.pt", telemetry_path
