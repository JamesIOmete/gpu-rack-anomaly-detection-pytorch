import json

from gpu_rack_anomaly.evaluate import evaluate_file, main
from gpu_rack_anomaly.features import telemetry_window_to_feature_matrix
from gpu_rack_anomaly.simulate_telemetry import simulate_window, write_window_json
from gpu_rack_anomaly.train import train_autoencoder


def test_evaluate_file_returns_structured_metrics(tmp_path) -> None:
    model_path, telemetry_path = _train_and_write_input(tmp_path)
    output_path = tmp_path / "evaluation.json"

    metrics = evaluate_file(
        model_path=model_path,
        input_path=telemetry_path,
        output_path=output_path,
        stride=4,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved == metrics
    assert metrics["rack_id"] == "rack-demo-001"
    assert metrics["window_count"] > 0
    assert metrics["mean_reconstruction_error"] >= 0.0
    assert set(metrics["thresholds"]) == {"watch", "warning", "critical"}
    assert metrics["windows"][0]["severity"] in {"normal", "watch", "warning", "critical"}


def test_evaluate_cli_writes_json_output(tmp_path) -> None:
    model_path, telemetry_path = _train_and_write_input(tmp_path)
    output_path = tmp_path / "cli-evaluation.json"

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
    assert payload["window_count"] > 0


def _train_and_write_input(tmp_path):
    normal = simulate_window("normal", window_size=80, seed=31)
    train_autoencoder(
        feature_matrix=telemetry_window_to_feature_matrix(normal),
        output_dir=tmp_path / "artifacts",
        window_size=8,
        stride=4,
        epochs=2,
        batch_size=8,
        hidden_dim=16,
        latent_dim=4,
        seed=31,
    )
    telemetry_path = tmp_path / "normal.json"
    write_window_json(simulate_window("normal", window_size=32, seed=32), telemetry_path)
    return tmp_path / "artifacts" / "autoencoder.pt", telemetry_path
