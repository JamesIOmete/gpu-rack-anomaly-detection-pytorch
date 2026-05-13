import json

from gpu_rack_anomaly.model import load_model_artifact
from gpu_rack_anomaly.train import main


def test_training_cli_creates_metrics_and_model_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "artifacts"

    exit_code = main(
        [
            "--output-dir",
            str(output_dir),
            "--samples",
            "80",
            "--window-size",
            "8",
            "--stride",
            "4",
            "--epochs",
            "2",
            "--batch-size",
            "8",
            "--hidden-dim",
            "16",
            "--latent-dim",
            "4",
            "--seed",
            "21",
        ]
    )

    metrics_path = output_dir / "metrics.json"
    model_path = output_dir / "autoencoder.pt"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    loaded = load_model_artifact(model_path)

    assert exit_code == 0
    assert metrics_path.exists()
    assert model_path.exists()
    assert metrics["model_type"] == "window_autoencoder"
    assert metrics["epochs"] == 2
    assert metrics["train_windows"] > 0
    assert metrics["test_windows"] > 0
    assert metrics["final_train_loss"] >= 0.0
    assert metrics["final_test_loss"] >= 0.0
    assert loaded.model.config.window_size == 8
