import torch

from gpu_rack_anomaly.features import FEATURE_COLUMNS, fit_standard_scaler
from gpu_rack_anomaly.model import (
    AutoencoderConfig,
    WindowAutoencoder,
    load_model_artifact,
    save_model_artifact,
)
from gpu_rack_anomaly.simulate_telemetry import simulate_window
from gpu_rack_anomaly.features import telemetry_window_to_feature_matrix


def test_autoencoder_forward_preserves_input_shape() -> None:
    config = AutoencoderConfig(window_size=5, feature_count=len(FEATURE_COLUMNS), hidden_dim=16, latent_dim=4)
    model = WindowAutoencoder(config)
    batch = torch.randn(3, config.window_size, config.feature_count)

    output = model(batch)

    assert output.shape == batch.shape


def test_autoencoder_rejects_wrong_input_shape() -> None:
    config = AutoencoderConfig(window_size=5, feature_count=len(FEATURE_COLUMNS))
    model = WindowAutoencoder(config)

    try:
        model(torch.randn(5, len(FEATURE_COLUMNS)))
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_model_artifact_round_trips_with_scaler(tmp_path) -> None:
    telemetry = simulate_window("normal", window_size=12, seed=13)
    features = telemetry_window_to_feature_matrix(telemetry)
    scaler = fit_standard_scaler(features)
    config = AutoencoderConfig(window_size=4, feature_count=len(FEATURE_COLUMNS), hidden_dim=16, latent_dim=4)
    model = WindowAutoencoder(config)
    path = tmp_path / "autoencoder.pt"
    metrics = {"final_train_loss": 0.1}

    save_model_artifact(path, model, scaler, metrics)
    loaded = load_model_artifact(path)

    assert path.exists()
    assert loaded.model.config == config
    assert loaded.scaler.feature_columns == FEATURE_COLUMNS
    assert loaded.metrics == metrics
