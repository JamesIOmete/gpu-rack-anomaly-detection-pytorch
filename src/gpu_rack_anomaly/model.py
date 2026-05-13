"""Small PyTorch autoencoder for telemetry feature windows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from gpu_rack_anomaly.features import FEATURE_COLUMNS, StandardScaler


@dataclass(frozen=True)
class AutoencoderConfig:
    """Shape and capacity settings for the window autoencoder."""

    window_size: int
    feature_count: int = len(FEATURE_COLUMNS)
    hidden_dim: int = 64
    latent_dim: int = 16

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.feature_count <= 0:
            raise ValueError("feature_count must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

    @property
    def input_dim(self) -> int:
        return self.window_size * self.feature_count


class WindowAutoencoder(nn.Module):
    """Fully connected autoencoder over flattened telemetry windows."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.input_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [batch, window_size, feature_count]")
        if inputs.shape[1] != self.config.window_size:
            raise ValueError("input window dimension does not match model config")
        if inputs.shape[2] != self.config.feature_count:
            raise ValueError("input feature dimension does not match model config")

        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded.view(-1, self.config.window_size, self.config.feature_count)


@dataclass(frozen=True)
class LoadedModel:
    """Loaded model artifact with preprocessing metadata."""

    model: WindowAutoencoder
    scaler: StandardScaler
    metrics: dict[str, Any]
    feature_columns: tuple[str, ...]


def reconstruction_loss(model: WindowAutoencoder, batch: torch.Tensor) -> torch.Tensor:
    """Mean squared reconstruction error for an autoencoder batch."""

    return nn.functional.mse_loss(model(batch), batch)


def save_model_artifact(
    path: str | Path,
    model: WindowAutoencoder,
    scaler: StandardScaler,
    metrics: dict[str, Any],
) -> None:
    """Persist model weights and preprocessing metadata."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_config": asdict(model.config),
        "state_dict": model.state_dict(),
        "scaler": {
            "feature_columns": scaler.feature_columns,
            "means": scaler.means,
            "stds": scaler.stds,
        },
        "metrics": metrics,
    }
    torch.save(artifact, output_path)


def load_model_artifact(path: str | Path, map_location: str | torch.device = "cpu") -> LoadedModel:
    """Load a saved autoencoder artifact."""

    artifact = torch.load(Path(path), map_location=map_location, weights_only=False)
    config = AutoencoderConfig(**artifact["model_config"])
    model = WindowAutoencoder(config)
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    scaler_payload = artifact["scaler"]
    scaler = StandardScaler(
        feature_columns=tuple(scaler_payload["feature_columns"]),
        means=tuple(float(value) for value in scaler_payload["means"]),
        stds=tuple(float(value) for value in scaler_payload["stds"]),
    )
    return LoadedModel(
        model=model,
        scaler=scaler,
        metrics=dict(artifact.get("metrics", {})),
        feature_columns=scaler.feature_columns,
    )
