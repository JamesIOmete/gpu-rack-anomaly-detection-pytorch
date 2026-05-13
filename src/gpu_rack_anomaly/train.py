"""Training CLI for the telemetry window autoencoder."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from gpu_rack_anomaly.features import (
    FEATURE_COLUMNS,
    FeatureMatrix,
    fit_standard_scaler,
    make_sliding_windows,
    normalize_features,
    sequential_train_test_split,
    telemetry_window_to_feature_matrix,
)
from gpu_rack_anomaly.model import (
    AutoencoderConfig,
    WindowAutoencoder,
    reconstruction_loss,
    save_model_artifact,
)
from gpu_rack_anomaly.schemas import TelemetryWindow
from gpu_rack_anomaly.simulate_telemetry import simulate_window


DEFAULT_OUTPUT_DIR = Path("artifacts")


def train_autoencoder(
    *,
    feature_matrix: FeatureMatrix,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    window_size: int = 24,
    stride: int = 4,
    train_fraction: float = 0.8,
    hidden_dim: int = 64,
    latent_dim: int = 16,
    epochs: int = 8,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 7,
) -> dict[str, Any]:
    """Train an autoencoder and write model/metrics artifacts."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")

    set_deterministic_training(seed)

    split = sequential_train_test_split(feature_matrix, train_fraction=train_fraction)
    scaler = fit_standard_scaler(split.train)
    normalized_train = normalize_features(split.train, scaler)
    normalized_test = normalize_features(split.test, scaler)
    train_windows = make_sliding_windows(normalized_train, window_size=window_size, stride=stride)
    test_windows = make_sliding_windows(normalized_test, window_size=window_size, stride=stride)

    train_tensor = _windows_to_tensor(train_windows.windows)
    test_tensor = _windows_to_tensor(test_windows.windows)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    config = AutoencoderConfig(
        window_size=window_size,
        feature_count=len(feature_matrix.feature_columns),
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    )
    model = WindowAutoencoder(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_losses = []
    for _epoch in range(epochs):
        model.train()
        batch_losses = []
        for (batch,) in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = reconstruction_loss(model, batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        epoch_losses.append(sum(batch_losses) / len(batch_losses))

    model.eval()
    with torch.no_grad():
        train_loss = float(reconstruction_loss(model, train_tensor).cpu())
        test_loss = float(reconstruction_loss(model, test_tensor).cpu())

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "autoencoder.pt"
    metrics_path = output_path / "metrics.json"
    metrics: dict[str, Any] = {
        "model_type": "window_autoencoder",
        "feature_columns": list(feature_matrix.feature_columns),
        "window_size": window_size,
        "stride": stride,
        "train_fraction": train_fraction,
        "train_rows": split.train.row_count,
        "test_rows": split.test.row_count,
        "train_windows": train_windows.window_count,
        "test_windows": test_windows.window_count,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "final_train_loss": train_loss,
        "final_test_loss": test_loss,
        "epoch_losses": epoch_losses,
        "model_config": asdict(config),
        "model_path": str(model_path),
    }

    save_model_artifact(model_path, model, scaler, metrics)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def set_deterministic_training(seed: int) -> None:
    """Set deterministic controls that are practical for CPU smoke training."""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def load_feature_matrix_from_json(path: str | Path) -> FeatureMatrix:
    """Load a TelemetryWindow JSON file and convert it to model features."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    telemetry = TelemetryWindow.from_dict(payload)
    return telemetry_window_to_feature_matrix(telemetry, FEATURE_COLUMNS)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a compact GPU rack telemetry autoencoder.")
    parser.add_argument("--input", type=Path, help="Optional TelemetryWindow JSON file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=240, help="Generated normal telemetry samples.")
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.input:
        feature_matrix = load_feature_matrix_from_json(args.input)
    else:
        telemetry = simulate_window("normal", window_size=args.samples, seed=args.seed)
        feature_matrix = telemetry_window_to_feature_matrix(telemetry, FEATURE_COLUMNS)

    metrics = train_autoencoder(
        feature_matrix=feature_matrix,
        output_dir=args.output_dir,
        window_size=args.window_size,
        stride=args.stride,
        train_fraction=args.train_fraction,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))
    return 0


def _windows_to_tensor(windows: list[list[list[float]]]) -> torch.Tensor:
    if not windows:
        raise ValueError("windows must not be empty")
    return torch.tensor(windows, dtype=torch.float32)


if __name__ == "__main__":
    raise SystemExit(main())
