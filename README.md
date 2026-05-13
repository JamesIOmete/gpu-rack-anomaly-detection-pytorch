# GPU Rack Anomaly Detection in PyTorch

Compact public portfolio project for anomaly detection on simulated
high-density GPU rack telemetry.

The current implementation covers the data foundation, feature preparation, a
compact PyTorch autoencoder training path, evaluation, and operational-style
inference reports.

## Scope

This repository uses public-safe synthetic data. It does not model proprietary
datacenter design details and does not process raw thermal camera images. Thermal
camera input is represented as derived features:

- `thermal_hotspot_score`
- `thermal_gradient_score`
- `hotspot_persistence_seconds`

## Telemetry Signals

Each simulated sample includes:

- GPU, inlet, outlet, coolant supply, and coolant return temperatures
- Airflow, humidity, rack power, and vibration
- Derived thermal-camera features
- A scenario label for simulation and testing

Supported scenarios:

- `normal`
- `cooling_degradation`
- `airflow_obstruction`
- `localized_hotspot`
- `coolant_loop_instability`
- `door_open_containment_event`
- `sensor_drift`

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,ml]"
```

## Generate Telemetry

```bash
gpu-rack-simulate \
  --scenario airflow_obstruction \
  --window-size 60 \
  --seed 7 \
  --output examples/airflow_obstruction_window.json
```

The simulator can also be called from Python:

```python
from gpu_rack_anomaly.simulate_telemetry import simulate_window

window = simulate_window("localized_hotspot", window_size=60, seed=7)
payload = window.to_dict()
```

## Examples

The `examples/` directory contains stable JSON windows for:

- `normal_window.json`
- `airflow_obstruction_window.json`
- `localized_hotspot_window.json`

## Prepare Features

```python
from gpu_rack_anomaly.features import (
    fit_standard_scaler,
    make_sliding_windows,
    normalize_features,
    telemetry_window_to_feature_matrix,
)
from gpu_rack_anomaly.simulate_telemetry import simulate_window

telemetry = simulate_window("normal", window_size=120, seed=7)
features = telemetry_window_to_feature_matrix(telemetry)
scaler = fit_standard_scaler(features)
normalized = normalize_features(features, scaler)
windows = make_sliding_windows(normalized, window_size=30, stride=5)
```

`windows.windows` is shaped as `[window, timestep, feature]` using plain Python
lists. The training code converts this structure to tensors internally.

## Train Autoencoder

Train a small CPU-friendly autoencoder on generated normal telemetry:

```bash
gpu-rack-train \
  --samples 240 \
  --window-size 24 \
  --stride 4 \
  --epochs 8 \
  --output-dir artifacts
```

The command writes:

- `artifacts/autoencoder.pt`
- `artifacts/metrics.json`

Metrics are also printed as JSON. The model trains only on normal synthetic
telemetry.

## Evaluate Model

Evaluate a trained artifact against telemetry JSON:

```bash
gpu-rack-evaluate \
  --model artifacts/autoencoder.pt \
  --input examples/localized_hotspot_window.json \
  --output artifacts/evaluation.json
```

Evaluation emits structured JSON with reconstruction error statistics, severity
counts, thresholds derived from training metrics, and per-window scores.

## Run Inference

Generate an operational-style anomaly report:

```bash
gpu-rack-infer \
  --model artifacts/autoencoder.pt \
  --input examples/localized_hotspot_window.json \
  --output artifacts/anomaly_report.json
```

Inference reports include `rack_id`, `anomaly_score`, `severity`,
`likely_pattern`, `contributing_signals`, and `recommended_action`. Pattern
classification uses deterministic, explainable heuristics over derived telemetry
features and reconstruction error.

## Tests

```bash
pytest
```

The tests validate schema constraints, deterministic simulation, anomaly
scenario coverage, command-line JSON generation, feature extraction,
normalization, sliding window generation, model forward pass, training smoke
test, artifact creation, evaluation metrics, and inference reports.

## Planned Later Phases

The operational inference output will eventually emit structured JSON with:

- `rack_id`
- `anomaly_score`
- `severity`
- `likely_pattern`
- `contributing_signals`
- `recommended_action`

Future phases may add richer calibration and batch workflows around this schema.
