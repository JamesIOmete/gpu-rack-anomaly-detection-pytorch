# GPU Rack Anomaly Detection in PyTorch

Compact public portfolio project for anomaly detection on **simulated**
high-density GPU rack telemetry.

This project demonstrates an end-to-end machine learning engineering workflow:
synthetic telemetry generation, typed schemas, feature preparation, PyTorch
autoencoder training, reconstruction-error evaluation, and structured
operations-style inference reports.

All telemetry in this repository is simulated. The project is intentionally
public-safe: it does not model proprietary datacenter designs, real facility
layouts, vendor-specific control loops, or raw thermal camera imagery.

## Architecture

```text
Synthetic telemetry simulator
        |
        v
Typed telemetry schemas
        |
        v
Feature selection, normalization, and sliding windows
        |
        v
PyTorch autoencoder trained on normal operating windows
        |
        v
Evaluation metrics and reconstruction-error thresholds
        |
        v
Structured anomaly report JSON for operational handoff
```

Core modules:

- `schemas.py`: public-safe telemetry and report contracts
- `simulate_telemetry.py`: deterministic synthetic rack telemetry generation
- `features.py`: feature selection, standard scaling, windowing, and splits
- `model.py`: compact PyTorch window autoencoder and artifact utilities
- `train.py`: CPU-friendly training CLI
- `evaluate.py`: reconstruction-error evaluation CLI
- `infer.py`: deterministic anomaly report CLI

## What This Demonstrates

- Practical PyTorch fluency without notebook-only implementation
- Production-minded project structure with CLI entry points and tests
- Public-safe simulation of operational telemetry patterns
- Deterministic preprocessing, training controls, and smoke-testable workflows
- Structured JSON outputs suitable for downstream automation or incident review
- Explainable heuristics layered on model reconstruction error

## What This Intentionally Does Not Include

- No raw thermal image processing
- No real datacenter telemetry, facility layout, or proprietary rack design data
- No vendor-specific cooling, power, or controls assumptions
- No web app, dashboard, or notebook workflow
- No claims that the simulated thresholds are production-ready
- No automated remediation or closed-loop control actions

## Telemetry Scope

Each simulated sample includes:

- GPU, inlet, outlet, coolant supply, and coolant return temperatures
- Airflow, humidity, rack power, and vibration
- Derived thermal-camera-style features
- A scenario label for simulation and testing

Thermal camera data is represented only as derived features:

- `thermal_hotspot_score`
- `thermal_gradient_score`
- `hotspot_persistence_seconds`

Supported simulated scenarios:

- `normal`
- `cooling_degradation`
- `airflow_obstruction`
- `localized_hotspot`
- `coolant_loop_instability`
- `door_open_containment_event`
- `sensor_drift`

## Install

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the base package and test dependencies:

```bash
python -m pip install -e ".[dev]"
```

Install CPU-only PyTorch explicitly to avoid accidental CUDA wheel downloads:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[ml]"
```

If you already have a suitable PyTorch install, `python -m pip install -e
".[dev,ml]"` is also supported. On Linux, the default PyPI PyTorch package may
download large CUDA-enabled wheels, so the CPU-only command above is recommended
for laptops and WSL environments.

## End-To-End Example

Generate telemetry, train on normal behavior, evaluate an anomalous window, and
produce an operational report:

```bash
gpu-rack-simulate \
  --scenario localized_hotspot \
  --window-size 60 \
  --seed 7 \
  --output examples/localized_hotspot_window.json

gpu-rack-train \
  --samples 240 \
  --window-size 24 \
  --stride 4 \
  --epochs 8 \
  --output-dir artifacts

gpu-rack-evaluate \
  --model artifacts/autoencoder.pt \
  --input examples/localized_hotspot_window.json \
  --output artifacts/evaluation.json

gpu-rack-infer \
  --model artifacts/autoencoder.pt \
  --input examples/localized_hotspot_window.json \
  --output artifacts/anomaly_report.json
```

The inference report includes:

- `rack_id`
- `anomaly_score`
- `severity`
- `likely_pattern`
- `contributing_signals`
- `recommended_action`

## Feature Preparation

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

## Outputs

Training writes:

- `artifacts/autoencoder.pt`
- `artifacts/metrics.json`

Evaluation emits structured JSON with reconstruction error statistics, severity
counts, thresholds derived from training metrics, and per-window scores.

Inference emits a compact anomaly report intended for operational triage. Pattern
classification uses deterministic, explainable heuristics over derived telemetry
features and reconstruction error.

## Tests

```bash
pytest
```

The tests cover schema validation, deterministic simulation, feature extraction,
normalization, sliding window generation, model forward pass, training smoke
tests, artifact creation, evaluation metrics, and inference reports.

## Future Work

- Raw thermal image processing with explicit privacy and safety boundaries
- Integration with real sensor streams or historical telemetry exports
- Dashboarding for anomaly trends, feature attribution, and rack-level drilldown
- Agentic operations handoff that converts reports into ticket drafts or runbook
  recommendations, without taking automated control actions
- Better threshold calibration using larger normal and anomalous validation sets
- Batch evaluation workflows for multiple racks and time ranges
