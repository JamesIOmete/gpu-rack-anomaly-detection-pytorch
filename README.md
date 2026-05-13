# GPU Rack Anomaly Detection in PyTorch

Compact public portfolio project for anomaly detection on simulated
high-density GPU rack telemetry.

Phase 1 implements the data foundation only: schemas, a synthetic telemetry
simulator, example JSON windows, and pytest coverage. PyTorch model training,
evaluation, and inference CLIs are intentionally deferred.

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
python -m pip install -e ".[dev]"
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

## Tests

```bash
pytest
```

The Phase 1 tests validate schema constraints, deterministic simulation, anomaly
scenario coverage, and command-line JSON generation.

## Planned Later Phases

The operational inference output will eventually emit structured JSON with:

- `rack_id`
- `anomaly_score`
- `severity`
- `likely_pattern`
- `contributing_signals`
- `recommended_action`

Future phases will add the PyTorch autoencoder, training workflow, evaluation,
and inference CLI around this schema.
