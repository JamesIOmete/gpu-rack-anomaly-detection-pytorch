"""Feature preparation utilities for future autoencoder training.

This module keeps Phase 2 dependency-light: it returns plain Python numeric
structures that can later be converted to tensors by the PyTorch training code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from gpu_rack_anomaly.schemas import TelemetrySample, TelemetryWindow


FEATURE_COLUMNS: tuple[str, ...] = (
    "gpu_temperature_c",
    "inlet_temperature_c",
    "outlet_temperature_c",
    "airflow_cfm",
    "relative_humidity_pct",
    "rack_power_kw",
    "coolant_supply_temperature_c",
    "coolant_return_temperature_c",
    "vibration_mm_s",
    "thermal_hotspot_score",
    "thermal_gradient_score",
    "hotspot_persistence_seconds",
)


Matrix = list[list[float]]
WindowTensor = list[list[list[float]]]


@dataclass(frozen=True)
class FeatureMatrix:
    """Numeric feature matrix with column order metadata."""

    values: Matrix
    feature_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.feature_columns:
            raise ValueError("feature_columns must not be empty")
        _validate_matrix_shape(self.values, len(self.feature_columns))

    @property
    def row_count(self) -> int:
        return len(self.values)

    @property
    def column_count(self) -> int:
        return len(self.feature_columns)


@dataclass(frozen=True)
class StandardScaler:
    """Simple per-column standardization parameters."""

    feature_columns: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.feature_columns:
            raise ValueError("feature_columns must not be empty")
        if len(self.means) != len(self.feature_columns):
            raise ValueError("means must match feature column count")
        if len(self.stds) != len(self.feature_columns):
            raise ValueError("stds must match feature column count")
        if any(std <= 0.0 for std in self.stds):
            raise ValueError("stds must be positive")

    def transform(self, matrix: FeatureMatrix | Sequence[Sequence[float]]) -> FeatureMatrix:
        values = matrix.values if isinstance(matrix, FeatureMatrix) else matrix
        _validate_matrix_shape(values, len(self.feature_columns))
        transformed = [
            [
                (float(value) - self.means[column_index]) / self.stds[column_index]
                for column_index, value in enumerate(row)
            ]
            for row in values
        ]
        return FeatureMatrix(values=transformed, feature_columns=self.feature_columns)

    def inverse_transform(self, matrix: FeatureMatrix | Sequence[Sequence[float]]) -> FeatureMatrix:
        values = matrix.values if isinstance(matrix, FeatureMatrix) else matrix
        _validate_matrix_shape(values, len(self.feature_columns))
        restored = [
            [
                float(value) * self.stds[column_index] + self.means[column_index]
                for column_index, value in enumerate(row)
            ]
            for row in values
        ]
        return FeatureMatrix(values=restored, feature_columns=self.feature_columns)

    @classmethod
    def fit(cls, matrix: FeatureMatrix) -> "StandardScaler":
        if not matrix.values:
            raise ValueError("cannot fit scaler on an empty matrix")
        _validate_matrix_shape(matrix.values, matrix.column_count)
        means = tuple(
            _mean(row[column_index] for row in matrix.values)
            for column_index in range(matrix.column_count)
        )
        stds = tuple(
            _safe_std((row[column_index] for row in matrix.values), mean)
            for column_index, mean in enumerate(means)
        )
        return cls(feature_columns=matrix.feature_columns, means=means, stds=stds)


@dataclass(frozen=True)
class WindowedFeatures:
    """Sliding windows plus metadata needed by training/evaluation code."""

    windows: WindowTensor
    feature_columns: tuple[str, ...]
    window_size: int
    stride: int
    start_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if not self.feature_columns:
            raise ValueError("feature_columns must not be empty")
        for window in self.windows:
            if len(window) != self.window_size:
                raise ValueError("all windows must match window_size")
            _validate_matrix_shape(window, len(self.feature_columns))
        if len(self.start_indices) != len(self.windows):
            raise ValueError("start_indices must match window count")

    @property
    def window_count(self) -> int:
        return len(self.windows)


@dataclass(frozen=True)
class FeatureSplit:
    """Deterministic sequential split for time-series style workflows."""

    train: FeatureMatrix
    test: FeatureMatrix
    split_index: int

    def __post_init__(self) -> None:
        if self.train.feature_columns != self.test.feature_columns:
            raise ValueError("train and test feature columns must match")
        if self.split_index != self.train.row_count:
            raise ValueError("split_index must match train row count")


def samples_to_feature_matrix(
    samples: Iterable[TelemetrySample],
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> FeatureMatrix:
    """Extract selected numeric fields from telemetry samples."""

    columns = tuple(feature_columns)
    if not columns:
        raise ValueError("feature_columns must not be empty")

    values: Matrix = []
    for sample in samples:
        row = []
        for column in columns:
            if not hasattr(sample, column):
                raise ValueError(f"unknown feature column: {column}")
            value = getattr(sample, column)
            if not isinstance(value, int | float):
                raise TypeError(f"feature column is not numeric: {column}")
            row.append(float(value))
        values.append(row)

    return FeatureMatrix(values=values, feature_columns=columns)


def telemetry_window_to_feature_matrix(
    window: TelemetryWindow,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> FeatureMatrix:
    """Extract model-ready features from a telemetry window schema."""

    return samples_to_feature_matrix(window.samples, feature_columns)


def fit_standard_scaler(matrix: FeatureMatrix) -> StandardScaler:
    """Fit per-column standardization statistics."""

    return StandardScaler.fit(matrix)


def normalize_features(matrix: FeatureMatrix, scaler: StandardScaler) -> FeatureMatrix:
    """Apply a fitted standard scaler."""

    if matrix.feature_columns != scaler.feature_columns:
        raise ValueError("matrix feature columns must match scaler feature columns")
    return scaler.transform(matrix)


def make_sliding_windows(matrix: FeatureMatrix, window_size: int, stride: int = 1) -> WindowedFeatures:
    """Create overlapping windows shaped as [window, timestep, feature]."""

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if matrix.row_count < window_size:
        raise ValueError("matrix has fewer rows than window_size")

    starts = tuple(range(0, matrix.row_count - window_size + 1, stride))
    windows = [
        [list(row) for row in matrix.values[start : start + window_size]]
        for start in starts
    ]
    return WindowedFeatures(
        windows=windows,
        feature_columns=matrix.feature_columns,
        window_size=window_size,
        stride=stride,
        start_indices=starts,
    )


def sequential_train_test_split(matrix: FeatureMatrix, train_fraction: float = 0.8) -> FeatureSplit:
    """Split a feature matrix without shuffling, preserving time order."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if matrix.row_count < 2:
        raise ValueError("matrix must contain at least two rows")

    split_index = int(matrix.row_count * train_fraction)
    split_index = min(max(split_index, 1), matrix.row_count - 1)
    return FeatureSplit(
        train=FeatureMatrix(matrix.values[:split_index], matrix.feature_columns),
        test=FeatureMatrix(matrix.values[split_index:], matrix.feature_columns),
        split_index=split_index,
    )


def _validate_matrix_shape(values: Sequence[Sequence[float]], column_count: int) -> None:
    for row in values:
        if len(row) != column_count:
            raise ValueError("all rows must match the feature column count")


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items)


def _safe_std(values: Iterable[float], mean: float) -> float:
    items = list(values)
    variance = sum((value - mean) ** 2 for value in items) / len(items)
    std = variance**0.5
    return std if std > 0.0 else 1.0
