import pytest

from gpu_rack_anomaly.features import (
    FEATURE_COLUMNS,
    FeatureMatrix,
    fit_standard_scaler,
    make_sliding_windows,
    normalize_features,
    sequential_train_test_split,
    telemetry_window_to_feature_matrix,
)
from gpu_rack_anomaly.simulate_telemetry import simulate_window


def test_extracts_default_feature_columns_in_stable_order() -> None:
    telemetry = simulate_window("normal", window_size=3, seed=3)

    matrix = telemetry_window_to_feature_matrix(telemetry)

    assert matrix.feature_columns == FEATURE_COLUMNS
    assert matrix.row_count == 3
    assert matrix.column_count == len(FEATURE_COLUMNS)
    assert matrix.values[0][0] == telemetry.samples[0].gpu_temperature_c
    assert matrix.values[0][-1] == telemetry.samples[0].hotspot_persistence_seconds


def test_extracts_custom_feature_columns() -> None:
    telemetry = simulate_window("normal", window_size=2, seed=3)

    matrix = telemetry_window_to_feature_matrix(
        telemetry,
        feature_columns=("rack_power_kw", "thermal_hotspot_score"),
    )

    assert matrix.feature_columns == ("rack_power_kw", "thermal_hotspot_score")
    assert matrix.values == [
        [
            telemetry.samples[0].rack_power_kw,
            telemetry.samples[0].thermal_hotspot_score,
        ],
        [
            telemetry.samples[1].rack_power_kw,
            telemetry.samples[1].thermal_hotspot_score,
        ],
    ]


def test_rejects_unknown_or_non_numeric_features() -> None:
    telemetry = simulate_window("normal", window_size=1, seed=3)

    with pytest.raises(ValueError, match="unknown feature column"):
        telemetry_window_to_feature_matrix(telemetry, feature_columns=("not_a_signal",))

    with pytest.raises(TypeError, match="not numeric"):
        telemetry_window_to_feature_matrix(telemetry, feature_columns=("rack_id",))


def test_standard_scaler_normalizes_columns_and_can_inverse_transform() -> None:
    matrix = FeatureMatrix(
        values=[
            [1.0, 10.0, 5.0],
            [2.0, 20.0, 5.0],
            [3.0, 30.0, 5.0],
        ],
        feature_columns=("a", "b", "constant"),
    )

    scaler = fit_standard_scaler(matrix)
    normalized = normalize_features(matrix, scaler)
    restored = scaler.inverse_transform(normalized)

    assert scaler.means == (2.0, 20.0, 5.0)
    assert scaler.stds[2] == 1.0
    assert _column_mean(normalized.values, 0) == pytest.approx(0.0)
    assert _column_mean(normalized.values, 1) == pytest.approx(0.0)
    assert [row[2] for row in normalized.values] == [0.0, 0.0, 0.0]
    for restored_row, expected_row in zip(restored.values, matrix.values, strict=True):
        assert restored_row == pytest.approx(expected_row)


def test_normalization_requires_matching_feature_columns() -> None:
    matrix = FeatureMatrix(values=[[1.0]], feature_columns=("a",))
    scaler = fit_standard_scaler(matrix)
    mismatched = FeatureMatrix(values=[[1.0]], feature_columns=("b",))

    with pytest.raises(ValueError, match="feature columns"):
        normalize_features(mismatched, scaler)


def test_sliding_window_generation_uses_window_size_and_stride() -> None:
    matrix = FeatureMatrix(
        values=[[float(index)] for index in range(6)],
        feature_columns=("signal",),
    )

    windowed = make_sliding_windows(matrix, window_size=3, stride=2)

    assert windowed.window_size == 3
    assert windowed.stride == 2
    assert windowed.start_indices == (0, 2)
    assert windowed.windows == [
        [[0.0], [1.0], [2.0]],
        [[2.0], [3.0], [4.0]],
    ]


def test_sliding_windows_reject_invalid_sizes() -> None:
    matrix = FeatureMatrix(values=[[1.0], [2.0]], feature_columns=("signal",))

    with pytest.raises(ValueError, match="window_size"):
        make_sliding_windows(matrix, window_size=0)

    with pytest.raises(ValueError, match="stride"):
        make_sliding_windows(matrix, window_size=1, stride=0)

    with pytest.raises(ValueError, match="fewer rows"):
        make_sliding_windows(matrix, window_size=3)


def test_sequential_train_test_split_preserves_order() -> None:
    matrix = FeatureMatrix(
        values=[[float(index)] for index in range(10)],
        feature_columns=("signal",),
    )

    split = sequential_train_test_split(matrix, train_fraction=0.6)

    assert split.split_index == 6
    assert split.train.values == [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    assert split.test.values == [[6.0], [7.0], [8.0], [9.0]]


def _column_mean(values: list[list[float]], column_index: int) -> float:
    return sum(row[column_index] for row in values) / len(values)
