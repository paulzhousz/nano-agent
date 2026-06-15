import pytest

from nano_tox_agent import __version__
from nano_tox_agent.schema import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PredictionInput,
    classify_viability,
    validate_prediction_input,
)


def make_sample(**overrides):
    values = {
        "particle_size_nm": 90.0,
        "zeta_potential_mv": -8.0,
        "dose_ug_ml": 50.0,
        "exposure_time_h": 24.0,
        "material_type": "liposome",
        "surface_modification": "PEG",
        "cell_type": "A549",
        "species": "Human",
        "tissue": "Lung",
        "assay_method": "MTT",
    }
    values.update(overrides)
    return PredictionInput(**values)


def test_package_imports():
    assert __version__ == "0.1.0"


def test_feature_lists_are_stable():
    assert NUMERIC_FEATURES == ["particle_size_nm", "zeta_potential_mv", "dose_ug_ml", "exposure_time_h"]
    assert CATEGORICAL_FEATURES == [
        "material_type",
        "surface_modification",
        "cell_type",
        "species",
        "tissue",
        "assay_method",
    ]


def test_classify_viability_thresholds():
    assert classify_viability(80.0) == "low"
    assert classify_viability(79.9) == "medium"
    assert classify_viability(50.0) == "medium"
    assert classify_viability(49.9) == "high"


def test_validate_prediction_input_accepts_valid_sample():
    sample = make_sample()
    assert validate_prediction_input(sample).cell_type == "A549"


def test_validate_prediction_input_rejects_invalid_numeric_ranges():
    sample = make_sample(particle_size_nm=-1.0)
    with pytest.raises(ValueError, match="particle_size_nm must be greater than 0"):
        validate_prediction_input(sample)


def test_validate_prediction_input_rejects_negative_dose():
    sample = make_sample(dose_ug_ml=-0.1)
    with pytest.raises(ValueError, match="dose_ug_ml must be greater than or equal to 0"):
        validate_prediction_input(sample)


def test_validate_prediction_input_rejects_non_positive_exposure_time():
    sample = make_sample(exposure_time_h=0.0)
    with pytest.raises(ValueError, match="exposure_time_h must be greater than 0"):
        validate_prediction_input(sample)


def test_validate_prediction_input_rejects_blank_categorical_fields():
    sample = make_sample(cell_type="   ")
    with pytest.raises(ValueError, match="cell_type must not be empty"):
        validate_prediction_input(sample)


def test_validate_prediction_input_rejects_non_string_categorical_fields():
    sample = make_sample(cell_type=123)
    with pytest.raises(ValueError, match="cell_type must be a string"):
        validate_prediction_input(sample)


@pytest.mark.parametrize("field_name", NUMERIC_FEATURES)
def test_validate_prediction_input_rejects_nan_numeric_fields(field_name):
    sample = make_sample(**{field_name: float("nan")})
    with pytest.raises(ValueError, match=f"{field_name} must be finite"):
        validate_prediction_input(sample)


@pytest.mark.parametrize("field_name", NUMERIC_FEATURES)
@pytest.mark.parametrize(
    ("value", "message"),
    [
        (True, "must be a number"),
        ("1.0", "must be a number"),
        (float("inf"), "must be finite"),
        (float("-inf"), "must be finite"),
    ],
)
def test_validate_prediction_input_rejects_invalid_numeric_field_types(field_name, value, message):
    sample = make_sample(**{field_name: value})
    with pytest.raises(ValueError, match=f"{field_name} {message}"):
        validate_prediction_input(sample)


def test_classify_viability_rejects_nan():
    with pytest.raises(ValueError, match="cell_viability_percent must be finite"):
        classify_viability(float("nan"))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (True, "cell_viability_percent must be a number"),
        (float("inf"), "cell_viability_percent must be finite"),
    ],
)
def test_classify_viability_rejects_invalid_numeric_values(value, message):
    with pytest.raises(ValueError, match=message):
        classify_viability(value)
