import pytest

from nano_tox_agent import __version__
from nano_tox_agent.schema import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PredictionInput,
    classify_viability,
    validate_prediction_input,
)


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
    sample = PredictionInput(90.0, -8.0, 50.0, 24.0, "liposome", "PEG", "A549", "Human", "Lung", "MTT")
    assert validate_prediction_input(sample).cell_type == "A549"


def test_validate_prediction_input_rejects_invalid_numeric_ranges():
    sample = PredictionInput(-1.0, -8.0, 50.0, 24.0, "liposome", "PEG", "A549", "Human", "Lung", "MTT")
    with pytest.raises(ValueError, match="particle_size_nm must be greater than 0"):
        validate_prediction_input(sample)
