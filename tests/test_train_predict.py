from pathlib import Path

import pandas as pd

from nano_tox_agent.schema import PredictionInput, TARGET_CLASSIFICATION
from nano_tox_agent.train import build_model_bundle, derive_training_labels, load_training_data, save_model_bundle


def test_load_training_data_has_expected_columns():
    frame = load_training_data(Path("data/processed/toxicity_clean.csv"))
    assert len(frame) >= 100
    assert "particle_size_nm" in frame.columns
    assert "cell_viability_percent" in frame.columns


def test_derive_training_labels_adds_toxicity_level():
    frame = pd.DataFrame({"cell_viability_percent": [84.0, 70.0, 40.0]})
    labeled = derive_training_labels(frame)
    assert labeled[TARGET_CLASSIFICATION].tolist() == ["low", "medium", "high"]


def test_build_model_bundle_trains_and_saves(tmp_path):
    frame = derive_training_labels(load_training_data(Path("data/processed/toxicity_clean.csv")))
    bundle = build_model_bundle(frame, random_state=7)
    assert bundle.classification_metrics["accuracy"] >= 0.0
    assert bundle.regression_metrics["mae"] >= 0.0
    assert bundle.feature_columns[0] == "particle_size_nm"
    output_path = tmp_path / "toxicity_bundle.joblib"
    save_model_bundle(bundle, output_path)
    assert output_path.exists()
