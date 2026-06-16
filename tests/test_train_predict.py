import os
from pathlib import Path
import subprocess

import pandas as pd

from nano_tox_agent.predict import load_or_train_bundle, predict_toxicity
from nano_tox_agent.schema import PredictionInput, TARGET_CLASSIFICATION
from nano_tox_agent.train import (
    build_model_bundle,
    derive_training_labels,
    load_model_bundle,
    load_training_data,
    save_model_bundle,
)


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
    assert isinstance(bundle.feature_columns, tuple)
    assert bundle.feature_columns[0] == "particle_size_nm"
    output_path = tmp_path / "toxicity_bundle.joblib"
    save_model_bundle(bundle, output_path)
    assert output_path.exists()
    loaded_bundle = load_model_bundle(output_path)
    assert loaded_bundle.feature_columns == bundle.feature_columns
    assert loaded_bundle.classification_metrics == bundle.classification_metrics
    assert loaded_bundle.regression_metrics == bundle.regression_metrics


def test_predict_toxicity_returns_model_owned_result(tmp_path):
    frame = derive_training_labels(load_training_data(Path("data/processed/toxicity_clean.csv")))
    bundle = build_model_bundle(frame, random_state=7)
    sample = PredictionInput(90.0, -8.0, 50.0, 24.0, "liposome", "PEG", "A549", "Human", "Lung", "MTT")
    result = predict_toxicity(bundle, sample)
    assert result.toxicity_level in {"low", "medium", "high"}
    assert 0.0 <= result.cell_viability_percent <= 100.0
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.top_features) == 5
    model_path = tmp_path / "toxicity_bundle.joblib"
    trained = load_or_train_bundle(model_path, Path("data/processed/toxicity_clean.csv"))
    assert trained.feature_columns == bundle.feature_columns


def test_predict_toxicity_uses_model_outputs():
    calls = {"predict": 0, "predict_proba": 0, "regressor": 0}

    class FakePreprocessor:
        def get_feature_names_out(self):
            return [
                "numeric__particle_size_nm",
                "numeric__zeta_potential_mv",
                "numeric__dose_ug_ml",
                "numeric__exposure_time_h",
                "categorical__material_type_liposome",
                "categorical__surface_modification_PEG",
            ]

    class FakeClassifierModel:
        feature_importances_ = [0.01, 0.2, 0.05, 0.4, 0.25, 0.09]

    class FakeClassifier:
        named_steps = {"preprocess": FakePreprocessor(), "model": FakeClassifierModel()}

        def predict(self, frame):
            calls["predict"] += 1
            assert frame.columns.tolist() == list(FakeBundle.feature_columns)
            assert frame.iloc[0].to_dict()["particle_size_nm"] == 90.0
            return ["medium"]

        def predict_proba(self, frame):
            calls["predict_proba"] += 1
            assert frame.columns.tolist() == list(FakeBundle.feature_columns)
            return [[0.1, 0.7, 0.2]]

    class FakeRegressor:
        def predict(self, frame):
            calls["regressor"] += 1
            assert frame.columns.tolist() == list(FakeBundle.feature_columns)
            return [37.654]

    class FakeBundle:
        feature_columns = (
            "particle_size_nm",
            "zeta_potential_mv",
            "dose_ug_ml",
            "exposure_time_h",
            "material_type",
            "surface_modification",
            "cell_type",
            "species",
            "tissue",
            "assay_method",
        )
        classifier = FakeClassifier()
        regressor = FakeRegressor()

    sample = PredictionInput(90.0, -8.0, 50.0, 24.0, "liposome", "PEG", "A549", "Human", "Lung", "MTT")
    result = predict_toxicity(FakeBundle(), sample)

    assert result.toxicity_level == "medium"
    assert result.cell_viability_percent == 37.65
    assert result.confidence == 0.7
    assert result.top_features[0] == ("numeric__exposure_time_h", 0.4)
    assert len(result.top_features) == 5
    assert calls == {"predict": 1, "predict_proba": 1, "regressor": 1}


def test_load_or_train_bundle_loads_existing_bundle(tmp_path, monkeypatch):
    model_path = tmp_path / "toxicity_bundle.joblib"
    model_path.write_text("sentinel")
    sentinel = object()
    calls = {"load": 0, "build": 0, "save": 0}

    def fake_load_model_bundle(path):
        calls["load"] += 1
        assert path == model_path
        return sentinel

    def fake_build_model_bundle(*args, **kwargs):
        calls["build"] += 1
        raise AssertionError("build_model_bundle should not run when model file exists")

    def fake_save_model_bundle(*args, **kwargs):
        calls["save"] += 1
        raise AssertionError("save_model_bundle should not run when model file exists")

    monkeypatch.setattr("nano_tox_agent.predict.load_model_bundle", fake_load_model_bundle)
    monkeypatch.setattr("nano_tox_agent.predict.build_model_bundle", fake_build_model_bundle)
    monkeypatch.setattr("nano_tox_agent.predict.save_model_bundle", fake_save_model_bundle)

    loaded = load_or_train_bundle(model_path, Path("data/processed/toxicity_clean.csv"))

    assert loaded is sentinel
    assert calls == {"load": 1, "build": 0, "save": 0}


def test_package_import_works_from_repo_root(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    command = ["uv", "run", "--no-sync", "python", "-c", "import nano_tox_agent; print(nano_tox_agent.__version__)"]
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")
    env["XDG_CACHE_HOME"] = str(tmp_path / "xdg-cache")
    result = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, env=env)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.1.0"


def test_package_import_works_with_venv_python():
    repo_root = Path(__file__).resolve().parents[1]
    command = [str(repo_root / ".venv/bin/python"), "-c", "import nano_tox_agent; print(nano_tox_agent.__version__)"]
    result = subprocess.run(command, cwd=repo_root, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.1.0"


def test_prediction_behavior_is_deterministic_for_same_input():
    frame = derive_training_labels(load_training_data(Path("data/processed/toxicity_clean.csv")))
    bundle = build_model_bundle(frame, random_state=7)
    sample = PredictionInput(90.0, -8.0, 50.0, 24.0, "liposome", "PEG", "A549", "Human", "Lung", "MTT")

    first = predict_toxicity(bundle, sample)
    second = predict_toxicity(bundle, sample)

    assert first == second
    assert first.toxicity_level in {"low", "medium", "high"}
    assert 0.0 <= first.cell_viability_percent <= 100.0
