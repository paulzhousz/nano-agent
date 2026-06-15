# Nano Toxicity Agent Modeling and Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train classifier/regressor pipelines and expose a prediction service that returns model-owned toxicity results.

**Architecture:** This subplan depends on subplan 01 for schema and demo data. It produces reusable training and prediction modules without Streamlit or LLM code.

**Tech Stack:** pandas, scikit-learn, joblib, pytest.

---

## Files

- Create: `src/nano_tox_agent/preprocess.py`
- Create: `src/nano_tox_agent/train.py`
- Create: `src/nano_tox_agent/predict.py`
- Create: `tests/test_train_predict.py`

## Task 1: Training Pipeline

- [ ] **Step 1: Write failing training tests**

Create `tests/test_train_predict.py`:

```python
from pathlib import Path

import pandas as pd

from nano_tox_agent.schema import PredictionInput, TARGET_CLASSIFICATION
from nano_tox_agent.train import build_model_bundle, derive_training_labels, load_training_data, save_model_bundle


def test_load_training_data_has_expected_columns():
    frame = load_training_data(Path("data/raw/toxicity_samples.csv"))
    assert len(frame) == 20
    assert "particle_size_nm" in frame.columns
    assert "cell_viability_percent" in frame.columns


def test_derive_training_labels_adds_toxicity_level():
    frame = pd.DataFrame({"cell_viability_percent": [84.0, 70.0, 40.0]})
    labeled = derive_training_labels(frame)
    assert labeled[TARGET_CLASSIFICATION].tolist() == ["low", "medium", "high"]


def test_build_model_bundle_trains_and_saves(tmp_path):
    frame = derive_training_labels(load_training_data(Path("data/raw/toxicity_samples.csv")))
    bundle = build_model_bundle(frame, random_state=7)
    assert bundle.classification_metrics["accuracy"] >= 0.0
    assert bundle.regression_metrics["mae"] >= 0.0
    assert bundle.feature_columns[0] == "particle_size_nm"
    output_path = tmp_path / "toxicity_bundle.joblib"
    save_model_bundle(bundle, output_path)
    assert output_path.exists()
```

- [ ] **Step 2: Verify failure**

Run: `rtk pytest tests/test_train_predict.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'nano_tox_agent.train'`.

- [ ] **Step 3: Create preprocessing module**

Create `src/nano_tox_agent/preprocess.py`:

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nano_tox_agent.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
```

- [ ] **Step 4: Create training module**

Create `src/nano_tox_agent/train.py`:

```python
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from nano_tox_agent.preprocess import make_preprocessor
from nano_tox_agent.schema import FEATURE_COLUMNS, TARGET_CLASSIFICATION, TARGET_REGRESSION, classify_viability


@dataclass(frozen=True)
class ModelBundle:
    classifier: Pipeline
    regressor: Pipeline
    feature_columns: list[str]
    classification_metrics: dict[str, float]
    regression_metrics: dict[str, float]


def load_training_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in FEATURE_COLUMNS + [TARGET_REGRESSION] if column not in frame.columns]
    if missing:
        raise ValueError(f"training data missing columns: {missing}")
    return frame


def derive_training_labels(frame: pd.DataFrame) -> pd.DataFrame:
    labeled = frame.copy()
    labeled[TARGET_CLASSIFICATION] = labeled[TARGET_REGRESSION].map(classify_viability)
    return labeled


def build_model_bundle(frame: pd.DataFrame, random_state: int = 42) -> ModelBundle:
    x = frame[FEATURE_COLUMNS]
    y_class = frame[TARGET_CLASSIFICATION]
    y_reg = frame[TARGET_REGRESSION]
    x_train, x_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        x, y_class, y_reg, test_size=0.25, random_state=random_state, stratify=y_class
    )
    evaluation_classifier = Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight="balanced", bootstrap=False)),
    ])
    evaluation_regressor = Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", RandomForestRegressor(n_estimators=100, random_state=random_state, bootstrap=False)),
    ])
    evaluation_classifier.fit(x_train, y_class_train)
    evaluation_regressor.fit(x_train, y_reg_train)
    class_pred = evaluation_classifier.predict(x_test)
    reg_pred = evaluation_regressor.predict(x_test)

    final_classifier = Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight="balanced", bootstrap=False)),
    ])
    final_regressor = Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", RandomForestRegressor(n_estimators=100, random_state=random_state, bootstrap=False)),
    ])
    final_classifier.fit(x, y_class)
    final_regressor.fit(x, y_reg)

    return ModelBundle(
        classifier=final_classifier,
        regressor=final_regressor,
        feature_columns=FEATURE_COLUMNS,
        classification_metrics={
            "accuracy": float(accuracy_score(y_class_test, class_pred)),
            "f1_macro": float(f1_score(y_class_test, class_pred, average="macro")),
        },
        regression_metrics={
            "mae": float(mean_absolute_error(y_reg_test, reg_pred)),
            "r2": float(r2_score(y_reg_test, reg_pred)),
        },
    )


def save_model_bundle(bundle: ModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model_bundle(path: Path) -> ModelBundle:
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise ValueError("model bundle file has an invalid format")
    return bundle
```

- [ ] **Step 5: Verify training**

Run: `rtk pytest tests/test_train_predict.py -q`

Expected: PASS with `3 passed`.

## Task 2: Prediction Service

- [ ] **Step 1: Add failing prediction test**

Append to `tests/test_train_predict.py`:

```python
from nano_tox_agent.predict import load_or_train_bundle, predict_toxicity


def test_predict_toxicity_returns_model_owned_result(tmp_path):
    frame = derive_training_labels(load_training_data(Path("data/raw/toxicity_samples.csv")))
    bundle = build_model_bundle(frame, random_state=7)
    sample = PredictionInput(90.0, -8.0, 50.0, 24.0, "liposome", "PEG", "A549", "Human", "Lung", "MTT")
    result = predict_toxicity(bundle, sample)
    assert result.toxicity_level in {"low", "medium", "high"}
    assert 0.0 <= result.cell_viability_percent <= 100.0
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.top_features) == 5
    model_path = tmp_path / "toxicity_bundle.joblib"
    trained = load_or_train_bundle(model_path, Path("data/raw/toxicity_samples.csv"))
    assert trained.feature_columns == bundle.feature_columns


def test_prediction_behavior_matches_demo_risk_direction():
    frame = derive_training_labels(load_training_data(Path("data/raw/toxicity_samples.csv")))
    bundle = build_model_bundle(frame, random_state=7)
    high_risk = PredictionInput(25.0, 30.0, 120.0, 48.0, "silver", "unmodified", "THP-1", "Human", "Blood", "MTT")
    lower_risk = PredictionInput(150.0, -20.0, 10.0, 24.0, "liposome", "PEG", "BEAS-2B", "Human", "Lung", "MTT")

    high_result = predict_toxicity(bundle, high_risk)
    lower_result = predict_toxicity(bundle, lower_risk)

    assert high_result.toxicity_level != "low"
    assert lower_result.toxicity_level != "high"
    assert high_result.toxicity_level == "high"
    assert lower_result.toxicity_level == "low"
    assert high_result.cell_viability_percent < lower_result.cell_viability_percent
```

- [ ] **Step 2: Verify failure**

Run: `rtk pytest tests/test_train_predict.py::test_predict_toxicity_returns_model_owned_result -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'nano_tox_agent.predict'`.

- [ ] **Step 3: Create prediction service**

Create `src/nano_tox_agent/predict.py`:

```python
from pathlib import Path

import pandas as pd

from nano_tox_agent.schema import PredictionInput, PredictionResult, validate_prediction_input
from nano_tox_agent.train import ModelBundle, build_model_bundle, derive_training_labels, load_model_bundle, load_training_data, save_model_bundle


def predict_toxicity(bundle: ModelBundle, sample: PredictionInput) -> PredictionResult:
    validated = validate_prediction_input(sample)
    frame = pd.DataFrame([validated.to_dict()], columns=bundle.feature_columns)
    toxicity_level = str(bundle.classifier.predict(frame)[0])
    confidence = float(max(bundle.classifier.predict_proba(frame)[0]))
    viability = max(0.0, min(100.0, float(bundle.regressor.predict(frame)[0])))
    return PredictionResult(toxicity_level, round(viability, 2), round(confidence, 3), _top_feature_importances(bundle, 5))


def load_or_train_bundle(model_path: Path, data_path: Path) -> ModelBundle:
    if model_path.exists():
        return load_model_bundle(model_path)
    frame = derive_training_labels(load_training_data(data_path))
    bundle = build_model_bundle(frame)
    save_model_bundle(bundle, model_path)
    return bundle


def _top_feature_importances(bundle: ModelBundle, limit: int) -> list[tuple[str, float]]:
    preprocessor = bundle.classifier.named_steps["preprocess"]
    model = bundle.classifier.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    pairs = sorted(zip(names, model.feature_importances_), key=lambda item: item[1], reverse=True)
    return [(str(name), round(float(score), 4)) for name, score in pairs[:limit]]
```

- [ ] **Step 4: Verify subplan**

Run: `rtk pytest tests/test_train_predict.py -q`

Expected: PASS with `5 passed`.

- [ ] **Step 5: Commit**

```bash
rtk git add src/nano_tox_agent/preprocess.py src/nano_tox_agent/train.py src/nano_tox_agent/predict.py tests/test_train_predict.py
rtk git commit -m "feat: add model training and prediction service"
```
