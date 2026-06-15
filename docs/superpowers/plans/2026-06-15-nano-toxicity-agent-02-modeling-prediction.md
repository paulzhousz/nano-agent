# 纳米毒性预测智能体建模与预测实现计划

> **给智能体执行者：** 必须使用子技能：`superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，按任务逐步执行本计划。步骤使用复选框（`- [ ]`）语法跟踪。

**目标：** 训练分类/回归 pipeline，并暴露一个返回模型自有毒性结论的预测服务。

**架构：** 本子计划依赖子计划 01 提供的 schema、公开数据来源契约和完整公开数据清洗结果。它产出可复用的训练与预测模块，不包含 Streamlit 或 LLM 代码。

**技术栈：** Python 3.12、uv、pandas、scikit-learn、joblib、pytest。

---

## 文件

- 创建：`src/nano_tox_agent/preprocess.py`
- 创建：`src/nano_tox_agent/train.py`
- 创建：`src/nano_tox_agent/predict.py`
- 创建：`tests/test_train_predict.py`

## 任务 1：训练 Pipeline

- [ ] **步骤 1：编写失败的训练测试**

创建 `tests/test_train_predict.py`：

```python
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
```

- [ ] **步骤 2：验证失败**

运行：`rtk uv run pytest tests/test_train_predict.py -q`

预期：失败，报 `ModuleNotFoundError: No module named 'nano_tox_agent.train'`。

- [ ] **步骤 3：创建预处理模块**

创建 `src/nano_tox_agent/preprocess.py`：

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

- [ ] **步骤 4：创建训练模块**

创建 `src/nano_tox_agent/train.py`：

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

- [ ] **步骤 5：验证训练**

运行：`rtk uv run pytest tests/test_train_predict.py -q`

预期：通过，显示 `3 passed`。

## 任务 2：预测服务

- [ ] **步骤 1：添加失败的预测测试**

追加到 `tests/test_train_predict.py`：

```python
from nano_tox_agent.predict import load_or_train_bundle, predict_toxicity


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


def test_prediction_behavior_is_deterministic_for_same_input():
    frame = derive_training_labels(load_training_data(Path("data/processed/toxicity_clean.csv")))
    bundle = build_model_bundle(frame, random_state=7)
    sample = PredictionInput(90.0, -8.0, 50.0, 24.0, "liposome", "PEG", "A549", "Human", "Lung", "MTT")

    first = predict_toxicity(bundle, sample)
    second = predict_toxicity(bundle, sample)

    assert first == second
    assert first.toxicity_level in {"low", "medium", "high"}
    assert 0.0 <= first.cell_viability_percent <= 100.0
```

- [ ] **步骤 2：验证失败**

运行：`rtk uv run pytest tests/test_train_predict.py::test_predict_toxicity_returns_model_owned_result -q`

预期：失败，报 `ModuleNotFoundError: No module named 'nano_tox_agent.predict'`。

- [ ] **步骤 3：创建预测服务**

创建 `src/nano_tox_agent/predict.py`：

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

- [ ] **步骤 4：验证子计划**

运行：`rtk uv run pytest tests/test_train_predict.py -q`

预期：通过，显示 `5 passed`。

- [ ] **步骤 5：提交**

```bash
rtk git add src/nano_tox_agent/preprocess.py src/nano_tox_agent/train.py src/nano_tox_agent/predict.py tests/test_train_predict.py
rtk git commit -m "feat: add model training and prediction service"
```
