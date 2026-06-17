"""模型训练与持久化。

这里负责读取清洗后的训练数据、派生毒性标签、训练分类/回归双模型，
并把模型与评估指标一起打包成可复用的 bundle。
"""

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
    """把预测所需模型、特征列和离线评估指标打包到一起。"""

    classifier: Pipeline
    regressor: Pipeline
    feature_columns: tuple[str, ...]
    classification_metrics: dict[str, float]
    regression_metrics: dict[str, float]


def load_training_data(path: Path) -> pd.DataFrame:
    """读取训练 CSV，并确认关键字段齐全。"""
    frame = pd.read_csv(path)
    missing = [column for column in FEATURE_COLUMNS + [TARGET_REGRESSION] if column not in frame.columns]
    if missing:
        raise ValueError(f"training data missing columns: {missing}")
    return frame


def derive_training_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """基于连续活力值派生离散毒性等级，供分类器训练使用。"""
    labeled = frame.copy()
    labeled[TARGET_CLASSIFICATION] = labeled[TARGET_REGRESSION].map(classify_viability)
    return labeled


def build_model_bundle(frame: pd.DataFrame, random_state: int = 42) -> ModelBundle:
    """训练分类器与回归器，并返回包含指标的统一 bundle。"""
    x = frame[FEATURE_COLUMNS]
    y_class = frame[TARGET_CLASSIFICATION]
    y_reg = frame[TARGET_REGRESSION]
    x_train, x_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        x, y_class, y_reg, test_size=0.25, random_state=random_state, stratify=y_class
    )
    # 先在留出的测试集上评估，再用全量数据重训最终上线模型。
    evaluation_classifier = Pipeline(
        [
            ("preprocess", make_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100, random_state=random_state, class_weight="balanced", bootstrap=False
                ),
            ),
        ]
    )
    evaluation_regressor = Pipeline(
        [
            ("preprocess", make_preprocessor()),
            ("model", RandomForestRegressor(n_estimators=100, random_state=random_state, bootstrap=False)),
        ]
    )
    evaluation_classifier.fit(x_train, y_class_train)
    evaluation_regressor.fit(x_train, y_reg_train)
    class_pred = evaluation_classifier.predict(x_test)
    reg_pred = evaluation_regressor.predict(x_test)

    final_classifier = Pipeline(
        [
            ("preprocess", make_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100, random_state=random_state, class_weight="balanced", bootstrap=False
                ),
            ),
        ]
    )
    final_regressor = Pipeline(
        [
            ("preprocess", make_preprocessor()),
            ("model", RandomForestRegressor(n_estimators=100, random_state=random_state, bootstrap=False)),
        ]
    )
    final_classifier.fit(x, y_class)
    final_regressor.fit(x, y_reg)

    return ModelBundle(
        classifier=final_classifier,
        regressor=final_regressor,
        feature_columns=tuple(FEATURE_COLUMNS),
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
    """把训练好的 bundle 序列化到磁盘。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model_bundle(path: Path) -> ModelBundle:
    """加载已训练模型，并校验文件内容类型。"""
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise ValueError("model bundle file has an invalid format")
    return bundle
