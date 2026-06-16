from pathlib import Path

import pandas as pd

from nano_tox_agent.schema import PredictionInput, PredictionResult, validate_prediction_input
from nano_tox_agent.train import (
    ModelBundle,
    build_model_bundle,
    derive_training_labels,
    load_model_bundle,
    load_training_data,
    save_model_bundle,
)


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
