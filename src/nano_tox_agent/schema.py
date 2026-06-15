from dataclasses import asdict, dataclass
from math import isfinite

NUMERIC_FEATURES = ["particle_size_nm", "zeta_potential_mv", "dose_ug_ml", "exposure_time_h"]
CATEGORICAL_FEATURES = ["material_type", "surface_modification", "cell_type", "species", "tissue", "assay_method"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_REGRESSION = "cell_viability_percent"
TARGET_CLASSIFICATION = "toxicity_level"
TOXICITY_LABELS = ["low", "medium", "high"]


def _validate_finite_number(field_name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    if not isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return float(value)


@dataclass(frozen=True)
class PredictionInput:
    particle_size_nm: float
    zeta_potential_mv: float
    dose_ug_ml: float
    exposure_time_h: float
    material_type: str
    surface_modification: str
    cell_type: str
    species: str
    tissue: str
    assay_method: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionResult:
    toxicity_level: str
    cell_viability_percent: float
    confidence: float
    top_features: list[tuple[str, float]]


def classify_viability(cell_viability_percent: float) -> str:
    cell_viability_percent = _validate_finite_number("cell_viability_percent", cell_viability_percent)
    if cell_viability_percent >= 80.0:
        return "low"
    if cell_viability_percent >= 50.0:
        return "medium"
    return "high"


def validate_prediction_input(sample: PredictionInput) -> PredictionInput:
    values = sample.to_dict()
    for field_name in NUMERIC_FEATURES:
        _validate_finite_number(field_name, values[field_name])
    if sample.particle_size_nm <= 0:
        raise ValueError("particle_size_nm must be greater than 0")
    if sample.dose_ug_ml < 0:
        raise ValueError("dose_ug_ml must be greater than or equal to 0")
    if sample.exposure_time_h <= 0:
        raise ValueError("exposure_time_h must be greater than 0")
    for field_name in CATEGORICAL_FEATURES:
        value = values[field_name]
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string")
        if not value.strip():
            raise ValueError(f"{field_name} must not be empty")
    return sample
