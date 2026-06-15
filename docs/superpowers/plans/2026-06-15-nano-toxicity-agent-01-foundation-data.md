# Nano Toxicity Agent Foundation and Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the project skeleton, domain schema, demo toxicity data, fixed literature library, and tests.

**Architecture:** This subplan establishes stable contracts used by every later subplan. It contains no model training and no UI code.

**Tech Stack:** Python 3.11+, pytest, JSON, CSV.

---

## Files

- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/nano_tox_agent/__init__.py`
- Create: `src/nano_tox_agent/schema.py`
- Create: `src/nano_tox_agent/literature.py`
- Create: `tests/test_schema.py`
- Create: `tests/test_literature.py`
- Create: `data/raw/toxicity_samples.csv`
- Create: `data/raw/toxicity_samples_metadata.json`
- Create: `literature/literature_base.json`

## Task 1: Package Skeleton

- [ ] **Step 1: Write failing import test**

Create `tests/test_schema.py`:

```python
from nano_tox_agent import __version__


def test_package_imports():
    assert __version__ == "0.1.0"
```

- [ ] **Step 2: Verify failure**

Run: `rtk pytest tests/test_schema.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'nano_tox_agent'` or dependency installation error if pytest is unavailable.

- [ ] **Step 3: Create package metadata**

Create `pyproject.toml`:

```toml
[project]
name = "nano-tox-agent"
version = "0.1.0"
description = "Nanomaterial cytotoxicity prediction agent prototype"
requires-python = ">=3.11"
dependencies = [
  "joblib>=1.4.2",
  "matplotlib>=3.8.0",
  "numpy>=1.26.0",
  "pandas>=2.2.0",
  "scikit-learn>=1.4.0",
  "streamlit>=1.35.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.2.0"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

Create `.gitignore`:

```gitignore
.DS_Store
.env
.pytest_cache/
.ruff_cache/
.streamlit/secrets.toml
__pycache__/
*.py[cod]
models/*.joblib
models/*.json
data/processed/
.superpowers/
```

Create `src/nano_tox_agent/__init__.py`:

```python
__version__ = "0.1.0"
```

- [ ] **Step 4: Verify package import**

Run:

```bash
rtk python -m pip install -e ".[dev]"
rtk pytest tests/test_schema.py -q
```

Expected: PASS with `1 passed`.

## Task 2: Domain Schema

- [ ] **Step 1: Replace schema tests**

Replace `tests/test_schema.py`:

```python
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
```

- [ ] **Step 2: Verify failure**

Run: `rtk pytest tests/test_schema.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'nano_tox_agent.schema'`.

- [ ] **Step 3: Create schema module**

Create `src/nano_tox_agent/schema.py`:

```python
from dataclasses import asdict, dataclass

NUMERIC_FEATURES = ["particle_size_nm", "zeta_potential_mv", "dose_ug_ml", "exposure_time_h"]
CATEGORICAL_FEATURES = ["material_type", "surface_modification", "cell_type", "species", "tissue", "assay_method"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_REGRESSION = "cell_viability_percent"
TARGET_CLASSIFICATION = "toxicity_level"
TOXICITY_LABELS = ["low", "medium", "high"]


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
    if cell_viability_percent >= 80.0:
        return "low"
    if cell_viability_percent >= 50.0:
        return "medium"
    return "high"


def validate_prediction_input(sample: PredictionInput) -> PredictionInput:
    if sample.particle_size_nm <= 0:
        raise ValueError("particle_size_nm must be greater than 0")
    if sample.dose_ug_ml < 0:
        raise ValueError("dose_ug_ml must be greater than or equal to 0")
    if sample.exposure_time_h <= 0:
        raise ValueError("exposure_time_h must be greater than 0")
    for field_name, value in sample.to_dict().items():
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"{field_name} must not be empty")
    return sample
```

- [ ] **Step 4: Verify schema**

Run: `rtk pytest tests/test_schema.py -q`

Expected: PASS with `5 passed`.

## Task 3: Demo Data and Literature Library

- [ ] **Step 1: Create demo data**

Create `data/raw/toxicity_samples.csv`:

```csv
particle_size_nm,zeta_potential_mv,dose_ug_ml,exposure_time_h,material_type,surface_modification,cell_type,species,tissue,assay_method,cell_viability_percent
30,22,100,24,metal_oxide,unmodified,A549,Human,Lung,MTT,38
45,18,80,24,metal_oxide,citrate,A549,Human,Lung,MTT,55
90,-8,50,24,liposome,PEG,A549,Human,Lung,MTT,85
120,-15,25,12,polymer,PEG,BEAS-2B,Human,Lung,MTT,92
70,5,75,48,metal_oxide,unmodified,HepG2,Human,Liver,LDH,58
25,30,120,48,silver,unmodified,THP-1,Human,Blood,MTT,31
150,-20,10,24,liposome,PEG,BEAS-2B,Human,Lung,MTT,96
60,12,60,24,gold,citrate,A549,Human,Lung,MTT,72
110,-5,40,24,polymer,unmodified,HepG2,Human,Liver,MTT,78
35,25,90,12,metal_oxide,unmodified,NRK-52E,Rat,Kidney,LDH,49
200,-18,15,24,carbon,PEG,BEAS-2B,Human,Lung,MTT,88
80,10,65,72,metal_oxide,citrate,THP-1,Human,Blood,LDH,53
55,16,95,24,zinc_oxide,unmodified,A549,Human,Lung,MTT,42
140,-12,35,48,polymer,PEG,HepG2,Human,Liver,MTT,83
20,28,110,24,silver,citrate,A549,Human,Lung,LDH,36
100,-6,45,12,gold,PEG,BEAS-2B,Human,Lung,MTT,89
65,14,70,24,copper_oxide,unmodified,HepG2,Human,Liver,MTT,57
180,-22,20,24,carbon,PEG,THP-1,Human,Blood,MTT,91
40,19,85,48,zinc_oxide,citrate,NRK-52E,Rat,Kidney,LDH,51
95,-10,55,24,liposome,PEG,A549,Human,Lung,MTT,82
```

- [ ] **Step 2: Create demo data metadata and literature JSON**

Create `data/raw/toxicity_samples_metadata.json`:

```json
{
  "dataset_name": "toxicity_samples.csv",
  "dataset_type": "synthetic_demo",
  "purpose": "Course prototype training and UI demonstration only",
  "scientific_use_limit": "Do not report model performance from this file as validated nanotoxicology evidence.",
  "replacement_path": "Replace with cleaned eNanoMapper, caNanoLab, NanoTox, or PubVINAS records before making scientific claims.",
  "created_for": "Nano Toxicity Agent first runnable prototype",
  "required_report_disclosure": "The first prototype uses a small synthetic demo dataset to validate software flow; real experimental data integration is future work."
}
```

Create `literature/literature_base.json`:

```json
[
  {
    "title": "The eNanoMapper database for nanomaterial safety information",
    "year": 2015,
    "url": "https://www.beilstein-journals.org/bjnano/articles/6/165",
    "used_for": ["data_source", "schema_design"],
    "key_points": [
      "eNanoMapper is designed for nanomaterial safety information management.",
      "The database supports structured nanomaterial characterization and toxicology data.",
      "It can support nanoQSAR and machine learning workflows."
    ]
  },
  {
    "title": "Predicting Cytotoxicity of Nanoparticles: A Meta-Analysis Using Machine Learning",
    "year": 2024,
    "url": "https://doi.org/10.1021/acsanm.4c02269",
    "used_for": ["model_design", "feature_selection", "feature_explanation"],
    "key_points": [
      "Machine learning can be used to predict nanoparticle cytotoxicity.",
      "Composition, concentration, Zeta potential, particle size, and exposure time are common predictive features.",
      "Random forests are suitable for viability prediction and feature importance analysis."
    ]
  },
  {
    "title": "Application of Machine Learning in Nanotoxicology: A Critical Review and Perspective",
    "year": 2024,
    "url": "https://doi.org/10.1021/acs.est.4c03217",
    "used_for": ["background", "limitations"],
    "key_points": [
      "Machine learning can accelerate nanotoxicology analysis.",
      "Nanotoxicology datasets are often small and heterogeneous.",
      "Interpretability and data quality remain central challenges."
    ]
  },
  {
    "title": "caNanoLab: data sharing to expedite the use of nanotechnology in biomedicine",
    "year": 2013,
    "url": "https://cananolab.cancer.gov/",
    "used_for": ["cancer_nanomedicine_context", "data_source"],
    "key_points": [
      "caNanoLab supports cancer nanotechnology data sharing.",
      "The platform provides context for biomedical and cancer nanomaterial applications.",
      "It is useful as background for tumor nanomedicine screening scenarios."
    ]
  },
  {
    "title": "AI and Machine Learning Approaches for Predicting Nanoparticles Toxicity: The Critical Role of Physiochemical Properties",
    "year": 2024,
    "url": "https://arxiv.org/abs/2409.15322",
    "used_for": ["feature_selection", "feature_explanation"],
    "key_points": [
      "Physicochemical properties are central inputs for nanoparticle toxicity prediction.",
      "Particle size, shape, surface charge, composition, and exposure conditions influence toxicity.",
      "Feature selection should reflect both material properties and biological context."
    ]
  },
  {
    "title": "Smart Drug-Delivery Systems for Cancer Nanotherapy",
    "year": 2024,
    "url": "https://arxiv.org/abs/2401.11192",
    "used_for": ["cancer_nanomedicine_context"],
    "key_points": [
      "Nanomedicine has important applications in cancer therapy.",
      "Drug delivery systems require safety screening before downstream validation.",
      "Toxicity prediction can support early-stage candidate filtering."
    ]
  }
]
```

- [ ] **Step 3: Write literature tests**

Create `tests/test_literature.py`:

```python
from pathlib import Path

from nano_tox_agent.literature import load_literature, select_literature


def test_load_literature_reads_entries():
    entries = load_literature(Path("literature/literature_base.json"))
    assert len(entries) == 6
    assert entries[0]["title"] == "The eNanoMapper database for nanomaterial safety information"
    assert "data_source" in entries[0]["used_for"]


def test_select_literature_filters_by_use():
    entries = load_literature(Path("literature/literature_base.json"))
    selected = select_literature(entries, "feature_explanation")
    titles = [entry["title"] for entry in selected]
    assert "AI and Machine Learning Approaches for Predicting Nanoparticles Toxicity: The Critical Role of Physiochemical Properties" in titles
    assert "Smart Drug-Delivery Systems for Cancer Nanotherapy" not in titles


def test_literature_entries_have_traceable_urls():
    entries = load_literature(Path("literature/literature_base.json"))
    for entry in entries:
        assert entry["url"].startswith(("https://doi.org/", "https://www.beilstein-journals.org/", "https://cananolab.cancer.gov/", "https://arxiv.org/"))
        assert entry["url"] != "https://pubs.acs.org/"
```

- [ ] **Step 4: Create literature module**

Create `src/nano_tox_agent/literature.py`:

```python
import json
from pathlib import Path
from typing import Any


def load_literature(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("literature library must be a list")
    return data


def select_literature(entries: list[dict[str, Any]], purpose: str) -> list[dict[str, Any]]:
    return [entry for entry in entries if purpose in entry.get("used_for", [])]
```

- [ ] **Step 5: Verify full subplan**

Run:

```bash
rtk pytest tests/test_schema.py tests/test_literature.py -q
```

Expected: PASS with `8 passed`.

- [ ] **Step 6: Commit**

```bash
rtk git add pyproject.toml .gitignore src/nano_tox_agent tests data/raw/toxicity_samples.csv data/raw/toxicity_samples_metadata.json literature/literature_base.json
rtk git commit -m "feat: add foundation data and literature library"
```
