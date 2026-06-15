import json
from pathlib import Path

import pandas as pd

from nano_tox_agent.schema import FEATURE_COLUMNS, TARGET_REGRESSION


EXPECTED_SOURCE_NAMES = ["eNanoMapper", "caNanoLab", "Curated nanotoxicity ML datasets"]
EXPECTED_COLUMNS = FEATURE_COLUMNS + [TARGET_REGRESSION]
EXPECTED_DEFERRED_SOURCES = ["caNanoLab", "Curated nanotoxicity ML datasets"]


def test_public_data_sources_policy_and_priority_order():
    sources = json.loads(Path("data/raw/data_sources.json").read_text(encoding="utf-8"))
    assert [entry["name"] for entry in sources["priority_order"]] == EXPECTED_SOURCE_NAMES
    assert sources["required_normalized_columns"] == EXPECTED_COLUMNS
    assert sources["source_policy"].startswith("第一版必须接入完整公开数据清洗结果")
    assert sources["active_training_sources"] == ["eNanoMapper"]
    assert sources["deferred_sources"] == EXPECTED_DEFERRED_SOURCES


def test_public_data_scripts_exist():
    assert Path("scripts/clean_public_data.py").exists()
    assert Path("scripts/build_literature_base.py").exists()


def test_only_active_training_source_requires_raw_csv_exports():
    enanomapper_exports = list(Path("data/raw/source_exports/enanomapper").glob("*.csv"))
    assert {path.name for path in enanomapper_exports} >= {
        "enanomapper_viability_2026-06-15.csv",
        "enanomapper_conditions_2026-06-15.csv",
        "enanomapper_pchem_2026-06-15.csv",
    }
    for deferred_dir in ["cananolab", "curated_ml"]:
        directory = Path("data/raw/source_exports") / deferred_dir
        assert (directory / "README.md").exists()
        assert list(directory.glob("*.csv")) == []


def test_cleaned_public_data_exists_with_expected_columns_and_records():
    frame = pd.read_csv("data/processed/toxicity_clean.csv")
    assert list(frame.columns) == EXPECTED_COLUMNS
    assert len(frame) >= 100
    assert not frame.duplicated().any()


def test_cleaned_public_data_has_valid_ranges_and_required_text():
    frame = pd.read_csv("data/processed/toxicity_clean.csv")
    assert (frame["particle_size_nm"] > 0).all()
    assert (frame["dose_ug_ml"] >= 0).all()
    assert (frame["exposure_time_h"] > 0).all()
    assert frame["cell_viability_percent"].between(0, 100).all()
    for column in ["material_type", "cell_type", "assay_method"]:
        assert frame[column].astype("string").str.strip().ne("").all()


def test_cleaned_public_data_metadata_matches_frame():
    frame = pd.read_csv("data/processed/toxicity_clean.csv")
    metadata = json.loads(Path("data/processed/toxicity_clean_metadata.json").read_text(encoding="utf-8"))
    assert metadata["dataset_type"] == "cleaned_public_nanotoxicity_dataset"
    assert metadata["record_count"] == len(frame)
    assert metadata["record_count"] >= 100
    assert sum(metadata["source_record_counts"].values()) >= metadata["record_count"]
    assert metadata["cleaned_output_path"] == "data/processed/toxicity_clean.csv"
    assert metadata["training_sources"] == ["eNanoMapper"]
    assert metadata["deferred_sources"] == EXPECTED_DEFERRED_SOURCES
    assert "合成" not in metadata["purpose"]
    assert "演示样例" not in metadata["purpose"]
