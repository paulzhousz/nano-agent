import json
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from nano_tox_agent.schema import FEATURE_COLUMNS, TARGET_REGRESSION


def _load_clean_public_data_module():
    spec = importlib.util.spec_from_file_location("clean_public_data", Path("scripts/clean_public_data.py"))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


clean_public_data = _load_clean_public_data_module()


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
    curated_source = next(entry for entry in sources["priority_order"] if entry["name"] == "Curated nanotoxicity ML datasets")
    assert "future_prototype_training" in curated_source["expected_use"]
    assert "prototype_training" not in curated_source["expected_use"]


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
    assert "不能外推到所有纳米材料、细胞类型或 assay" in metadata["scientific_use_limit"]
    assert "覆盖范围很窄" in metadata["required_report_disclosure"]


def test_cleaning_script_reads_source_policy_file():
    policy = clean_public_data.load_source_policy(Path("data/raw/data_sources.json"))
    assert policy["active_training_sources"] == ["eNanoMapper"]
    assert policy["deferred_sources"] == EXPECTED_DEFERRED_SOURCES


def test_source_policy_requires_active_deferred_partition(tmp_path):
    policy = {
        "priority_order": [{"name": name} for name in EXPECTED_SOURCE_NAMES],
        "active_training_sources": ["eNanoMapper", "caNanoLab"],
        "deferred_sources": EXPECTED_DEFERRED_SOURCES,
    }
    policy_path = tmp_path / "overlap.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(SystemExit, match="both active and deferred"):
        clean_public_data.load_source_policy(policy_path)

    policy["active_training_sources"] = ["eNanoMapper"]
    policy["deferred_sources"] = ["caNanoLab"]
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(SystemExit, match="must cover priority_order"):
        clean_public_data.load_source_policy(policy_path)


def test_metadata_contains_auditable_cleaning_chain_and_rules():
    metadata = json.loads(Path("data/processed/toxicity_clean_metadata.json").read_text(encoding="utf-8"))
    audit = metadata["cleaning_audit"]
    assert audit["raw_file_row_counts"]["eNanoMapper"]["viability"] == 508
    assert audit["raw_file_row_counts"]["eNanoMapper"]["conditions"] == 4285
    assert audit["raw_file_row_counts"]["eNanoMapper"]["pchem"] == 2649
    assert audit["raw_file_row_counts"]["eNanoMapper"]["params"] == 5621
    assert audit["intermediate_counts"]["enanomapper_viability_condition_join_rows"] >= metadata["record_count"]
    assert audit["intermediate_counts"]["enanomapper_normalized_rows"] == metadata["source_record_counts"]["eNanoMapper"]
    assert audit["drop_filter_counts"]["final_rows_removed_by_required_values_and_ranges"] >= 0
    assert audit["drop_filter_counts"]["duplicates_removed"] >= 0
    assert audit["drop_filter_counts"]["viability_rows_rejected_by_target_endpoint_or_unit"] > 0
    assert metadata["unit_normalization"]["dose_ug_ml"]["allowed_units"]
    assert "mg/l" in metadata["unit_normalization"]["dose_ug_ml"]["allowed_units"]
    assert metadata["unit_normalization"]["exposure_time_h"]["allowed_units"] == ["h", "hr", "hour", "hours"]
    assert metadata["target_endpoint_policy"] == {
        "source": "eNanoMapper viability export",
        "effectendpoint_s": "% CELL VIABILITY",
        "unit_s": "%",
        "normalization": "case-insensitive trim before comparison",
    }
    assert metadata["pchem_size_endpoint_policy"]["endpoint_priority"]
    assert "median within endpoint bucket" in metadata["pchem_size_endpoint_policy"]["aggregation"].lower()
    assert "enanomapper_params_2026-06-15.csv" in metadata["unused_raw_exports"]["eNanoMapper"]["files"]
    assert "保留" in metadata["excluded_record_policy"]
    assert "无法映射材料组成或细胞类型" not in metadata["excluded_record_policy"]


def test_metadata_final_coverage_summary_matches_cleaned_csv():
    frame = pd.read_csv("data/processed/toxicity_clean.csv")
    metadata = json.loads(Path("data/processed/toxicity_clean_metadata.json").read_text(encoding="utf-8"))
    summary = metadata["final_coverage_summary"]
    categorical_columns = [
        "material_type",
        "surface_modification",
        "cell_type",
        "species",
        "tissue",
        "assay_method",
    ]
    numeric_columns = [
        "particle_size_nm",
        "zeta_potential_mv",
        "dose_ug_ml",
        "exposure_time_h",
        "cell_viability_percent",
    ]

    assert summary["record_count"] == len(frame) == metadata["record_count"]
    for column in categorical_columns:
        assert summary["categorical_unique_counts"][column] == frame[column].nunique()
        assert summary["categorical_top_values"][column] == frame[column].value_counts().to_dict()

    for column in numeric_columns:
        assert summary["numeric_ranges"][column]["min"] == pytest.approx(frame[column].min())
        assert summary["numeric_ranges"][column]["max"] == pytest.approx(frame[column].max())

    assert summary["categorical_unique_counts"]["assay_method"] == 1
    assert summary["categorical_unique_counts"]["cell_type"] == 1
    assert summary["categorical_unique_counts"]["species"] == 1
    assert summary["categorical_unique_counts"]["tissue"] == 1


def test_cleaning_rebuilds_committed_processed_outputs_without_rewriting_repo(tmp_path):
    output_path = tmp_path / "toxicity_clean.csv"
    metadata_path = tmp_path / "toxicity_clean_metadata.json"
    clean_public_data.main(processed_output_path=output_path, metadata_output_path=metadata_path)

    rebuilt_frame = pd.read_csv(output_path)
    committed_frame = pd.read_csv("data/processed/toxicity_clean.csv")
    pd.testing.assert_frame_equal(rebuilt_frame, committed_frame)

    rebuilt_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    committed_metadata = json.loads(Path("data/processed/toxicity_clean_metadata.json").read_text(encoding="utf-8"))
    assert rebuilt_metadata["record_count"] == committed_metadata["record_count"] == len(committed_frame)
    assert rebuilt_metadata["source_record_counts"] == committed_metadata["source_record_counts"]
    assert rebuilt_metadata["cleaning_audit"] == committed_metadata["cleaning_audit"]
    assert rebuilt_metadata["target_endpoint_policy"] == committed_metadata["target_endpoint_policy"]
    assert rebuilt_metadata["final_coverage_summary"] == committed_metadata["final_coverage_summary"]


def test_deferred_csv_exports_are_rejected(tmp_path):
    active_dir = tmp_path / "enanomapper"
    deferred_dir = tmp_path / "cananolab"
    other_deferred_dir = tmp_path / "curated_ml"
    for directory in [active_dir, deferred_dir, other_deferred_dir]:
        directory.mkdir()
    (active_dir / "active.csv").write_text("id\n1\n", encoding="utf-8")
    (deferred_dir / "deferred.csv").write_text("id\n1\n", encoding="utf-8")

    source_dirs = {
        "eNanoMapper": active_dir,
        "caNanoLab": deferred_dir,
        "Curated nanotoxicity ML datasets": other_deferred_dir,
    }
    policy = {
        "active_training_sources": ["eNanoMapper"],
        "deferred_sources": EXPECTED_DEFERRED_SOURCES,
    }

    with pytest.raises(SystemExit, match="not active for this stage"):
        clean_public_data.validate_source_exports(policy, source_dirs)
