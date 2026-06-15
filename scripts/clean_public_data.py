import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = [
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
    "cell_viability_percent",
]

SOURCE_DIRS = {
    "eNanoMapper": Path("data/raw/source_exports/enanomapper"),
    "caNanoLab": Path("data/raw/source_exports/cananolab"),
    "Curated nanotoxicity ML datasets": Path("data/raw/source_exports/curated_ml"),
}
SOURCE_POLICY_PATH = Path("data/raw/data_sources.json")

COLUMN_ALIASES = {
    "particle_size_nm": ["particle_size_nm", "size_nm", "diameter_nm", "primary_size_nm"],
    "zeta_potential_mv": ["zeta_potential_mv", "zeta_mv", "surface_charge_mv"],
    "dose_ug_ml": ["dose_ug_ml", "concentration_ug_ml", "exposure_concentration_ug_ml"],
    "exposure_time_h": ["exposure_time_h", "time_h", "duration_h"],
    "material_type": ["material_type", "composition", "core_material", "nanomaterial_type"],
    "surface_modification": ["surface_modification", "coating", "surface_coating", "functionalization"],
    "cell_type": ["cell_type", "cell_line", "cell"],
    "species": ["species", "organism", "cell_species"],
    "tissue": ["tissue", "organ", "cell_tissue"],
    "assay_method": ["assay_method", "assay", "endpoint_assay"],
    "cell_viability_percent": ["cell_viability_percent", "viability_percent", "cell_viability", "percent_viability"],
}

NUMERIC_COLUMNS = [
    "particle_size_nm",
    "zeta_potential_mv",
    "dose_ug_ml",
    "exposure_time_h",
    "cell_viability_percent",
]
TEXT_COLUMNS = [
    "material_type",
    "surface_modification",
    "cell_type",
    "species",
    "tissue",
    "assay_method",
]

DOSE_MEASUREMENTS = [
    {"value_column": "Concentration_d", "unit_column": "Concentration_UNIT_s"},
    {"value_column": "Dose_d", "unit_column": "Dose_UNIT_s"},
    {"value_column": "Doses/concentrations_d", "unit_column": "Doses/concentrations_UNIT_s"},
    {"value_column": "_CONDITION_Dose_d", "unit_column": "_CONDITION_Dose_UNIT_s"},
    {"value_column": "_CONDITION_concentration_d", "unit_column": "_CONDITION_concentration_UNIT_s"},
    {"value_column": "Concentration in culture medium_d", "unit_column": "Concentration in culture medium_UNIT_s"},
    {"value_column": "concentration_d", "unit_column": "concentration_UNIT_s"},
]
DOSE_UNIT_FACTORS_TO_UG_ML = {
    "ug/ml": 1.0,
    "mg/l": 1.0,
}
TIME_MEASUREMENTS = [
    {"value_column": "Time point_d", "unit_column": "Time point_UNIT_s"},
    {"value_column": "Time_d", "unit_column": "Time_UNIT_s"},
    {"value_column": "E.exposure_time_d", "unit_column": "E.exposure_time_UNIT_s"},
    {"value_column": "E.EXPOSURE_TIME_d", "unit_column": "E.EXPOSURE_TIME_UNIT_s"},
    {"value_column": "_CONDITION_exposure_time_d", "unit_column": "_CONDITION_exposure_time_UNIT_s"},
    {"value_column": "Incubation Time_d", "unit_column": None},
    {"value_column": "Incubation time_d", "unit_column": None},
]
TIME_UNIT_FACTORS_TO_HOURS = {
    "h": 1.0,
    "hr": 1.0,
    "hour": 1.0,
    "hours": 1.0,
}

SIZE_ENDPOINT_BUCKETS = [
    {
        "bucket": "primary_core_particle_size",
        "endpoints": [
            "PRIMARY_SIZE",
            "PRIMARY PARTICLE SIZE",
            "PRIMARY SIZE 1ST DIMENSION",
            "PRIMARY SIZE 2ND DIMENSION",
            "CORE SIZE",
            "PARTICLE SIZE",
            "DIAMETER",
        ],
    },
    {
        "bucket": "hydrodynamic_size",
        "endpoints": [
            "HYDRODYNAMIC_SIZE",
            "Z-AVERAGE HYDRODYNAMIC DIAMETER",
            "NUMBER MEAN HYDRODYNAMIC DIAMETER",
            "INTENSITY MEAN HYDRODYNAMIC DIAMETER",
            "VOLUME MEAN HYDRODYNAMIC DIAMETER",
        ],
    },
    {
        "bucket": "in_situ_size",
        "endpoints": ["SIZE IN SITU"],
    },
]
SIZE_ENDPOINT_TO_BUCKET = {
    endpoint: bucket["bucket"]
    for bucket in SIZE_ENDPOINT_BUCKETS
    for endpoint in bucket["endpoints"]
}
SIZE_BUCKET_PRIORITY = {
    bucket["bucket"]: priority
    for priority, bucket in enumerate(SIZE_ENDPOINT_BUCKETS)
}


def main(policy_path: Path = SOURCE_POLICY_PATH) -> None:
    policy = load_source_policy(policy_path)
    export_paths = validate_source_exports(policy, SOURCE_DIRS)
    active_training_sources = set(policy["active_training_sources"])
    frames: list[pd.DataFrame] = []
    source_counts: dict[str, int] = {}
    audit: dict[str, Any] = {
        "raw_file_row_counts": {},
        "intermediate_counts": {},
        "drop_filter_counts": {},
    }
    for source_name, directory in SOURCE_DIRS.items():
        csv_paths = export_paths[source_name]
        if source_name not in active_training_sources:
            source_counts[source_name] = 0
            continue
        if source_name == "eNanoMapper":
            normalized, source_audit = _normalize_enanomapper_exports(directory)
            _merge_audit(audit, source_audit)
        else:
            source_frames = [_normalize_columns(_read_source_file(path)) for path in csv_paths]
            normalized = pd.concat(source_frames, ignore_index=True) if source_frames else pd.DataFrame(columns=REQUIRED_COLUMNS)
            audit["raw_file_row_counts"][source_name] = {path.name: len(_read_source_file(path)) for path in csv_paths}
        source_counts[source_name] = len(normalized)
        if not normalized.empty:
            frames.append(normalized)
    if not frames:
        raise SystemExit("No public source CSV files found under data/raw/source_exports")

    combined = pd.concat(frames, ignore_index=True)
    audit["intermediate_counts"]["combined_normalized_rows_before_filter"] = len(combined)
    clean, filter_counts = _coerce_and_filter(combined)
    audit["drop_filter_counts"].update(filter_counts)
    rows_before_dedup = len(clean)
    clean = clean.drop_duplicates().reset_index(drop=True)
    audit["drop_filter_counts"]["duplicates_removed"] = rows_before_dedup - len(clean)
    audit["intermediate_counts"]["final_rows_before_dedup"] = rows_before_dedup
    audit["intermediate_counts"]["final_rows_after_dedup"] = len(clean)
    if len(clean) < 100:
        raise SystemExit(f"Cleaned public dataset has {len(clean)} rows; expected at least 100")
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    clean.to_csv("data/processed/toxicity_clean.csv", index=False)
    _write_metadata(clean, source_counts, audit, policy)


def load_source_policy(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        policy = json.load(handle)
    priority_names = [entry["name"] for entry in policy.get("priority_order", [])]
    active = policy.get("active_training_sources")
    deferred = policy.get("deferred_sources")
    if not isinstance(active, list) or not active:
        raise SystemExit("data_sources.json must define non-empty active_training_sources")
    if not isinstance(deferred, list):
        raise SystemExit("data_sources.json must define deferred_sources")
    unknown = (set(active) | set(deferred)) - set(priority_names)
    if unknown:
        raise SystemExit(f"data_sources.json references unknown sources: {sorted(unknown)}")
    return policy


def validate_source_exports(policy: dict[str, Any], source_dirs: dict[str, Path]) -> dict[str, list[Path]]:
    active_training_sources = set(policy["active_training_sources"])
    export_paths: dict[str, list[Path]] = {}
    for source_name, directory in source_dirs.items():
        csv_paths = sorted(directory.glob("*.csv"))
        export_paths[source_name] = csv_paths
        if source_name in active_training_sources and not csv_paths:
            raise SystemExit(f"Active training source {source_name} has no CSV exports in {directory}")
        if source_name not in active_training_sources and csv_paths:
            raise SystemExit(
                f"Source {source_name} has CSV exports but is not active for this stage; "
                "update data/raw/data_sources.json active_training_sources before merging it into training data"
            )
    return export_paths


def _merge_audit(target: dict[str, Any], source: dict[str, Any]) -> None:
    for section, values in source.items():
        target.setdefault(section, {}).update(values)


def _read_source_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    lowered = {column.lower().strip(): column for column in frame.columns}
    output = pd.DataFrame()
    for target, aliases in COLUMN_ALIASES.items():
        source_column = next((lowered[alias] for alias in aliases if alias in lowered), None)
        output[target] = frame[source_column] if source_column else pd.NA
    return output[REQUIRED_COLUMNS]


def _normalize_enanomapper_exports(directory: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    viability = _read_matching_export(directory, "*viability*.csv")
    conditions = _read_matching_export(directory, "*conditions*.csv")
    pchem = _read_matching_export(directory, "*pchem*.csv")
    params = _read_matching_export(directory, "*params*.csv")
    conditions, condition_audit = _prepare_conditions(conditions)
    pchem_lookup, pchem_audit = _build_pchem_lookup(pchem)
    merged = viability.merge(
        conditions[["effectid_hs", "dose_ug_ml", "exposure_time_h"]],
        left_on="id",
        right_on="effectid_hs",
        how="inner",
    )
    material_keys = merged["name_s"].map(_material_key)
    output = pd.DataFrame(
        {
            "particle_size_nm": merged["s_uuid_s"].map(pchem_lookup["size_by_uuid"]).fillna(
                material_keys.map(pchem_lookup["size_by_material_key"])
            ),
            "zeta_potential_mv": merged["s_uuid_s"].map(pchem_lookup["zeta_by_uuid"]).fillna(
                material_keys.map(pchem_lookup["zeta_by_material_key"])
            ),
            "dose_ug_ml": merged["dose_ug_ml"],
            "exposure_time_h": merged["exposure_time_h"],
            "material_type": merged.apply(_material_type, axis=1),
            "surface_modification": merged["name_s"].map(_surface_modification),
            "cell_type": merged["guidance_s"].map(_cell_type),
            "species": merged["guidance_s"].map(_species),
            "tissue": merged["guidance_s"].map(_tissue),
            "assay_method": merged["endpoint_s"].map(_assay_method),
            "cell_viability_percent": merged["loValue_d"],
        }
    )
    audit = {
        "raw_file_row_counts": {
            "eNanoMapper": {
                "viability": len(viability),
                "conditions": len(conditions),
                "pchem": len(pchem),
                "params": len(params),
            }
        },
        "intermediate_counts": {
            "enanomapper_condition_rows_with_supported_dose": int(conditions["dose_ug_ml"].notna().sum()),
            "enanomapper_condition_rows_with_supported_exposure_time": int(conditions["exposure_time_h"].notna().sum()),
            "enanomapper_viability_condition_join_rows": len(merged),
            "enanomapper_normalized_rows": len(output),
            **pchem_audit["intermediate_counts"],
        },
        "drop_filter_counts": {
            **condition_audit["drop_filter_counts"],
            **pchem_audit["drop_filter_counts"],
        },
    }
    return output[REQUIRED_COLUMNS], audit


def _read_matching_export(directory: Path, pattern: str) -> pd.DataFrame:
    paths = sorted(directory.glob(pattern))
    if len(paths) != 1:
        raise SystemExit(f"Expected exactly one eNanoMapper export matching {pattern}, found {len(paths)}")
    return pd.read_csv(paths[0], low_memory=False)


def _prepare_conditions(conditions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    prepared = conditions.copy()
    prepared["dose_ug_ml"], dose_audit = _first_supported_measurement(
        prepared,
        DOSE_MEASUREMENTS,
        DOSE_UNIT_FACTORS_TO_UG_ML,
    )
    prepared["exposure_time_h"], time_audit = _first_supported_measurement(
        prepared,
        TIME_MEASUREMENTS,
        TIME_UNIT_FACTORS_TO_HOURS,
    )
    audit = {
        "drop_filter_counts": {
            "condition_dose_values_rejected_by_missing_or_unsupported_unit": dose_audit["rejected_value_rows"],
            "condition_exposure_time_values_rejected_by_missing_or_unsupported_unit": time_audit["rejected_value_rows"],
        }
    }
    return prepared, audit


def _first_supported_measurement(
    frame: pd.DataFrame,
    measurements: list[dict[str, str | None]],
    unit_factors: dict[str, float],
) -> tuple[pd.Series, dict[str, Any]]:
    converted_columns: list[pd.Series] = []
    rejected_value_rows = 0
    for measurement in measurements:
        value_column = measurement["value_column"]
        unit_column = measurement["unit_column"]
        if value_column not in frame.columns:
            continue
        values = pd.to_numeric(frame[value_column], errors="coerce")
        has_value = values.notna()
        if unit_column and unit_column in frame.columns:
            units = frame[unit_column].map(_normalize_unit)
            factors = units.map(unit_factors)
        else:
            factors = pd.Series(pd.NA, index=frame.index, dtype="Float64")
        supported = has_value & factors.notna()
        rejected_value_rows += int((has_value & ~supported).sum())
        converted_columns.append((values * pd.to_numeric(factors, errors="coerce")).where(supported))
    if not converted_columns:
        return pd.Series(pd.NA, index=frame.index), {"rejected_value_rows": 0}
    converted = pd.concat(converted_columns, axis=1)
    return converted.bfill(axis=1).iloc[:, 0], {"rejected_value_rows": rejected_value_rows}


def _normalize_unit(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("µ", "u").replace("μ", "u")
    text = re.sub(r"\s+", "", text)
    return text


def _build_pchem_lookup(pchem: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[str, Any]]:
    prepared = pchem.copy()
    prepared["effect_key"] = prepared["effectendpoint_s"].astype("string").str.upper()
    prepared["unit_key"] = prepared["unit_s"].astype("string").str.lower()
    prepared["loValue_d"] = pd.to_numeric(prepared["loValue_d"], errors="coerce")
    prepared["material_key"] = prepared["publicname_s"].fillna(prepared["name_s"]).map(_material_key)
    prepared["size_bucket"] = prepared["effect_key"].map(SIZE_ENDPOINT_TO_BUCKET)
    has_size_endpoint = prepared["size_bucket"].notna()
    has_zeta_endpoint = prepared["effect_key"].str.contains("ZETA", regex=True, na=False).fillna(False)
    has_nm_unit = prepared["unit_key"].eq("nm").fillna(False)
    has_mv_unit = prepared["unit_key"].eq("mv").fillna(False)
    size_rows = prepared[has_size_endpoint & has_nm_unit & prepared["loValue_d"].gt(0)]
    zeta_rows = prepared[has_zeta_endpoint & has_mv_unit & prepared["loValue_d"].notna()]
    lookup = {
        "size_by_uuid": _preferred_size_median(size_rows, "s_uuid_s"),
        "zeta_by_uuid": zeta_rows.groupby("s_uuid_s")["loValue_d"].median(),
        "size_by_material_key": _preferred_size_median(size_rows, "material_key"),
        "zeta_by_material_key": zeta_rows.groupby("material_key")["loValue_d"].median(),
    }
    audit = {
        "intermediate_counts": {
            "enanomapper_pchem_size_candidate_rows": int(has_size_endpoint.sum()),
            "enanomapper_pchem_size_whitelisted_nm_rows": len(size_rows),
            "enanomapper_pchem_zeta_mv_rows": len(zeta_rows),
            "enanomapper_pchem_size_uuid_lookup_keys": len(lookup["size_by_uuid"]),
            "enanomapper_pchem_size_material_lookup_keys": len(lookup["size_by_material_key"]),
            "enanomapper_pchem_zeta_uuid_lookup_keys": len(lookup["zeta_by_uuid"]),
            "enanomapper_pchem_zeta_material_lookup_keys": len(lookup["zeta_by_material_key"]),
        },
        "drop_filter_counts": {
            "pchem_size_rows_rejected_by_endpoint_or_unit": int((prepared["effect_key"].str.contains("SIZE|DIAMETER", regex=True, na=False).fillna(False) & ~has_size_endpoint).sum())
            + int((has_size_endpoint & ~has_nm_unit).sum()),
            "pchem_zeta_rows_rejected_by_unit": int((has_zeta_endpoint & ~has_mv_unit).sum()),
        },
    }
    return lookup, audit


def _preferred_size_median(rows: pd.DataFrame, key_column: str) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype="float64")
    grouped = rows.groupby([key_column, "size_bucket"], dropna=True)["loValue_d"].median().reset_index()
    grouped["bucket_priority"] = grouped["size_bucket"].map(SIZE_BUCKET_PRIORITY)
    selected = grouped.sort_values([key_column, "bucket_priority"]).drop_duplicates(key_column)
    return selected.set_index(key_column)["loValue_d"]


def _material_key(value: object) -> str:
    text = str(value).lower()
    nm_match = re.search(r"nm[-\s]?\d+", text)
    formula = next((formula for formula in ["tio2", "zno", "sio2"] if formula in text), None)
    if formula and nm_match:
        return f"{formula} {nm_match.group(0).replace(' ', '-')}"
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _material_type(row: pd.Series) -> str:
    name = str(row.get("name_s", ""))
    public_name = str(row.get("publicname_s", ""))
    match = re.search(r"\(([^)]+)\)", name)
    if match:
        return match.group(1)
    return public_name or name


def _surface_modification(value: object) -> str:
    text = str(value).lower()
    if "corona" in text:
        return "protein_corona"
    if "peg" in text:
        return "PEG"
    if "cooh" in text:
        return "COOH"
    return "unreported"


def _cell_type(value: object) -> str:
    text = str(value)
    match = re.search(r"\(([^)]+)\)", text)
    if match:
        return match.group(1)
    if "macrophage" in text.lower():
        return "HMDM"
    return "unreported_cell"


def _species(value: object) -> str:
    text = str(value).lower()
    if "human" in text or "hmdm" in text:
        return "Human"
    if "rat" in text:
        return "Rat"
    if "mouse" in text:
        return "Mouse"
    return "unreported_species"


def _tissue(value: object) -> str:
    text = str(value).lower()
    if "macrophage" in text or "monocyte" in text or "hmdm" in text:
        return "blood_immune"
    if "lung" in text:
        return "lung"
    return "unreported_tissue"


def _assay_method(value: object) -> str:
    text = str(value)
    upper = text.upper()
    if "LDH" in upper:
        return "LDH"
    if "MTT" in upper:
        return "MTT"
    if "WST" in upper:
        return "WST"
    return re.sub(r"\s+", " ", text).strip() or "cell_viability_assay"


def _coerce_and_filter(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    clean = frame.copy()
    for column in NUMERIC_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    for column in TEXT_COLUMNS:
        clean[column] = clean[column].astype("string").str.strip().str.replace(r"\s+", "_", regex=True)
    rows_before_required = len(clean)
    clean = clean.dropna(subset=REQUIRED_COLUMNS)
    rows_removed_by_required_values = rows_before_required - len(clean)
    rows_before_ranges = len(clean)
    clean = clean[
        (clean["particle_size_nm"] > 0)
        & (clean["dose_ug_ml"] >= 0)
        & (clean["exposure_time_h"] > 0)
        & (clean["cell_viability_percent"].between(0, 100))
    ]
    rows_removed_by_ranges = rows_before_ranges - len(clean)
    return clean[REQUIRED_COLUMNS], {
        "final_rows_removed_by_required_values": rows_removed_by_required_values,
        "final_rows_removed_by_numeric_ranges": rows_removed_by_ranges,
        "final_rows_removed_by_required_values_and_ranges": rows_removed_by_required_values + rows_removed_by_ranges,
    }


def _write_metadata(
    frame: pd.DataFrame,
    source_counts: dict[str, int],
    audit: dict[str, Any],
    policy: dict[str, Any],
) -> None:
    training_sources = policy["active_training_sources"]
    deferred_sources = policy["deferred_sources"]
    metadata = {
        "dataset_name": "toxicity_clean.csv",
        "dataset_type": "cleaned_public_nanotoxicity_dataset",
        "purpose": "第一版模型训练、评估和 UI 预测使用的 eNanoMapper 公开数据清洗结果。",
        "source_policy_file": "data/raw/data_sources.json",
        "training_sources": training_sources,
        "deferred_sources": deferred_sources,
        "source_record_counts": source_counts,
        "record_count": len(frame),
        "cleaned_output_path": "data/processed/toxicity_clean.csv",
        "deduplication_key": REQUIRED_COLUMNS,
        "unit_normalization": {
            "particle_size_nm": "nm; eNanoMapper loValue_d with unit_s == nm",
            "zeta_potential_mv": "mV; eNanoMapper loValue_d with unit_s == mV",
            "dose_ug_ml": {
                "target_unit": "ug/mL",
                "allowed_units": ["ug/ml", "µg/ml", "μg/ml", "mg/l"],
                "conversion": "ug/ml, µg/ml, μg/ml and mg/L are normalized to ug/mL with factor 1.0; area dose, molar, mass-only, and unknown units are excluded.",
                "source_columns": DOSE_MEASUREMENTS,
            },
            "exposure_time_h": {
                "target_unit": "h",
                "allowed_units": ["h", "hr", "hour", "hours"],
                "conversion": "Only explicit hour units are accepted with factor 1.0; missing, minute-only, and unknown units are excluded.",
                "source_columns": TIME_MEASUREMENTS,
            },
            "cell_viability_percent": "%; eNanoMapper % CELL VIABILITY records",
            "pchem_alignment": "优先按 s_uuid 精确匹配；同一公开材料编码存在物化记录时，按材料编码中位数补齐粒径和 zeta。",
        },
        "pchem_size_endpoint_policy": {
            "endpoint_priority": SIZE_ENDPOINT_BUCKETS,
            "unit_filter": "Only nm pchem rows are eligible for particle_size_nm.",
            "aggregation": "Median within endpoint bucket and s_uuid first; if no s_uuid match exists, median within endpoint bucket and normalized public material key is used.",
            "excluded_endpoint_examples": [
                "PARTICLE SIZE DISTRIBUTION:*",
                "INNER_DIAMETER",
                "non-nm PARTICLE SIZE/DIAMETER rows",
            ],
        },
        "cleaning_audit": audit,
        "unused_raw_exports": {
            "eNanoMapper": {
                "files": ["enanomapper_params_2026-06-15.csv"],
                "reason": "保留原始 params 导出以便审计条件字段来源；当前规范化字段来自 viability、conditions 和 pchem 导出，params 不参与训练表合并。",
            }
        },
        "excluded_record_policy": "删除缺失目标值、缺失剂量或暴露时间、无法从公开物化记录匹配粒径或 zeta、数值范围无效或重复的记录；保留无法从 guidance 明确映射的细胞/物种/组织上下文，并以 unreported_cell、unreported_species、unreported_tissue 标记。",
        "scientific_use_limit": "该清洗数据来自 eNanoMapper 公开 Solr 导出并可用于第一版原型训练；正式科学结论仍需复核原始实验条件、剂量单位、终点定义和文献上下文。",
        "created_for": "纳米毒性预测智能体第一版可运行原型",
        "required_report_disclosure": "第一版原型当前只使用 eNanoMapper 公开数据清洗结果训练模型；报告中必须说明数据来源、清洗规则、记录数量、物化字段匹配策略、caNanoLab/Curated ML 延后接入状态和公开数据局限。",
    }
    Path("data/processed/toxicity_clean_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
