import json
import re
from pathlib import Path

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
ACTIVE_TRAINING_SOURCES = {"eNanoMapper"}
DEFERRED_SOURCES = ["caNanoLab", "Curated nanotoxicity ML datasets"]

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


def main() -> None:
    frames: list[pd.DataFrame] = []
    source_counts: dict[str, int] = {}
    for source_name, directory in SOURCE_DIRS.items():
        csv_paths = sorted(directory.glob("*.csv"))
        if not csv_paths:
            if source_name in ACTIVE_TRAINING_SOURCES:
                raise SystemExit(f"Active training source {source_name} has no CSV exports in {directory}")
            source_counts[source_name] = 0
            continue
        if source_name not in ACTIVE_TRAINING_SOURCES:
            raise SystemExit(
                f"Source {source_name} has CSV exports but is not active for this stage; "
                "add it to ACTIVE_TRAINING_SOURCES before merging it into training data"
            )
        if source_name == "eNanoMapper":
            normalized = _normalize_enanomapper_exports(directory)
        else:
            source_frames = [_normalize_columns(_read_source_file(path)) for path in csv_paths]
            normalized = pd.concat(source_frames, ignore_index=True) if source_frames else pd.DataFrame(columns=REQUIRED_COLUMNS)
        source_counts[source_name] = len(normalized)
        if not normalized.empty:
            frames.append(normalized)
    if not frames:
        raise SystemExit("No public source CSV files found under data/raw/source_exports")

    clean = pd.concat(frames, ignore_index=True)
    clean = _coerce_and_filter(clean)
    clean = clean.drop_duplicates().reset_index(drop=True)
    if len(clean) < 100:
        raise SystemExit(f"Cleaned public dataset has {len(clean)} rows; expected at least 100")
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    clean.to_csv("data/processed/toxicity_clean.csv", index=False)
    _write_metadata(clean, source_counts)


def _read_source_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    lowered = {column.lower().strip(): column for column in frame.columns}
    output = pd.DataFrame()
    for target, aliases in COLUMN_ALIASES.items():
        source_column = next((lowered[alias] for alias in aliases if alias in lowered), None)
        output[target] = frame[source_column] if source_column else pd.NA
    return output[REQUIRED_COLUMNS]


def _normalize_enanomapper_exports(directory: Path) -> pd.DataFrame:
    viability = _read_matching_export(directory, "*viability*.csv")
    conditions = _read_matching_export(directory, "*conditions*.csv")
    pchem = _read_matching_export(directory, "*pchem*.csv")
    conditions = _prepare_conditions(conditions)
    pchem_lookup = _build_pchem_lookup(pchem)
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
    return output[REQUIRED_COLUMNS]


def _read_matching_export(directory: Path, pattern: str) -> pd.DataFrame:
    paths = sorted(directory.glob(pattern))
    if len(paths) != 1:
        raise SystemExit(f"Expected exactly one eNanoMapper export matching {pattern}, found {len(paths)}")
    return pd.read_csv(paths[0], low_memory=False)


def _prepare_conditions(conditions: pd.DataFrame) -> pd.DataFrame:
    dose_columns = [
        "Concentration_d",
        "Dose_d",
        "Doses/concentrations_d",
        "_CONDITION_Dose_d",
        "_CONDITION_concentration_d",
        "Concentration in culture medium_d",
        "concentration_d",
    ]
    time_columns = [
        "Time point_d",
        "Time_d",
        "E.exposure_time_d",
        "E.EXPOSURE_TIME_d",
        "_CONDITION_exposure_time_d",
        "Incubation Time_d",
        "Incubation time_d",
    ]
    prepared = conditions.copy()
    prepared["dose_ug_ml"] = _first_numeric(prepared, dose_columns)
    prepared["exposure_time_h"] = _first_numeric(prepared, time_columns)
    return prepared


def _first_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    existing = [column for column in columns if column in frame.columns]
    if not existing:
        return pd.Series(pd.NA, index=frame.index)
    numeric = frame[existing].apply(pd.to_numeric, errors="coerce")
    return numeric.bfill(axis=1).iloc[:, 0]


def _build_pchem_lookup(pchem: pd.DataFrame) -> dict[str, pd.Series]:
    prepared = pchem.copy()
    prepared["effect_key"] = prepared["effectendpoint_s"].astype("string").str.upper()
    prepared["unit_key"] = prepared["unit_s"].astype("string").str.lower()
    prepared["loValue_d"] = pd.to_numeric(prepared["loValue_d"], errors="coerce")
    prepared["material_key"] = prepared["publicname_s"].fillna(prepared["name_s"]).map(_material_key)
    has_size_endpoint = prepared["effect_key"].str.contains("SIZE|DIAMETER", regex=True, na=False).fillna(False)
    has_zeta_endpoint = prepared["effect_key"].str.contains("ZETA", regex=True, na=False).fillna(False)
    has_nm_unit = prepared["unit_key"].eq("nm").fillna(False)
    has_mv_unit = prepared["unit_key"].eq("mv").fillna(False)
    size_rows = prepared[has_size_endpoint & has_nm_unit & prepared["loValue_d"].gt(0)]
    zeta_rows = prepared[has_zeta_endpoint & has_mv_unit & prepared["loValue_d"].notna()]
    return {
        "size_by_uuid": size_rows.groupby("s_uuid_s")["loValue_d"].median(),
        "zeta_by_uuid": zeta_rows.groupby("s_uuid_s")["loValue_d"].median(),
        "size_by_material_key": size_rows.groupby("material_key")["loValue_d"].median(),
        "zeta_by_material_key": zeta_rows.groupby("material_key")["loValue_d"].median(),
    }


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


def _coerce_and_filter(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    for column in NUMERIC_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    for column in TEXT_COLUMNS:
        clean[column] = clean[column].astype("string").str.strip().str.replace(r"\s+", "_", regex=True)
    clean = clean.dropna(subset=REQUIRED_COLUMNS)
    clean = clean[
        (clean["particle_size_nm"] > 0)
        & (clean["dose_ug_ml"] >= 0)
        & (clean["exposure_time_h"] > 0)
        & (clean["cell_viability_percent"].between(0, 100))
    ]
    return clean[REQUIRED_COLUMNS]


def _write_metadata(frame: pd.DataFrame, source_counts: dict[str, int]) -> None:
    metadata = {
        "dataset_name": "toxicity_clean.csv",
        "dataset_type": "cleaned_public_nanotoxicity_dataset",
        "purpose": "第一版模型训练、评估和 UI 预测使用的 eNanoMapper 公开数据清洗结果。",
        "source_policy_file": "data/raw/data_sources.json",
        "training_sources": sorted(ACTIVE_TRAINING_SOURCES),
        "deferred_sources": DEFERRED_SOURCES,
        "source_record_counts": source_counts,
        "record_count": len(frame),
        "cleaned_output_path": "data/processed/toxicity_clean.csv",
        "deduplication_key": REQUIRED_COLUMNS,
        "unit_normalization": {
            "particle_size_nm": "nm; eNanoMapper loValue_d with unit_s == nm",
            "zeta_potential_mv": "mV; eNanoMapper loValue_d with unit_s == mV",
            "dose_ug_ml": "ug/mL; eNanoMapper Concentration/Dose numeric condition fields",
            "exposure_time_h": "h; eNanoMapper Time/exposure numeric condition fields",
            "cell_viability_percent": "%; eNanoMapper % CELL VIABILITY records",
            "pchem_alignment": "优先按 s_uuid 精确匹配；同一公开材料编码存在物化记录时，按材料编码中位数补齐粒径和 zeta。",
        },
        "excluded_record_policy": "删除缺失目标值、缺失剂量或暴露时间、无法从公开物化记录匹配粒径或 zeta、无法映射材料组成或细胞类型的记录；删除 cell viability 超出 0-100 的记录。",
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
