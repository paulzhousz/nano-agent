# 纳米毒性预测智能体基础与数据实现计划

> **给智能体执行者：** 必须使用子技能：`superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，按任务逐步执行本计划。步骤使用复选框（`- [ ]`）语法跟踪。

**目标：** 创建项目骨架、领域 schema、完整公开数据清洗结果、核心文献完整摘要库和测试。

**架构：** 本子计划建立后续所有子计划都会使用的稳定契约。它不包含模型训练，也不包含 UI 代码。

**技术栈：** Python 3.12、uv、pytest、JSON、CSV。

---

## 文件

- 创建：`pyproject.toml`
- 创建：`.python-version`
- 创建：`.gitignore`
- 创建：`src/nano_tox_agent/__init__.py`
- 创建：`src/nano_tox_agent/schema.py`
- 创建：`src/nano_tox_agent/literature.py`
- 创建：`scripts/clean_public_data.py`
- 创建：`scripts/build_literature_base.py`
- 创建：`tests/test_schema.py`
- 创建：`tests/test_public_data.py`
- 创建：`tests/test_literature.py`
- 创建：`data/raw/data_sources.json`
- 创建：`data/processed/toxicity_clean.csv`
- 创建：`data/processed/toxicity_clean_metadata.json`
- 创建：`literature/literature_base.json`

## 任务 1：包骨架

- [ ] **步骤 1：编写失败的导入测试**

创建 `tests/test_schema.py`：

```python
from nano_tox_agent import __version__


def test_package_imports():
    assert __version__ == "0.1.0"
```

- [ ] **步骤 2：验证失败**

运行：`rtk uv run pytest tests/test_schema.py -q`

预期：失败，报 `ModuleNotFoundError: No module named 'nano_tox_agent'`；如果 uv 或 Python 3.12 不可用，则先报环境创建错误。

- [ ] **步骤 3：创建包 metadata**

创建 `pyproject.toml`：

```toml
[project]
name = "nano-tox-agent"
version = "0.1.0"
description = "Nanomaterial cytotoxicity prediction agent prototype"
requires-python = ">=3.12,<3.13"
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

创建 `.python-version`：

```text
3.12
```

创建 `.gitignore`：

```gitignore
.DS_Store
.env
.venv/
.pytest_cache/
.ruff_cache/
.streamlit/secrets.toml
__pycache__/
*.py[cod]
models/*.joblib
models/*.json
data/cache/
.superpowers/
```

创建 `src/nano_tox_agent/__init__.py`：

```python
__version__ = "0.1.0"
```

- [ ] **步骤 4：验证包导入**

运行：

```bash
rtk uv venv --python 3.12
rtk uv sync --extra dev
rtk uv run pytest tests/test_schema.py -q
```

预期：uv 创建 `.venv`，安装项目和 dev 依赖；测试通过，显示 `1 passed`。

## 任务 2：领域 Schema

- [ ] **步骤 1：替换 schema 测试**

替换 `tests/test_schema.py`：

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

- [ ] **步骤 2：验证失败**

运行：`rtk uv run pytest tests/test_schema.py -q`

预期：失败，报 `ModuleNotFoundError: No module named 'nano_tox_agent.schema'`。

- [ ] **步骤 3：创建 schema 模块**

创建 `src/nano_tox_agent/schema.py`：

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

- [ ] **步骤 4：验证 schema**

运行：`rtk uv run pytest tests/test_schema.py -q`

预期：通过，显示 `5 passed`。

## 任务 3：完整公开数据清洗流水线与核心文献完整摘要

- [ ] **步骤 1：创建公开数据来源清单**

创建 `data/raw/data_sources.json`：

```json
{
  "source_policy": "第一版必须接入完整公开数据清洗结果；不得用合成数据或小型演示样例替代训练数据。",
  "cleaned_output_path": "data/processed/toxicity_clean.csv",
  "priority_order": [
    {
      "name": "eNanoMapper",
      "priority": 1,
      "purpose": "通用纳米材料安全与毒性数据，适合训练理化性质到细胞毒性的预测模型。",
      "url": "https://search.data.enanomapper.net/",
      "expected_use": ["training_data", "schema_alignment", "nanotoxicology_safety_context"]
    },
    {
      "name": "caNanoLab",
      "priority": 2,
      "purpose": "癌症纳米技术相关数据平台，适合支撑肿瘤纳米药物筛选应用背景。",
      "url": "https://cananolab.cancer.gov/",
      "expected_use": ["cancer_nanomedicine_context", "source_cross_check"]
    },
    {
      "name": "Curated nanotoxicity ML datasets",
      "priority": 3,
      "purpose": "当原始数据库清洗难度较高时，使用已整理的公开纳米毒性机器学习数据集作为第一版原型训练数据。",
      "url": "https://doi.org/10.1021/acsanm.4c02269",
      "expected_use": ["prototype_training", "feature_selection", "model_validation"]
    }
  ],
  "required_normalized_columns": [
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
    "cell_viability_percent"
  ]
}
```

- [ ] **步骤 2：提交完整公开数据清洗结果**

将公开来源原始导出文件放入以下目录。文件名可以按实际导出日期命名，但必须保留原始导出，不要直接手改：

```text
data/raw/source_exports/enanomapper/
data/raw/source_exports/cananolab/
data/raw/source_exports/curated_ml/
```

创建 `scripts/clean_public_data.py`：

```python
import json
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


def main() -> None:
    frames: list[pd.DataFrame] = []
    source_counts: dict[str, int] = {}
    for source_name, directory in SOURCE_DIRS.items():
        source_frames = [_read_source_file(path) for path in sorted(directory.glob("*.csv"))]
        if not source_frames:
            source_counts[source_name] = 0
            continue
        combined = pd.concat(source_frames, ignore_index=True)
        normalized = _normalize_columns(combined)
        source_counts[source_name] = len(normalized)
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
    return pd.read_csv(path)


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    lowered = {column.lower().strip(): column for column in frame.columns}
    output = pd.DataFrame()
    for target, aliases in COLUMN_ALIASES.items():
        source_column = next((lowered[alias] for alias in aliases if alias in lowered), None)
        output[target] = frame[source_column] if source_column else pd.NA
    return output[REQUIRED_COLUMNS]


def _coerce_and_filter(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    for column in ["particle_size_nm", "zeta_potential_mv", "dose_ug_ml", "exposure_time_h", "cell_viability_percent"]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    for column in ["material_type", "surface_modification", "cell_type", "species", "tissue", "assay_method"]:
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
        "purpose": "第一版模型训练、评估和 UI 预测使用的完整公开数据清洗结果。",
        "source_policy_file": "data/raw/data_sources.json",
        "source_record_counts": source_counts,
        "record_count": len(frame),
        "deduplication_key": REQUIRED_COLUMNS,
        "unit_normalization": {
            "particle_size_nm": "nm",
            "zeta_potential_mv": "mV",
            "dose_ug_ml": "ug/mL",
            "exposure_time_h": "h",
            "cell_viability_percent": "%",
        },
        "excluded_record_policy": "删除缺失目标值、无法统一剂量单位、无法映射材料组成或缺少细胞类型的记录；保留清洗脚本或人工清洗记录中的排除原因。",
        "scientific_use_limit": "该清洗数据来自公开来源并可用于第一版原型训练；正式科学结论仍需复核原始实验条件、剂量单位、终点定义和文献上下文。",
        "created_for": "纳米毒性预测智能体第一版可运行原型",
        "required_report_disclosure": "第一版原型使用完整公开数据清洗结果训练模型；报告中必须说明数据来源、清洗规则、记录数量和公开数据局限。",
    }
    Path("data/processed/toxicity_clean_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
```

运行：

```bash
rtk uv run python scripts/clean_public_data.py
```

预期：生成 `data/processed/toxicity_clean.csv` 和 `data/processed/toxicity_clean_metadata.json`。CSV 必须使用以下表头，数据行不得是合成样例：

```csv
particle_size_nm,zeta_potential_mv,dose_ug_ml,exposure_time_h,material_type,surface_modification,cell_type,species,tissue,assay_method,cell_viability_percent
```

- [ ] **步骤 3：创建完整清洗 metadata 和核心文献 JSON**

确认 `scripts/clean_public_data.py` 已生成 `data/processed/toxicity_clean_metadata.json`，并人工检查 `source_record_counts` 是否与各公开来源清洗前进入合并流程的实际数量一致。测试会校验总数不小于最终去重后的 `record_count`。

创建 `scripts/build_literature_base.py`：

```python
import json
from pathlib import Path


def main() -> None:
    entries = [
        {
            "title": "The eNanoMapper database for nanomaterial safety information",
            "year": 2015,
            "url": "https://www.beilstein-journals.org/bjnano/articles/6/165",
            "used_for": ["data_source", "schema_design"],
            "full_summary": "该文献介绍 eNanoMapper 面向纳米材料安全信息的数据管理体系，重点是把纳米材料表征、实验条件和毒理结果组织成可查询、可复用的结构化记录。它为本项目的数据 schema 提供依据：纳米材料理化性质、生物实验条件和毒性终点必须同时保留，才能支持后续 nanoQSAR 或机器学习建模。第一版数据清洗应优先参考该类结构化安全数据，而不是只保留模型输入所需的最少字段。",
            "method_relevance": "用于定义公开数据来源优先级、字段追溯和纳米材料安全数据管理边界。",
            "data_relevance": "支持把材料组成、粒径、表面电荷、暴露条件、细胞类型和毒性终点作为统一清洗对象。",
            "limitations": "数据库记录来自多实验条件，直接合并前必须处理单位、缺失值、重复记录和终点定义差异。",
            "key_points": [
                "eNanoMapper 面向纳米材料安全信息管理。",
                "该数据库支持结构化的纳米材料表征和毒理数据。",
                "它可以支持 nanoQSAR 和机器学习工作流。",
            ],
        },
        {
            "title": "Predicting Cytotoxicity of Nanoparticles: A Meta-Analysis Using Machine Learning",
            "year": 2024,
            "url": "https://doi.org/10.1021/acsanm.4c02269",
            "used_for": ["model_design", "feature_selection", "feature_explanation"],
            "full_summary": "该文献以机器学习方式总结纳米颗粒细胞毒性预测任务，强调细胞活力等毒性终点可以由材料组成、剂量、粒径、Zeta 电位、暴露时间和细胞背景共同预测。它支撑第一版同时训练分类模型和回归模型：分类模型输出低毒、中毒、高毒，回归模型输出细胞活力百分比。它也支撑随机森林作为第一版模型，因为随机森林适合处理中小规模表格数据、非线性关系和特征重要性解释。",
            "method_relevance": "用于选择 RandomForestClassifier、RandomForestRegressor、holdout 评估和 feature importance 解释。",
            "data_relevance": "用于确认组成、剂量、粒径、Zeta 电位和暴露时间是清洗数据必须保留的核心字段。",
            "limitations": "不同文献的细胞毒性实验条件并不完全一致，模型表现不能被解释为跨所有纳米材料和细胞类型的通用毒理结论。",
            "key_points": [
                "机器学习可用于预测纳米颗粒细胞毒性。",
                "组成、浓度、Zeta 电位、粒径和暴露时间是常见预测特征。",
                "随机森林适用于细胞活力预测和特征重要性分析。",
            ],
        },
        {
            "title": "Application of Machine Learning in Nanotoxicology: A Critical Review and Perspective",
            "year": 2024,
            "url": "https://doi.org/10.1021/acs.est.4c03217",
            "used_for": ["background", "limitations"],
            "full_summary": "该综述说明机器学习在纳米毒理学中可以加速风险筛查和假设生成，但核心瓶颈来自数据规模有限、来源异质、表征字段不完整、实验协议差异和模型可解释性不足。本项目据此把第一版定位为公开数据驱动的原型系统，而不是临床或监管结论系统。解释层必须说明数据和模型局限，不能让 LLM 编造指标或替代模型结论。",
            "method_relevance": "用于约束模型解释、报告措辞和第一版适用范围。",
            "data_relevance": "用于要求 metadata 记录来源、清洗规则、单位规范化、去重策略和排除记录原则。",
            "limitations": "综述强调的异质性问题不会因清洗而消失，第一版应显式披露残余偏差。",
            "key_points": [
                "机器学习可以加速纳米毒理学分析。",
                "纳米毒理学数据集通常规模较小且异质性强。",
                "可解释性和数据质量仍是核心挑战。",
            ],
        },
        {
            "title": "caNanoLab: data sharing to expedite the use of nanotechnology in biomedicine",
            "year": 2013,
            "url": "https://cananolab.cancer.gov/",
            "used_for": ["cancer_nanomedicine_context", "data_source"],
            "full_summary": "caNanoLab 面向癌症纳米技术数据共享，提供纳米材料、表征、样品和生物医学应用相关记录。它为本项目的肿瘤纳米药物筛选背景提供依据：应用场景可以聚焦候选纳米材料在肿瘤相关细胞或正常对照细胞中的体外毒性风险，但系统不能声称预测人体临床毒性或抗肿瘤疗效。",
            "method_relevance": "用于限定 Streamlit 应用和 README 的产品边界。",
            "data_relevance": "用于补充癌症纳米技术背景来源，并与通用纳米安全数据交叉验证材料和实验字段。",
            "limitations": "caNanoLab 更偏生物医学数据共享平台，直接训练前仍需筛选具有明确体外细胞毒性终点的记录。",
            "key_points": [
                "caNanoLab 支持癌症纳米技术数据共享。",
                "该平台为生物医学和癌症纳米材料应用提供背景。",
                "它可作为肿瘤纳米医学筛选场景的背景资料。",
            ],
        },
        {
            "title": "AI and Machine Learning Approaches for Predicting Nanoparticles Toxicity: The Critical Role of Physiochemical Properties",
            "year": 2024,
            "url": "https://arxiv.org/abs/2409.15322",
            "used_for": ["feature_selection", "feature_explanation"],
            "full_summary": "该文献强调纳米颗粒毒性预测高度依赖理化性质，包括粒径、形状、表面电荷、材料组成、表面修饰和暴露条件。这直接支撑第一版输入 schema 和解释模板：当模型输出毒性等级时，应把剂量、粒径、Zeta 电位、材料组成、表面修饰和细胞背景作为解释候选因素，而不是只输出黑盒结论。",
            "method_relevance": "用于构建 feature importance 到中文解释的映射。",
            "data_relevance": "用于确认清洗数据必须统一粒径、Zeta 电位、剂量和暴露时间单位。",
            "limitations": "理化性质的重要性不代表单一特征可独立决定毒性，解释必须保留多因素共同作用的表述。",
            "key_points": [
                "理化性质是纳米颗粒毒性预测的核心输入。",
                "粒径、形状、表面电荷、组成和暴露条件都会影响毒性。",
                "特征选择应同时反映材料性质和生物学背景。",
            ],
        },
        {
            "title": "Smart Drug-Delivery Systems for Cancer Nanotherapy",
            "year": 2024,
            "url": "https://arxiv.org/abs/2401.11192",
            "used_for": ["cancer_nanomedicine_context"],
            "full_summary": "该文献介绍智能药物递送系统在癌症纳米治疗中的应用背景，强调纳米药物递送系统在进入更深入验证前需要考虑安全性、毒性和生物相容性。本项目据此把毒性预测定位为早期候选材料筛查工具：它可以帮助发现潜在高风险组合并建议补充体外验证，但不能替代实验。",
            "method_relevance": "用于生成风险建议和下一步实验建议。",
            "data_relevance": "用于解释为什么应用背景选择肿瘤纳米药物筛选，同时仍以体外细胞毒性数据训练模型。",
            "limitations": "该背景文献不能提供细胞毒性模型训练标签，只用于场景和建议措辞。",
            "key_points": [
                "纳米医学在癌症治疗中具有重要应用。",
                "药物递送系统在下游验证前需要进行安全性筛查。",
                "毒性预测可支持早期候选材料过滤。",
            ],
        },
    ]
    Path("literature").mkdir(parents=True, exist_ok=True)
    Path("literature/literature_base.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
```

运行：

```bash
rtk uv run python scripts/build_literature_base.py
```

预期：生成 `literature/literature_base.json`，每篇核心文献包含 `full_summary`、`method_relevance`、`data_relevance`、`limitations` 和 `key_points`。

`literature/literature_base.json` 的结构如下：

```json
[
  {
    "title": "The eNanoMapper database for nanomaterial safety information",
    "year": 2015,
    "url": "https://www.beilstein-journals.org/bjnano/articles/6/165",
    "used_for": ["data_source", "schema_design"],
    "full_summary": "该文献介绍 eNanoMapper 面向纳米材料安全信息的数据管理体系，重点是把纳米材料表征、实验条件和毒理结果组织成可查询、可复用的结构化记录。它为本项目的数据 schema 提供依据：纳米材料理化性质、生物实验条件和毒性终点必须同时保留，才能支持后续 nanoQSAR 或机器学习建模。第一版数据清洗应优先参考该类结构化安全数据，而不是只保留模型输入所需的最少字段。",
    "method_relevance": "用于定义公开数据来源优先级、字段追溯和纳米材料安全数据管理边界。",
    "data_relevance": "支持把材料组成、粒径、表面电荷、暴露条件、细胞类型和毒性终点作为统一清洗对象。",
    "limitations": "数据库记录来自多实验条件，直接合并前必须处理单位、缺失值、重复记录和终点定义差异。",
    "key_points": [
      "eNanoMapper 面向纳米材料安全信息管理。",
      "该数据库支持结构化的纳米材料表征和毒理数据。",
      "它可以支持 nanoQSAR 和机器学习工作流。"
    ]
  },
  {
    "title": "Predicting Cytotoxicity of Nanoparticles: A Meta-Analysis Using Machine Learning",
    "year": 2024,
    "url": "https://doi.org/10.1021/acsanm.4c02269",
    "used_for": ["model_design", "feature_selection", "feature_explanation"],
    "full_summary": "该文献以机器学习方式总结纳米颗粒细胞毒性预测任务，强调细胞活力等毒性终点可以由材料组成、剂量、粒径、Zeta 电位、暴露时间和细胞背景共同预测。它支撑第一版同时训练分类模型和回归模型：分类模型输出低毒、中毒、高毒，回归模型输出细胞活力百分比。它也支撑随机森林作为第一版模型，因为随机森林适合处理中小规模表格数据、非线性关系和特征重要性解释。",
    "method_relevance": "用于选择 RandomForestClassifier、RandomForestRegressor、holdout 评估和 feature importance 解释。",
    "data_relevance": "用于确认组成、剂量、粒径、Zeta 电位和暴露时间是清洗数据必须保留的核心字段。",
    "limitations": "不同文献的细胞毒性实验条件并不完全一致，模型表现不能被解释为跨所有纳米材料和细胞类型的通用毒理结论。",
    "key_points": [
      "机器学习可用于预测纳米颗粒细胞毒性。",
      "组成、浓度、Zeta 电位、粒径和暴露时间是常见预测特征。",
      "随机森林适用于细胞活力预测和特征重要性分析。"
    ]
  },
  {
    "title": "Application of Machine Learning in Nanotoxicology: A Critical Review and Perspective",
    "year": 2024,
    "url": "https://doi.org/10.1021/acs.est.4c03217",
    "used_for": ["background", "limitations"],
    "full_summary": "该综述说明机器学习在纳米毒理学中可以加速风险筛查和假设生成，但核心瓶颈来自数据规模有限、来源异质、表征字段不完整、实验协议差异和模型可解释性不足。本项目据此把第一版定位为公开数据驱动的原型系统，而不是临床或监管结论系统。解释层必须说明数据和模型局限，不能让 LLM 编造指标或替代模型结论。",
    "method_relevance": "用于约束模型解释、报告措辞和第一版适用范围。",
    "data_relevance": "用于要求 metadata 记录来源、清洗规则、单位规范化、去重策略和排除记录原则。",
    "limitations": "综述强调的异质性问题不会因清洗而消失，第一版应显式披露残余偏差。",
    "key_points": [
      "机器学习可以加速纳米毒理学分析。",
      "纳米毒理学数据集通常规模较小且异质性强。",
      "可解释性和数据质量仍是核心挑战。"
    ]
  },
  {
    "title": "caNanoLab: data sharing to expedite the use of nanotechnology in biomedicine",
    "year": 2013,
    "url": "https://cananolab.cancer.gov/",
    "used_for": ["cancer_nanomedicine_context", "data_source"],
    "full_summary": "caNanoLab 面向癌症纳米技术数据共享，提供纳米材料、表征、样品和生物医学应用相关记录。它为本项目的肿瘤纳米药物筛选背景提供依据：应用场景可以聚焦候选纳米材料在肿瘤相关细胞或正常对照细胞中的体外毒性风险，但系统不能声称预测人体临床毒性或抗肿瘤疗效。",
    "method_relevance": "用于限定 Streamlit 应用和 README 的产品边界。",
    "data_relevance": "用于补充癌症纳米技术背景来源，并与通用纳米安全数据交叉验证材料和实验字段。",
    "limitations": "caNanoLab 更偏生物医学数据共享平台，直接训练前仍需筛选具有明确体外细胞毒性终点的记录。",
    "key_points": [
      "caNanoLab 支持癌症纳米技术数据共享。",
      "该平台为生物医学和癌症纳米材料应用提供背景。",
      "它可作为肿瘤纳米医学筛选场景的背景资料。"
    ]
  },
  {
    "title": "AI and Machine Learning Approaches for Predicting Nanoparticles Toxicity: The Critical Role of Physiochemical Properties",
    "year": 2024,
    "url": "https://arxiv.org/abs/2409.15322",
    "used_for": ["feature_selection", "feature_explanation"],
    "full_summary": "该文献强调纳米颗粒毒性预测高度依赖理化性质，包括粒径、形状、表面电荷、材料组成、表面修饰和暴露条件。这直接支撑第一版输入 schema 和解释模板：当模型输出毒性等级时，应把剂量、粒径、Zeta 电位、材料组成、表面修饰和细胞背景作为解释候选因素，而不是只输出黑盒结论。",
    "method_relevance": "用于构建 feature importance 到中文解释的映射。",
    "data_relevance": "用于确认清洗数据必须统一粒径、Zeta 电位、剂量和暴露时间单位。",
    "limitations": "理化性质的重要性不代表单一特征可独立决定毒性，解释必须保留多因素共同作用的表述。",
    "key_points": [
      "理化性质是纳米颗粒毒性预测的核心输入。",
      "粒径、形状、表面电荷、组成和暴露条件都会影响毒性。",
      "特征选择应同时反映材料性质和生物学背景。"
    ]
  },
  {
    "title": "Smart Drug-Delivery Systems for Cancer Nanotherapy",
    "year": 2024,
    "url": "https://arxiv.org/abs/2401.11192",
    "used_for": ["cancer_nanomedicine_context"],
    "full_summary": "该文献介绍智能药物递送系统在癌症纳米治疗中的应用背景，强调纳米药物递送系统在进入更深入验证前需要考虑安全性、毒性和生物相容性。本项目据此把毒性预测定位为早期候选材料筛查工具：它可以帮助发现潜在高风险组合并建议补充体外验证，但不能替代实验。",
    "method_relevance": "用于生成风险建议和下一步实验建议。",
    "data_relevance": "用于解释为什么应用背景选择肿瘤纳米药物筛选，同时仍以体外细胞毒性数据训练模型。",
    "limitations": "该背景文献不能提供细胞毒性模型训练标签，只用于场景和建议措辞。",
    "key_points": [
      "纳米医学在癌症治疗中具有重要应用。",
      "药物递送系统在下游验证前需要进行安全性筛查。",
      "毒性预测可支持早期候选材料过滤。"
    ]
  }
]
```

- [ ] **步骤 4：编写公开数据测试**

创建 `tests/test_public_data.py`：

```python
from pathlib import Path
import json

import pandas as pd

from nano_tox_agent.schema import FEATURE_COLUMNS, TARGET_REGRESSION


def test_data_source_policy_matches_spec_priority():
    sources = json.loads(Path("data/raw/data_sources.json").read_text(encoding="utf-8"))
    names = [source["name"] for source in sources["priority_order"]]
    assert names == ["eNanoMapper", "caNanoLab", "Curated nanotoxicity ML datasets"]
    assert sources["required_normalized_columns"] == FEATURE_COLUMNS + [TARGET_REGRESSION]
    assert sources["source_policy"].startswith("第一版必须接入完整公开数据清洗结果")


def test_public_data_pipeline_scripts_exist():
    assert Path("scripts/clean_public_data.py").exists()
    assert Path("scripts/build_literature_base.py").exists()


def test_cleaned_public_dataset_exists_and_matches_schema():
    frame = pd.read_csv("data/processed/toxicity_clean.csv")
    assert list(frame.columns) == FEATURE_COLUMNS + [TARGET_REGRESSION]
    assert len(frame) >= 100
    assert not frame.duplicated().any()


def test_cleaned_public_dataset_values_are_in_valid_ranges():
    frame = pd.read_csv("data/processed/toxicity_clean.csv")
    assert frame["particle_size_nm"].gt(0).all()
    assert frame["dose_ug_ml"].ge(0).all()
    assert frame["exposure_time_h"].gt(0).all()
    assert frame["cell_viability_percent"].between(0, 100).all()
    for column in ["material_type", "cell_type", "assay_method"]:
        assert frame[column].astype(str).str.strip().ne("").all()


def test_cleaned_public_dataset_metadata_matches_file():
    frame = pd.read_csv("data/processed/toxicity_clean.csv")
    metadata = json.loads(Path("data/processed/toxicity_clean_metadata.json").read_text(encoding="utf-8"))
    sources = json.loads(Path("data/raw/data_sources.json").read_text(encoding="utf-8"))
    assert metadata["dataset_type"] == "cleaned_public_nanotoxicity_dataset"
    assert metadata["record_count"] == len(frame)
    assert metadata["record_count"] >= 100
    assert sum(metadata["source_record_counts"].values()) >= metadata["record_count"]
    assert sources["cleaned_output_path"] == "data/processed/toxicity_clean.csv"
    assert "合成" not in metadata["purpose"]
    assert "演示样例" not in metadata["purpose"]
```

- [ ] **步骤 5：编写核心文献测试**

创建 `tests/test_literature.py`：

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


def test_literature_entries_have_complete_summaries():
    required_titles = {
        "The eNanoMapper database for nanomaterial safety information",
        "Predicting Cytotoxicity of Nanoparticles: A Meta-Analysis Using Machine Learning",
        "Application of Machine Learning in Nanotoxicology: A Critical Review and Perspective",
        "caNanoLab: data sharing to expedite the use of nanotechnology in biomedicine",
        "AI and Machine Learning Approaches for Predicting Nanoparticles Toxicity: The Critical Role of Physiochemical Properties",
        "Smart Drug-Delivery Systems for Cancer Nanotherapy",
    }
    entries = load_literature(Path("literature/literature_base.json"))
    assert {entry["title"] for entry in entries} == required_titles
    for entry in entries:
        assert len(entry["full_summary"]) >= 120
        assert entry["method_relevance"]
        assert entry["data_relevance"]
        assert entry["limitations"]
        assert len(entry["key_points"]) >= 3
```

- [ ] **步骤 6：创建文献模块**

创建 `src/nano_tox_agent/literature.py`：

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

- [ ] **步骤 7：验证完整子计划**

运行：

```bash
rtk uv run pytest tests/test_schema.py tests/test_public_data.py tests/test_literature.py -q
```

预期：通过，显示 `14 passed`。

- [ ] **步骤 8：提交**

```bash
rtk git add pyproject.toml .python-version .gitignore src/nano_tox_agent scripts tests data/raw/data_sources.json data/processed/toxicity_clean.csv data/processed/toxicity_clean_metadata.json literature/literature_base.json
rtk git commit -m "feat: add foundation data and literature library"
```
