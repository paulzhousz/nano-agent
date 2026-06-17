# 模块关系总览

这份文档面向第一次接手本项目的开发者，帮助快速理解“数据清洗 -> 模型训练 -> 在线预测 -> 解释生成 -> Streamlit 展示”的完整链路。

## 项目主链路

项目可以分成两条主线：

1. 离线准备链路
   - `scripts/clean_public_data.py` 读取公开原始数据，生成清洗后的训练集 `data/processed/toxicity_clean.csv` 和配套审计文件 `data/processed/toxicity_clean_metadata.json`
   - `scripts/build_literature_base.py` 生成文献摘要库 `literature/literature_base.json`
   - `src/nano_tox_agent/train.py` 基于清洗后的训练集训练模型，并保存为 `models/toxicity_bundle.joblib`

2. 在线预测链路
   - `app.py` 接收用户输入
   - `src/nano_tox_agent/predict.py` 加载模型并完成毒性预测
   - `src/nano_tox_agent/explain.py` 生成规则化解释文本
   - `src/nano_tox_agent/llm_layer.py` 可选地改写解释文本
   - `app.py` 将预测结果、解释文本和文献信息渲染成页面工作区

## 核心模块职责

| 模块 | 主要职责 | 关键输入 | 关键输出 |
| --- | --- | --- | --- |
| `app.py` | Streamlit 页面入口，组织输入、预测、解释和渲染 | 用户表单输入、模型文件、文献库 | 页面结果卡片、错误提示 |
| `scripts/clean_public_data.py` | 清洗公开数据并生成统一训练表 | `data/raw/source_exports/*` | `toxicity_clean.csv`、`toxicity_clean_metadata.json` |
| `scripts/build_literature_base.py` | 维护并输出静态文献摘要库 | 文献常量列表 | `literature_base.json` |
| `src/nano_tox_agent/schema.py` | 定义字段、数据类和输入校验规则 | 原始输入值、训练表字段 | `PredictionInput`、`PredictionResult`、字段常量 |
| `src/nano_tox_agent/preprocess.py` | 定义训练/推理共用的 sklearn 预处理流程 | 数值列、类别列 | `ColumnTransformer` |
| `src/nano_tox_agent/train.py` | 训练分类器与回归器，并打包模型 | 清洗后的训练数据 | `ModelBundle`、模型文件 |
| `src/nano_tox_agent/predict.py` | 执行在线推理并提取特征重要性 | `ModelBundle`、`PredictionInput` | `PredictionResult` |
| `src/nano_tox_agent/explain.py` | 基于预测结果拼装模板解释 | 预测结果、输入样本、文献条目 | 结构化解释文本 |
| `src/nano_tox_agent/literature.py` | 读取并筛选文献摘要 | 文献 JSON 文件、用途标签 | 当前页面要使用的文献条目 |
| `src/nano_tox_agent/llm_layer.py` | 可选的 LLM 改写层，且校验核心结论不被改动 | 模板解释、环境变量配置 | 改写后的解释文本或 `None` |

## 模块调用关系

可以把主要调用关系理解成下面这条链：

```text
app.py
├─ schema.PredictionInput
├─ predict.load_or_train_bundle()
│  ├─ train.load_model_bundle()
│  └─ train.load_training_data()
│     ├─ train.derive_training_labels()
│     ├─ train.build_model_bundle()
│     └─ train.save_model_bundle()
├─ literature.load_literature()
├─ literature.select_literature()
├─ predict.predict_toxicity()
│  ├─ schema.validate_prediction_input()
│  └─ predict._top_feature_importances()
├─ explain.build_explanation()
└─ llm_layer.generate_llm_response()  # 可选
```

离线数据准备链路则是：

```text
scripts/clean_public_data.py
├─ load_source_policy()
├─ validate_source_exports()
├─ _normalize_enanomapper_exports()
│  ├─ _filter_target_viability()
│  ├─ _prepare_conditions()
│  └─ _build_pchem_lookup()
├─ _coerce_and_filter()
└─ _write_metadata()
```

## 数据文件与模型文件关系

### 1. 原始数据

- `data/raw/source_exports/enanomapper/*.csv`
- `data/raw/data_sources.json`

这些文件只用于离线清洗，不直接进入前端预测。

### 2. 清洗后训练数据

- `data/processed/toxicity_clean.csv`
- `data/processed/toxicity_clean_metadata.json`

其中：
- `toxicity_clean.csv` 是训练输入的统一表
- `toxicity_clean_metadata.json` 记录来源、过滤规则、单位归一化、去重和覆盖范围等审计信息

### 3. 文献库

- `literature/literature_base.json`

它不参与模型训练，只服务于解释生成和结果展示。

### 4. 模型文件

- `models/toxicity_bundle.joblib`

该文件由 `train.py` 生成，内部包含：
- 分类器
- 回归器
- 特征列顺序
- 评估指标

`app.py` 启动预测时会优先加载这个文件；如果文件不存在，则通过 `predict.load_or_train_bundle()` 触发一次即时训练。

## 一次完整预测的执行顺序

用户在页面点击“预测”后，执行顺序大致如下：

1. `app.py` 从表单收集参数，构造 `PredictionInput`
2. `predict.load_or_train_bundle()` 确保模型可用
3. `literature.load_literature()` 读取文献库
4. `literature.select_literature(..., "feature_explanation")` 挑选解释用文献
5. `predict.predict_toxicity()` 输出：
   - 毒性等级
   - 预测细胞活力
   - 模型置信度
   - 主要特征重要性
6. `explain.build_explanation()` 生成模板解释
7. 如果用户开启 LLM 改写：
   - `llm_layer.generate_llm_response()` 请求外部接口
   - 若改写结果改变了核心结论，系统拒绝使用并自动回退模板解释
8. `app.py` 渲染研究结论、样本信息、特征影响、解释与文献依据

## 关键设计边界

### 预测与解释分离

- `predict.py` 只负责模型推理，不负责文案组织
- `explain.py` 只负责解释文本，不反向影响模型结论

这样做的好处是：
- 预测逻辑更稳定
- 解释可以独立演进
- 即使 LLM 不可用，系统也能用模板解释正常工作

### LLM 是增强层，不是决策层

`llm_layer.py` 的职责是“改写表达”，不是“决定结果”。  
它必须保留以下核心结论行：

- `预测毒性等级：...`
- `预测细胞活力：...`
- `模型置信度：...`

只要这些内容被改动，改写结果就会被丢弃。

### 数据清洗与 UI 分离

- `scripts/clean_public_data.py` 负责把异构公开数据压成统一训练表
- `app.py` 从不直接读取原始公开导出表

这样可以把“科学数据处理”和“页面展示”解耦，降低在线路径复杂度。

## 测试文件与模块映射

| 测试文件 | 主要覆盖模块 |
| --- | --- |
| `tests/test_public_data.py` | `scripts/clean_public_data.py`、训练数据产物、metadata 审计逻辑 |
| `tests/test_schema.py` | `src/nano_tox_agent/schema.py` |
| `tests/test_train_predict.py` | `preprocess.py`、`train.py`、`predict.py` |
| `tests/test_explain_llm.py` | `explain.py`、`llm_layer.py` |
| `tests/test_literature.py` | `literature.py` |
| `tests/test_app_payload.py` | `app.py` 的页面输出结构与文案 |

## 新人阅读顺序建议

如果你是第一次阅读这个仓库，建议按下面顺序看：

1. `README.md`
2. `docs/module-overview.md`
3. `src/nano_tox_agent/schema.py`
4. `scripts/clean_public_data.py`
5. `src/nano_tox_agent/train.py`
6. `src/nano_tox_agent/predict.py`
7. `src/nano_tox_agent/explain.py`
8. `src/nano_tox_agent/llm_layer.py`
9. `app.py`

这样可以先理解项目契约，再看离线数据，再看在线推理和页面层。
