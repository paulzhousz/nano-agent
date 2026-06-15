# 纳米毒性预测智能体 Streamlit 应用与 E2E 实现计划

> **给智能体执行者：** 必须使用子技能：`superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，按任务逐步执行本计划。步骤使用复选框（`- [ ]`）语法跟踪。

**目标：** 添加可运行的 Streamlit 界面、README 和端到端验证命令。

**架构：** 本子计划依赖子计划 01-03。UI 调用预测服务、文献加载器、解释引擎和可选 LLM 层，但不改变它们的行为。

**技术栈：** Python 3.12、uv、Streamlit、pytest、Python CLI 验证。

---

## 文件

- 创建：`app.py`
- 创建：`tests/test_app_payload.py`
- 创建：`README.md`

## 任务 1：Streamlit 结果载荷辅助函数与应用

- [ ] **步骤 1：编写失败的应用 helper 测试**

创建 `tests/test_app_payload.py`：

```python
from streamlit.testing.v1 import AppTest

from app import build_result_payload
from nano_tox_agent.schema import PredictionResult


def test_build_result_payload_maps_labels_for_display():
    result = PredictionResult("medium", 66.5, 0.72, [("numeric__dose_ug_ml", 0.3)])
    payload = build_result_payload(result, "解释文本")
    assert payload["toxicity_label"] == "中毒"
    assert payload["cell_viability_percent"] == "66.5%"
    assert payload["confidence"] == "0.72"
    assert payload["explanation"] == "解释文本"


def test_format_user_error_is_actionable():
    from app import format_user_error

    assert "数据或模型文件缺失" in format_user_error(FileNotFoundError("missing"))
    assert "输入或数据格式错误" in format_user_error(ValueError("bad data"))
    assert "预测失败" in format_user_error(RuntimeError("boom"))


def test_streamlit_default_prediction_flow_renders_result():
    app = AppTest.from_file("app.py")
    app.run()
    assert not app.exception

    app.button[0].click().run()
    assert not app.exception

    metric_labels = [metric.label for metric in app.metric]
    assert "毒性等级" in metric_labels
    assert "预测细胞活力" in metric_labels
    assert "模型置信度" in metric_labels
    assert any("智能体解释" in subheader.value for subheader in app.subheader)
    assert any("特征重要性" in subheader.value for subheader in app.subheader)


def test_streamlit_llm_checkbox_without_env_uses_template_fallback(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_BASE", raising=False)

    app = AppTest.from_file("app.py")
    app.run()
    assert not app.exception

    app.checkbox[0].check().run()
    app.button[0].click().run()
    assert not app.exception
    assert any("智能体解释" in subheader.value for subheader in app.subheader)
```

- [ ] **步骤 2：验证失败**

运行：`rtk uv run pytest tests/test_app_payload.py -q`

预期：失败，报 `ModuleNotFoundError: No module named 'app'`。

- [ ] **步骤 3：创建 Streamlit 应用**

创建 `app.py`：

```python
from pathlib import Path

import streamlit as st

from nano_tox_agent.explain import TOXICITY_TEXT, build_explanation
from nano_tox_agent.literature import load_literature, select_literature
from nano_tox_agent.llm_layer import generate_llm_response
from nano_tox_agent.predict import load_or_train_bundle, predict_toxicity
from nano_tox_agent.schema import PredictionInput, PredictionResult

DATA_PATH = Path("data/processed/toxicity_clean.csv")
MODEL_PATH = Path("models/toxicity_bundle.joblib")
LITERATURE_PATH = Path("literature/literature_base.json")


def build_result_payload(result: PredictionResult, explanation: str) -> dict[str, str]:
    return {
        "toxicity_label": TOXICITY_TEXT.get(result.toxicity_level, result.toxicity_level),
        "cell_viability_percent": f"{result.cell_viability_percent}%",
        "confidence": f"{result.confidence}",
        "explanation": explanation,
    }


def format_user_error(error: Exception) -> str:
    if isinstance(error, FileNotFoundError):
        return f"数据或模型文件缺失：{error}"
    if isinstance(error, ValueError):
        return f"输入或数据格式错误：{error}"
    return f"预测失败：{error}"


def main() -> None:
    st.set_page_config(page_title="纳米毒性预测智能体", layout="wide")
    st.title("面向肿瘤纳米药物筛选的纳米材料毒性预测智能体")
    with st.sidebar:
        st.header("候选材料参数")
        sample = PredictionInput(
            st.number_input("粒径 (nm)", min_value=1.0, value=90.0, step=1.0),
            st.number_input("Zeta 电位 (mV)", value=-8.0, step=1.0),
            st.number_input("暴露剂量 (μg/mL)", min_value=0.0, value=50.0, step=5.0),
            st.number_input("暴露时间 (h)", min_value=1.0, value=24.0, step=1.0),
            st.selectbox("材料类型", ["liposome", "polymer", "metal_oxide", "gold", "silver", "carbon", "zinc_oxide", "copper_oxide"]),
            st.selectbox("表面修饰", ["PEG", "unmodified", "citrate"]),
            st.selectbox("细胞类型", ["A549", "BEAS-2B", "HepG2", "THP-1", "NRK-52E"]),
            st.selectbox("来源物种", ["Human", "Rat", "Mouse"]),
            st.selectbox("组织来源", ["Lung", "Liver", "Blood", "Kidney"]),
            st.selectbox("检测方法", ["MTT", "LDH"]),
        )
        use_llm = st.checkbox("使用可选 LLM 改写解释", value=False)

    if st.button("预测毒性"):
        try:
            bundle = load_or_train_bundle(MODEL_PATH, DATA_PATH)
            literature = select_literature(load_literature(LITERATURE_PATH), "feature_explanation")
            result = predict_toxicity(bundle, sample)
            explanation = build_explanation(sample, result, literature)
            if use_llm:
                explanation = generate_llm_response(explanation) or explanation
            payload = build_result_payload(result, explanation)
        except Exception as error:
            st.error(format_user_error(error))
            return
        col1, col2, col3 = st.columns(3)
        col1.metric("毒性等级", payload["toxicity_label"])
        col2.metric("预测细胞活力", payload["cell_viability_percent"])
        col3.metric("模型置信度", payload["confidence"])
        st.subheader("智能体解释")
        st.text(payload["explanation"])
        st.subheader("特征重要性")
        for feature_name, score in result.top_features:
            st.write(f"{feature_name}: {score}")


if __name__ == "__main__":
    main()
```

- [ ] **步骤 4：验证应用 helper**

运行：`rtk uv run pytest tests/test_app_payload.py -q`

预期：通过，显示 `4 passed`。

## 任务 2：README 与端到端检查

- [ ] **步骤 1：创建 README**

创建 `README.md`：

````markdown
# 纳米毒性预测智能体

用于纳米材料细胞毒性预测的可运行 Streamlit 原型。

## 范围

本应用预测候选纳米材料的体外细胞毒性风险。肿瘤纳米医学是应用背景；本应用不预测临床人体毒性，也不预测抗肿瘤疗效。

第一版模型使用 `data/processed/toxicity_clean.csv` 中的完整公开数据清洗结果训练。报告和演示必须同时披露 `data/processed/toxicity_clean_metadata.json` 中的数据来源、清洗规则和记录数量。

## 安装

```bash
rtk uv venv --python 3.12
rtk uv sync --extra dev
```

## 测试

```bash
rtk uv run pytest -q
```

## 运行

```bash
rtk uv run streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

## 可选 LLM 层

设置 `LLM_API_KEY`、`LLM_API_BASE` 和可选的 `LLM_MODEL`，即可启用 OpenAI-compatible 的解释改写。LLM 永远不决定毒性等级或细胞活力。

## 手动验收清单

1. 使用 `rtk uv run streamlit run app.py --server.address 127.0.0.1 --server.port 8501` 启动应用。
2. 打开 `http://127.0.0.1:8501`。
3. 保持侧边栏默认值，点击 `预测毒性`。
4. 确认页面显示 `毒性等级`、`预测细胞活力`、`模型置信度`、`智能体解释` 和 `特征重要性`。
5. 在不设置环境变量的情况下启用 `使用可选 LLM 改写解释`，再次点击 `预测毒性`。
6. 确认预测仍通过模板 fallback 成功渲染。
````

- [ ] **步骤 2：运行完整测试**

运行：`rtk uv run pytest -q`

预期：通过，所有测试都通过，包括两个 Streamlit `AppTest` smoke 检查。

- [ ] **步骤 3：验证 CLI 预测**

运行：

```bash
rtk uv run python -c "from pathlib import Path; from nano_tox_agent.predict import load_or_train_bundle, predict_toxicity; from nano_tox_agent.schema import PredictionInput; bundle=load_or_train_bundle(Path('models/toxicity_bundle.joblib'), Path('data/processed/toxicity_clean.csv')); sample=PredictionInput(90,-8,50,24,'liposome','PEG','A549','Human','Lung','MTT'); print(predict_toxicity(bundle, sample))"
```

预期：打印一个 `PredictionResult`，其中毒性等级为 `low`、`medium` 或 `high`。

- [ ] **步骤 4：启动本地应用**

运行：

```bash
rtk uv run streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

预期：Streamlit 打印一个以 `http://127.0.0.1:8501` 结尾的本地 URL。

- [ ] **步骤 5：运行手动浏览器验收**

打开 `http://127.0.0.1:8501`，并完成 README 清单：

```text
[ ] 使用默认值点击 预测毒性。
[ ] 确认毒性等级、细胞活力和置信度指标已渲染。
[ ] 确认 智能体解释 渲染为文本，而不是堆栈错误。
[ ] 确认 特征重要性 至少渲染一行特征。
[ ] 在没有 LLM 环境变量的情况下启用 使用可选 LLM 改写解释，并确认模板 fallback 仍然渲染。
```

预期：所有清单项都通过。

- [ ] **步骤 6：提交**

```bash
rtk git add app.py tests/test_app_payload.py README.md
rtk git commit -m "feat: add streamlit app and e2e docs"
```
