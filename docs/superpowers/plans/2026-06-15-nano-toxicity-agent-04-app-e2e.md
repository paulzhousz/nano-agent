# Nano Toxicity Agent Streamlit App and E2E Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the runnable Streamlit interface, README, and end-to-end verification commands.

**Architecture:** This subplan depends on subplans 01-03. The UI calls the prediction service, literature loader, explanation engine, and optional LLM layer without changing their behavior.

**Tech Stack:** Streamlit, pytest, Python CLI verification.

---

## Files

- Create: `app.py`
- Create: `tests/test_app_payload.py`
- Create: `README.md`

## Task 1: Streamlit Payload Helper and App

- [ ] **Step 1: Write failing app helper test**

Create `tests/test_app_payload.py`:

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

- [ ] **Step 2: Verify failure**

Run: `rtk pytest tests/test_app_payload.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'app'`.

- [ ] **Step 3: Create Streamlit app**

Create `app.py`:

```python
from pathlib import Path

import streamlit as st

from nano_tox_agent.explain import TOXICITY_TEXT, build_explanation
from nano_tox_agent.literature import load_literature, select_literature
from nano_tox_agent.llm_layer import generate_llm_response
from nano_tox_agent.predict import load_or_train_bundle, predict_toxicity
from nano_tox_agent.schema import PredictionInput, PredictionResult

DATA_PATH = Path("data/raw/toxicity_samples.csv")
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

- [ ] **Step 4: Verify app helper**

Run: `rtk pytest tests/test_app_payload.py -q`

Expected: PASS with `4 passed`.

## Task 2: README and End-to-End Checks

- [ ] **Step 1: Create README**

Create `README.md`:

````markdown
# Nano Toxicity Agent

Runnable Streamlit prototype for nanomaterial cytotoxicity prediction.

## Scope

The app predicts in vitro cytotoxicity risk for candidate nanomaterials. Tumor nanomedicine is the application background; the app does not predict clinical human toxicity or anti-tumor efficacy.

## Setup

```bash
python -m pip install -e ".[dev]"
```

## Test

```bash
pytest -q
```

## Run

```bash
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

## Optional LLM layer

Set `LLM_API_KEY`, `LLM_API_BASE`, and optionally `LLM_MODEL` to enable OpenAI-compatible explanation rewriting. The LLM never decides toxicity level or cell viability.

## Manual acceptance checklist

1. Start the app with `streamlit run app.py --server.address 127.0.0.1 --server.port 8501`.
2. Open `http://127.0.0.1:8501`.
3. Keep the default sidebar values and click `预测毒性`.
4. Verify the page shows `毒性等级`, `预测细胞活力`, `模型置信度`, `智能体解释`, and `特征重要性`.
5. Enable `使用可选 LLM 改写解释` without environment variables and click `预测毒性` again.
6. Verify prediction still succeeds through template fallback.
````

- [ ] **Step 2: Run full tests**

Run: `rtk pytest -q`

Expected: PASS with all tests passing, including the two Streamlit `AppTest` smoke checks.

- [ ] **Step 3: Verify CLI prediction**

Run:

```bash
rtk python -c "from pathlib import Path; from nano_tox_agent.predict import load_or_train_bundle, predict_toxicity; from nano_tox_agent.schema import PredictionInput; bundle=load_or_train_bundle(Path('models/toxicity_bundle.joblib'), Path('data/raw/toxicity_samples.csv')); sample=PredictionInput(90,-8,50,24,'liposome','PEG','A549','Human','Lung','MTT'); print(predict_toxicity(bundle, sample))"
```

Expected: prints a `PredictionResult` with toxicity level `low`, `medium`, or `high`.

- [ ] **Step 4: Launch local app**

Run:

```bash
rtk streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Expected: Streamlit prints a local URL ending with `http://127.0.0.1:8501`.

- [ ] **Step 5: Run manual browser acceptance**

Open `http://127.0.0.1:8501` and complete the README checklist:

```text
[ ] Click 预测毒性 with default values.
[ ] Confirm toxicity level, cell viability, and confidence metrics render.
[ ] Confirm 智能体解释 renders text instead of a stack trace.
[ ] Confirm 特征重要性 renders at least one feature row.
[ ] Enable 使用可选 LLM 改写解释 with no LLM env vars and confirm template fallback still renders.
```

Expected: all checklist items pass.

- [ ] **Step 6: Commit**

```bash
rtk git add app.py tests/test_app_payload.py README.md
rtk git commit -m "feat: add streamlit app and e2e docs"
```
