# Nano Toxicity Agent Explanation and LLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate deterministic Chinese explanations and provide an optional LLM rewrite layer that safely falls back when no API is configured.

**Architecture:** This subplan depends on subplans 01 and 02 for schema, prediction result types, and literature entries. It does not alter model prediction decisions.

**Tech Stack:** Python standard library, pytest.

---

## Files

- Create: `src/nano_tox_agent/explain.py`
- Create: `src/nano_tox_agent/llm_layer.py`
- Create: `tests/test_explain_llm.py`

## Task 1: Explanation Engine

- [ ] **Step 1: Write failing tests**

Create `tests/test_explain_llm.py`:

```python
from nano_tox_agent.explain import build_explanation
from nano_tox_agent.llm_layer import generate_llm_response
from nano_tox_agent.schema import PredictionInput, PredictionResult


def test_build_explanation_includes_prediction_and_suggestions():
    sample = PredictionInput(35.0, 25.0, 90.0, 24.0, "zinc_oxide", "unmodified", "A549", "Human", "Lung", "MTT")
    result = PredictionResult("high", 42.0, 0.88, [("numeric__dose_ug_ml", 0.31), ("numeric__particle_size_nm", 0.22)])
    literature = [{"title": "Predicting Cytotoxicity of Nanoparticles: A Meta-Analysis Using Machine Learning", "key_points": ["Dose and particle size are common predictive features."]}]
    explanation = build_explanation(sample, result, literature)
    assert "预测毒性等级：高毒" in explanation
    assert "预测细胞活力：42.0%" in explanation
    assert "建议降低暴露剂量" in explanation
    assert "Predicting Cytotoxicity of Nanoparticles" in explanation


def test_generate_llm_response_returns_none_without_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_URL", raising=False)
    assert generate_llm_response("explain this") is None
```

- [ ] **Step 2: Verify failure**

Run: `rtk pytest tests/test_explain_llm.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'nano_tox_agent.explain'`.

- [ ] **Step 3: Create explanation module**

Create `src/nano_tox_agent/explain.py`:

```python
from typing import Any

from nano_tox_agent.schema import PredictionInput, PredictionResult

TOXICITY_TEXT = {"low": "低毒", "medium": "中毒", "high": "高毒"}


def build_explanation(sample: PredictionInput, result: PredictionResult, literature_entries: list[dict[str, Any]]) -> str:
    label = TOXICITY_TEXT.get(result.toxicity_level, result.toxicity_level)
    feature_text = "、".join(_clean_feature_name(name) for name, _score in result.top_features)
    return "\n".join([
        f"预测毒性等级：{label}",
        f"预测细胞活力：{result.cell_viability_percent}%",
        f"模型置信度：{result.confidence}",
        "",
        f"主要影响因素：{feature_text}",
        "",
        "解释：",
        f"该样本的剂量为 {sample.dose_ug_ml} μg/mL，粒径为 {sample.particle_size_nm} nm，Zeta 电位为 {sample.zeta_potential_mv} mV。模型根据这些理化性质、实验条件和细胞类型综合判断毒性风险。",
        "",
        "建议：",
        _build_suggestion(result),
        "",
        "文献依据：",
        _format_literature(literature_entries),
    ])


def _build_suggestion(result: PredictionResult) -> str:
    if result.toxicity_level == "high":
        return "建议降低暴露剂量，增加 PEG 等表面修饰，并补充正常细胞系对照验证。"
    if result.toxicity_level == "medium":
        return "建议增加剂量梯度实验，并比较癌细胞系与正常细胞系的敏感性差异。"
    return "可进入下一轮体外验证，并保留正常细胞系对照以确认安全性。"


def _format_literature(literature_entries: list[dict[str, Any]]) -> str:
    lines = []
    for entry in literature_entries[:3]:
        points = entry.get("key_points", [])
        point = points[0] if points else "该文献用于支撑本次解释。"
        lines.append(f"- {entry.get('title', 'Untitled')}: {point}")
    return "\n".join(lines)


def _clean_feature_name(feature_name: str) -> str:
    return feature_name.replace("numeric__", "").replace("categorical__", "").replace("_", " ")
```

- [ ] **Step 4: Create optional LLM module**

Create `src/nano_tox_agent/llm_layer.py`:

```python
import json
import os
import urllib.request


def generate_llm_response(prompt: str) -> str | None:
    api_key = os.environ.get("LLM_API_KEY")
    api_url = os.environ.get("LLM_API_URL")
    if not api_key or not api_url:
        return None
    payload = json.dumps({"prompt": prompt}).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        body = json.loads(response.read().decode("utf-8"))
    text = body.get("text")
    return text.strip() if isinstance(text, str) and text.strip() else None
```

- [ ] **Step 5: Verify subplan**

Run:

```bash
rtk pytest tests/test_explain_llm.py -q
rtk pytest tests/test_schema.py tests/test_literature.py tests/test_train_predict.py tests/test_explain_llm.py -q
```

Expected: PASS with all existing tests passing.

- [ ] **Step 6: Commit**

```bash
rtk git add src/nano_tox_agent/explain.py src/nano_tox_agent/llm_layer.py tests/test_explain_llm.py
rtk git commit -m "feat: add explanation engine and llm fallback"
```
