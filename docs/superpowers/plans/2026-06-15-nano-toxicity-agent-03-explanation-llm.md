# 纳米毒性预测智能体解释与 LLM 实现计划

> **给智能体执行者：** 必须使用子技能：`superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，按任务逐步执行本计划。步骤使用复选框（`- [ ]`）语法跟踪。

**目标：** 生成确定性的中文解释，并提供一个可选 LLM 改写层；未配置 API 时安全回退。

**架构：** 本子计划依赖子计划 01 和 02 提供的 schema、预测结果类型和文献条目。它不改变模型预测结论。

**技术栈：** Python 3.12、uv、Python 标准库、pytest。

---

## 文件

- 创建：`src/nano_tox_agent/explain.py`
- 创建：`src/nano_tox_agent/llm_layer.py`
- 创建：`tests/test_explain_llm.py`

## 任务 1：解释引擎

- [ ] **步骤 1：编写失败测试**

创建 `tests/test_explain_llm.py`：

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
    monkeypatch.delenv("LLM_API_BASE", raising=False)
    assert generate_llm_response("explain this") is None


def test_generate_llm_response_parses_openai_compatible_response(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"rewritten explanation"}}]}'

    def fake_urlopen(request, timeout):
        assert request.full_url == "https://api.example.com/v1/chat/completions"
        assert timeout == 20
        return FakeResponse()

    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_API_BASE", "https://api.example.com/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert generate_llm_response("explain this") == "rewritten explanation"


def test_generate_llm_response_returns_none_on_bad_response(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"choices":[]}'

    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_API_BASE", "https://api.example.com/v1")
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: FakeResponse())

    assert generate_llm_response("explain this") is None
```

- [ ] **步骤 2：验证失败**

运行：`rtk uv run pytest tests/test_explain_llm.py -q`

预期：失败，报 `ModuleNotFoundError: No module named 'nano_tox_agent.explain'`。

- [ ] **步骤 3：创建解释模块**

创建 `src/nano_tox_agent/explain.py`：

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

- [ ] **步骤 4：创建可选 LLM 模块**

创建 `src/nano_tox_agent/llm_layer.py`：

```python
import json
import os
import urllib.request
from urllib.error import URLError


def generate_llm_response(prompt: str) -> str | None:
    api_key = os.environ.get("LLM_API_KEY")
    api_base = os.environ.get("LLM_API_BASE")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    if not api_key or not api_base:
        return None
    api_url = api_base.rstrip("/") + "/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "请用清晰中文改写模型拥有的毒性预测解释。不要改变毒性等级、细胞活力、置信度或引用依据。",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError):
        return None
    choices = body.get("choices", [])
    if not choices:
        return None
    message = choices[0].get("message", {})
    text = message.get("content")
    return text.strip() if isinstance(text, str) and text.strip() else None
```

- [ ] **步骤 5：验证子计划**

运行：

```bash
rtk uv run pytest tests/test_explain_llm.py -q
rtk uv run pytest tests/test_schema.py tests/test_literature.py tests/test_train_predict.py tests/test_explain_llm.py -q
```

预期：通过，并且所有既有测试都通过。

- [ ] **步骤 6：提交**

```bash
rtk git add src/nano_tox_agent/explain.py src/nano_tox_agent/llm_layer.py tests/test_explain_llm.py
rtk git commit -m "feat: add explanation engine and llm fallback"
```
