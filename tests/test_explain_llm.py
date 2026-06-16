import json
from unittest.mock import patch

import pytest

from nano_tox_agent.explain import build_explanation
from nano_tox_agent.llm_layer import generate_llm_response
from nano_tox_agent.schema import PredictionInput, PredictionResult


def test_build_explanation_includes_key_chinese_fields_and_literature_title():
    sample = PredictionInput(
        particle_size_nm=90.0,
        zeta_potential_mv=-8.0,
        dose_ug_ml=50.0,
        exposure_time_h=24.0,
        material_type="liposome",
        surface_modification="PEG",
        cell_type="A549",
        species="Human",
        tissue="Lung",
        assay_method="MTT",
    )
    result = PredictionResult(
        toxicity_level="high",
        cell_viability_percent=42.0,
        confidence=0.91,
        top_features=[
            ("numeric__dose_ug_ml", 0.4),
            ("categorical__surface_modification_PEG", 0.25),
        ],
    )
    literature = [
        {
            "title": "Predicting Cytotoxicity of Nanoparticles",
            "key_points": ["机器学习可提升细胞毒性预测的一致性"],
        }
    ]

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
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_API_BASE", "https://api.example.com/v1/")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "rewritten explanation"}}]}
            ).encode("utf-8")

    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = generate_llm_response("explain this")

    assert result == "rewritten explanation"
    assert captured["request"].full_url == "https://api.example.com/v1/chat/completions"
    assert captured["timeout"] == 20


def test_generate_llm_response_returns_none_on_bad_response(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_API_BASE", "https://api.example.com/v1")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"choices":[]}'

    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        assert generate_llm_response("explain this") is None


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"choices": [None]},
        {"choices": [{"message": None}]},
        {"choices": [{"message": {"content": ""}}]},
    ],
)
def test_generate_llm_response_returns_none_on_malformed_or_empty_response(
    monkeypatch, payload
):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_API_BASE", "https://api.example.com/v1")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        assert generate_llm_response("explain this") is None
