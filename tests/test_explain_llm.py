import os
import json
from unittest.mock import patch

import pytest

from nano_tox_agent.explain import build_explanation, clean_feature_name
from nano_tox_agent.llm_layer import generate_llm_response, load_env_file
from nano_tox_agent.schema import PredictionInput, PredictionResult


def test_clean_feature_name_removes_model_prefixes_and_formats_spacing():
    assert clean_feature_name("numeric__dose_ug_ml") == "dose ug ml"
    assert clean_feature_name("categorical__surface_modification_PEG") == "surface modification PEG"


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
    assert "剂量 50.0 ug/mL" in explanation
    assert "粒径 90.0 nm" in explanation
    assert "Zeta 电位 -8.0 mV" in explanation
    assert "理化性质、实验条件和细胞类型综合判断" in explanation
    assert "建议降低暴露剂量" in explanation
    assert "Predicting Cytotoxicity of Nanoparticles" in explanation


def test_generate_llm_response_returns_none_without_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_BASE", raising=False)

    assert generate_llm_response("explain this") is None


def test_load_env_file_sets_missing_llm_variables_from_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_API_KEY=dotenv-key",
                "LLM_API_BASE=https://dotenv.example.com/v1",
                "LLM_MODEL=qwen-plus",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_BASE", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    load_env_file(env_file)

    assert os.getenv("LLM_API_KEY") == "dotenv-key"
    assert os.getenv("LLM_API_BASE") == "https://dotenv.example.com/v1"
    assert os.getenv("LLM_MODEL") == "qwen-plus"


def test_load_env_file_does_not_override_existing_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_API_KEY=dotenv-key",
                "LLM_API_BASE=https://dotenv.example.com/v1",
                "LLM_MODEL=dotenv-model",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_API_KEY", "shell-key")
    monkeypatch.setenv("LLM_API_BASE", "https://shell.example.com/v1")
    monkeypatch.setenv("LLM_MODEL", "shell-model")

    load_env_file(env_file)

    assert os.getenv("LLM_API_KEY") == "shell-key"
    assert os.getenv("LLM_API_BASE") == "https://shell.example.com/v1"
    assert os.getenv("LLM_MODEL") == "shell-model"


def test_load_env_file_ignores_blank_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_API_KEY=",
                "LLM_API_BASE=   ",
                "LLM_MODEL=",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_BASE", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    load_env_file(env_file)

    assert os.getenv("LLM_API_KEY") is None
    assert os.getenv("LLM_API_BASE") is None
    assert os.getenv("LLM_MODEL") is None


def test_generate_llm_response_returns_none_when_env_vars_are_blank(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_API_BASE", " ")

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


def test_generate_llm_response_returns_none_on_non_utf8_response(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_API_BASE", "https://api.example.com/v1")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"\xff\xfe\xfd"

    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        assert generate_llm_response("explain this") is None


@pytest.mark.parametrize(
    "rewritten",
    [
        "\n".join(
            [
                "预测毒性等级：高毒",
                "预测细胞活力：42.0%",
                "解释：保留了细胞活力但漏掉了模型置信度。",
            ]
        ),
        "\n".join(
            [
                "预测毒性等级：低毒",
                "预测细胞活力：42.0%",
                "模型置信度：91.0%",
                "解释：改坏了毒性等级。",
            ]
        ),
        "\n".join(
            [
                "预测毒性等级：高毒",
                "预测细胞活力：55.0%",
                "模型置信度：91.0%",
                "解释：改坏了细胞活力。",
            ]
        ),
    ],
)
def test_generate_llm_response_returns_none_when_key_conclusion_lines_missing_or_changed(
    monkeypatch, rewritten
):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_API_BASE", "https://api.example.com/v1")
    original_explanation = "\n".join(
        [
            "预测毒性等级：高毒",
            "预测细胞活力：42.0%",
            "模型置信度：91.0%",
            "解释：原始解释正文。",
        ]
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": rewritten}}]}
            ).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        assert generate_llm_response(original_explanation) is None


def test_generate_llm_response_accepts_rewrite_when_key_conclusion_lines_preserved(
    monkeypatch,
):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_API_BASE", "https://api.example.com/v1")
    original_explanation = "\n".join(
        [
            "预测毒性等级：高毒",
            "预测细胞活力：42.0%",
            "模型置信度：91.0%",
            "解释：原始解释正文。",
        ]
    )
    rewritten = "\n".join(
        [
            "预测毒性等级：高毒",
            "预测细胞活力：42.0%",
            "模型置信度：91.0%",
            "解释：这是更流畅的改写，但保留关键结论。",
        ]
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": rewritten}}]}
            ).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        assert generate_llm_response(original_explanation) == rewritten


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
