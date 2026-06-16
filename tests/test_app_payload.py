import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_FILE = REPO_ROOT / "app.py"

sys.path.insert(0, str(REPO_ROOT))

from app import DATA_PATH, LITERATURE_PATH, MODEL_PATH, build_result_payload
from nano_tox_agent.explain import build_explanation, clean_feature_name
from nano_tox_agent.literature import load_literature, select_literature
from nano_tox_agent.predict import load_or_train_bundle, predict_toxicity
from nano_tox_agent.schema import PredictionInput
from nano_tox_agent.schema import PredictionResult


def make_sample(**overrides):
    values = {
        "particle_size_nm": 90.0,
        "zeta_potential_mv": -8.0,
        "dose_ug_ml": 50.0,
        "exposure_time_h": 24.0,
        "material_type": "liposome",
        "surface_modification": "PEG",
        "cell_type": "A549",
        "species": "Human",
        "tissue": "Lung",
        "assay_method": "MTT",
    }
    values.update(overrides)
    return PredictionInput(**values)


def build_default_prediction_context():
    sample = make_sample()
    bundle = load_or_train_bundle(MODEL_PATH, DATA_PATH)
    literature = select_literature(load_literature(LITERATURE_PATH), "feature_explanation")
    result = predict_toxicity(bundle, sample)
    explanation = build_explanation(sample, result, literature)
    return sample, result, explanation


def markdown_values(app: AppTest) -> list[str]:
    return [node.value for node in app.markdown]


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
    app = AppTest.from_file(str(APP_FILE))
    app.run()
    assert not app.exception

    app.button[0].click().run()
    assert not app.exception

    metric_labels = [metric.label for metric in app.metric]
    assert "毒性等级" in metric_labels
    assert "预测细胞活力" in metric_labels
    assert "模型置信度" in metric_labels
    values = markdown_values(app)
    assert any("当前预测毒性等级为" in value for value in values)
    assert any("模型置信度为" in value for value in values)


def test_streamlit_default_view_shows_empty_state_guidance():
    app = AppTest.from_file(str(APP_FILE))
    app.run()
    assert not app.exception

    values = markdown_values(app)
    assert any("纳米材料毒性评估与解释" in value for value in values)
    assert any("当前页面可完成什么任务" in value for value in values)
    assert any("预测后将输出哪些结果" in value for value in values)


def test_streamlit_empty_state_uses_guidance_not_result_language():
    app = AppTest.from_file(str(APP_FILE))
    app.run()
    assert not app.exception

    rendered = "\n".join(markdown_values(app))
    assert "当前页面可完成什么任务" in rendered
    assert "预测后将输出哪些结果" in rendered
    assert '<div class="panel-title">研究结论</div>' not in rendered


def test_streamlit_prediction_flow_renders_new_summary_sections():
    _, result, _ = build_default_prediction_context()
    app = AppTest.from_file(str(APP_FILE))
    app.run()
    assert not app.exception

    app.button[0].click().run()
    assert not app.exception

    values = markdown_values(app)
    rendered = "\n".join(values)
    top_feature_name, top_feature_weight = result.top_features[0]
    assert any("研究结论" in value for value in values)
    assert any("解释与依据" in value for value in values)
    assert any("特征影响" in value for value in values)
    assert "当前页面可完成什么任务" not in rendered
    assert "当前预测毒性等级为" in rendered
    assert clean_feature_name(top_feature_name) in rendered
    assert f"{top_feature_weight:.3f}" in rendered


def test_streamlit_prediction_flow_renders_input_summary_card():
    sample, _, _ = build_default_prediction_context()
    app = AppTest.from_file(str(APP_FILE))
    app.run()
    assert not app.exception

    app.button[0].click().run()
    assert not app.exception

    rendered = "\n".join(markdown_values(app))
    assert "输入摘要" in rendered
    assert f"粒径 {sample.particle_size_nm:.0f} nm" in rendered
    assert f"剂量 {sample.dose_ug_ml:.0f} μg/mL" in rendered
    assert sample.cell_type in rendered


def test_streamlit_prediction_flow_renders_explanation_as_sections():
    _, _, explanation = build_default_prediction_context()
    app = AppTest.from_file(str(APP_FILE))
    app.run()
    assert not app.exception

    app.button[0].click().run()
    assert not app.exception

    rendered = "\n".join(markdown_values(app))
    assert "解释与依据" in rendered
    assert "建议：" in rendered
    assert "文献依据：" in rendered
    assert explanation.splitlines()[-1] in rendered
    assert explanation not in markdown_values(app)


def test_streamlit_llm_checkbox_without_env_uses_template_fallback(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_BASE", raising=False)
    _, _, expected_explanation = build_default_prediction_context()

    app = AppTest.from_file(str(APP_FILE))
    app.run()
    assert not app.exception

    app.checkbox[0].check().run()
    app.button[0].click().run()
    assert not app.exception
    values = markdown_values(app)
    assert any("解释与依据" in value for value in values)
    assert expected_explanation in values
