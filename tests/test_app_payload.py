import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_FILE = REPO_ROOT / "app.py"

sys.path.insert(0, str(REPO_ROOT))

from app import DATA_PATH, LITERATURE_PATH, MODEL_PATH, build_result_payload
from nano_tox_agent.explain import build_explanation
from nano_tox_agent.literature import load_literature, select_literature
from nano_tox_agent.predict import load_or_train_bundle, predict_toxicity
from nano_tox_agent.schema import PredictionInput
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
    app = AppTest.from_file(str(APP_FILE))
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
    bundle = load_or_train_bundle(MODEL_PATH, DATA_PATH)
    literature = select_literature(load_literature(LITERATURE_PATH), "feature_explanation")
    result = predict_toxicity(bundle, sample)
    expected_explanation = build_explanation(sample, result, literature)

    app = AppTest.from_file(str(APP_FILE))
    app.run()
    assert not app.exception

    app.checkbox[0].check().run()
    app.button[0].click().run()
    assert not app.exception
    assert any("智能体解释" in subheader.value for subheader in app.subheader)
    assert len(app.text) == 1
    assert app.text[0].value == expected_explanation
