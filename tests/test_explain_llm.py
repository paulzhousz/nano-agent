from nano_tox_agent.explain import build_explanation

try:
    from nano_tox_agent.llm_layer import generate_llm_response  # noqa: F401
except ImportError:  # pragma: no cover - 任务 1 只实现解释引擎
    generate_llm_response = None  # noqa: F401

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
