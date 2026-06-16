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
        sample = PredictionInput(
            particle_size_nm=st.number_input("粒径 (nm)", min_value=1.0, value=90.0, step=1.0),
            zeta_potential_mv=st.number_input("Zeta 电位 (mV)", value=-8.0, step=1.0),
            dose_ug_ml=st.number_input("暴露剂量 (μg/mL)", min_value=0.0, value=50.0, step=5.0),
            exposure_time_h=st.number_input("暴露时间 (h)", min_value=1.0, value=24.0, step=1.0),
            material_type=st.selectbox(
                "材料类型",
                ["liposome", "polymer", "metal_oxide", "gold", "silver", "carbon", "zinc_oxide", "copper_oxide"],
            ),
            surface_modification=st.selectbox("表面修饰", ["PEG", "unmodified", "citrate"]),
            cell_type=st.selectbox("细胞类型", ["A549", "BEAS-2B", "HepG2", "THP-1", "NRK-52E"]),
            species=st.selectbox("来源物种", ["Human", "Rat", "Mouse"]),
            tissue=st.selectbox("组织来源", ["Lung", "Liver", "Blood", "Kidney"]),
            assay_method=st.selectbox("检测方法", ["MTT", "LDH"]),
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
