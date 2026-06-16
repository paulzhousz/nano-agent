from pathlib import Path
import html

import streamlit as st

from nano_tox_agent.explain import TOXICITY_TEXT, _clean_feature_name, build_explanation
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


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #F8FAFC;
            --surface: #FFFFFF;
            --surface-alt: #F1F5F9;
            --border: #D9E2EC;
            --text: #020617;
            --muted: #475569;
            --primary: #0F172A;
            --secondary: #334155;
            --accent: #0369A1;
            --low: #15803D;
            --medium: #B45309;
            --high: #B91C1C;
            --shadow: 0 14px 36px rgba(15, 23, 42, 0.08);
            --radius: 18px;
        }

        .stApp {
            background: linear-gradient(180deg, #FCFDFE 0%, var(--bg) 100%);
            color: var(--text);
        }

        .main .block-container {
            max-width: 1240px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero-card,
        .section-card,
        .summary-card,
        .empty-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
        }

        .hero-card {
            padding: 1.4rem 1.5rem 1.5rem;
            margin-bottom: 1.25rem;
        }

        .section-card {
            padding: 1.25rem;
        }

        .eyebrow {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--accent);
            font-weight: 700;
        }

        .hero-title {
            font-size: 2rem;
            line-height: 1.2;
            color: var(--primary);
            font-weight: 800;
            margin: 0.4rem 0;
        }

        .hero-copy,
        .section-copy,
        .muted-text {
            color: var(--muted);
            line-height: 1.7;
        }

        .summary-card,
        .empty-card,
        .detail-card {
            padding: 1.25rem;
            margin-bottom: 1rem;
        }

        .panel-title {
            color: var(--primary);
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .summary-text {
            color: var(--secondary);
            line-height: 1.75;
            margin-bottom: 0.9rem;
        }

        .feature-row {
            margin-bottom: 0.95rem;
        }

        .feature-meta {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            font-size: 0.95rem;
            color: var(--secondary);
        }

        .feature-bar {
            height: 10px;
            border-radius: 999px;
            background: #E2E8F0;
            overflow: hidden;
            margin-top: 0.35rem;
        }

        .feature-bar > span {
            display: block;
            height: 100%;
            background: linear-gradient(90deg, var(--accent) 0%, #38BDF8 100%);
        }

        @media (max-width: 960px) {
            .main .block-container {
                padding-top: 1.25rem;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            * {
                animation: none !important;
                transition: none !important;
                scroll-behavior: auto !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_summary_text(payload: dict[str, str], result: PredictionResult) -> str:
    if result.toxicity_level == "high":
        tendency = "样本呈现较强毒性信号，当前条件下不适合作为优先候选。"
    elif result.toxicity_level == "medium":
        tendency = "样本呈现中等毒性风险，建议继续做梯度验证后再判断是否推进。"
    else:
        tendency = "样本整体呈低毒性倾向，可作为后续体外验证的优先候选。"

    return (
        f"{tendency} 当前预测毒性等级为{payload['toxicity_label']}，"
        f"预测细胞活力约为 {payload['cell_viability_percent']}，"
        f"模型置信度为 {payload['confidence']}。"
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
          <div class="eyebrow">Academic Screening Interface</div>
          <div class="hero-title">纳米材料毒性评估与解释</div>
          <div class="hero-copy">
            面向肿瘤纳米药物筛选的科研工作台，突出研究结论、关键指标与解释依据，
            适合日常复核，也适合截图用于课堂、答辩和组会汇报。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-card">
          <div class="panel-title">研究结论</div>
          <div class="summary-text">
            右栏将在提交预测后切换为正式结果视图，先给出一句可直接用于汇报的结论摘要，
            再展开关键指标、解释与特征影响。
          </div>
          <div class="panel-title">当前页面可完成什么任务</div>
          <div class="summary-text">
            填写纳米材料属性、暴露条件和实验对象信息，快速完成一次科研筛选式毒性预测。
          </div>
          <div class="panel-title">预测后将输出哪些结果</div>
          <div class="summary-text">
            页面会输出毒性等级、预测细胞活力、模型置信度，以及“解释与依据”和“特征影响”两块结果内容。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_panel() -> tuple[PredictionInput, bool, bool]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 参数录入")
    st.markdown(
        '<div class="section-copy">按材料属性、暴露条件和实验对象填写参数后统一预测。</div>',
        unsafe_allow_html=True,
    )

    with st.form("prediction_form", clear_on_submit=False):
        st.markdown("#### 材料属性")
        particle_size_nm = st.number_input("粒径 (nm)", min_value=1.0, value=90.0, step=1.0)
        zeta_potential_mv = st.number_input("Zeta 电位 (mV)", value=-8.0, step=1.0)
        material_type = st.selectbox(
            "材料类型",
            ["liposome", "polymer", "metal_oxide", "gold", "silver", "carbon", "zinc_oxide", "copper_oxide"],
        )
        surface_modification = st.selectbox("表面修饰", ["PEG", "unmodified", "citrate"])

        st.markdown("#### 暴露条件")
        dose_ug_ml = st.number_input("暴露剂量 (μg/mL)", min_value=0.0, value=50.0, step=5.0)
        exposure_time_h = st.number_input("暴露时间 (h)", min_value=1.0, value=24.0, step=1.0)

        st.markdown("#### 实验对象")
        cell_type = st.selectbox("细胞类型", ["A549", "BEAS-2B", "HepG2", "THP-1", "NRK-52E"])
        species = st.selectbox("来源物种", ["Human", "Rat", "Mouse"])
        tissue = st.selectbox("组织来源", ["Lung", "Liver", "Blood", "Kidney"])
        assay_method = st.selectbox("检测方法", ["MTT", "LDH"])

        st.markdown("#### 解释增强")
        use_llm = st.checkbox("使用可选 LLM 改写解释", value=False)
        submitted = st.form_submit_button("预测毒性", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    sample = PredictionInput(
        particle_size_nm=particle_size_nm,
        zeta_potential_mv=zeta_potential_mv,
        dose_ug_ml=dose_ug_ml,
        exposure_time_h=exposure_time_h,
        material_type=material_type,
        surface_modification=surface_modification,
        cell_type=cell_type,
        species=species,
        tissue=tissue,
        assay_method=assay_method,
    )
    return sample, use_llm, submitted


def render_feature_bars(top_features: list[tuple[str, float]]) -> None:
    st.markdown('<div class="detail-card">', unsafe_allow_html=True)
    st.markdown("### 特征影响")

    if not top_features:
        st.markdown("暂无可展示的主要影响因素。")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    max_weight = max(abs(weight) for _, weight in top_features) or 1.0
    for feature_name, weight in top_features:
        width = max(abs(weight) / max_weight * 100.0, 8.0)
        label = html.escape(_clean_feature_name(feature_name))
        st.markdown(
            f"""
            <div class="feature-row">
              <div class="feature-meta">
                <span>{label}</span>
                <span>{weight:.3f}</span>
              </div>
              <div class="feature-bar"><span style="width: {width:.1f}%;"></span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_result_panel(payload: dict[str, str], result: PredictionResult) -> None:
    st.markdown('<div class="summary-card">', unsafe_allow_html=True)
    st.markdown("### 研究结论")
    st.markdown(
        f'<div class="summary-text">{build_summary_text(payload, result)}</div>',
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(3)
    metric_columns[0].metric("毒性等级", payload["toxicity_label"])
    metric_columns[1].metric("预测细胞活力", payload["cell_viability_percent"])
    metric_columns[2].metric("模型置信度", payload["confidence"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="detail-card">', unsafe_allow_html=True)
    st.markdown("### 解释与依据")
    st.markdown(payload["explanation"])
    st.markdown("</div>", unsafe_allow_html=True)

    render_feature_bars(result.top_features)


def main() -> None:
    st.set_page_config(page_title="纳米毒性预测智能体", layout="wide")
    inject_styles()
    render_hero()

    left_col, right_col = st.columns([0.95, 1.25], gap="large")
    with left_col:
        sample, use_llm, submitted = render_input_panel()
    with right_col:
        if not submitted:
            render_empty_state()
            return

        try:
            bundle = load_or_train_bundle(MODEL_PATH, DATA_PATH)
            literature = select_literature(load_literature(LITERATURE_PATH), "feature_explanation")
            result = predict_toxicity(bundle, sample)
            explanation = build_explanation(sample, result, literature)
            if use_llm:
                explanation = generate_llm_response(explanation) or explanation
            payload = build_result_payload(result, explanation)
            render_result_panel(payload, result)
        except Exception as error:
            st.error(format_user_error(error))


if __name__ == "__main__":
    main()
