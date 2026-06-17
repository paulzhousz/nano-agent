"""Streamlit 单页应用入口。

页面负责承接参数录入、模型加载、预测推理、解释生成与结果展示，
把后端产出的结构化结果组织成适合研究人员阅读的工作台。
"""

from pathlib import Path
import html
import re
from textwrap import dedent
from urllib.parse import quote

import streamlit as st

from nano_tox_agent.explain import TOXICITY_TEXT, build_explanation, clean_feature_name
from nano_tox_agent.literature import load_literature, select_literature
from nano_tox_agent.llm_layer import generate_llm_response
from nano_tox_agent.predict import load_or_train_bundle, predict_toxicity
from nano_tox_agent.schema import PredictionInput, PredictionResult

DATA_PATH = Path("data/processed/toxicity_clean.csv")
MODEL_PATH = Path("models/toxicity_bundle.joblib")
LITERATURE_PATH = Path("literature/literature_base.json")
HERO_BACKGROUND_SVG = quote("""
    <svg width="1600" height="640" viewBox="0 0 1600 640" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="1600" height="640" fill="#F6FAFF"/>
      <g opacity="0.9">
        <circle cx="1240" cy="188" r="170" fill="url(#haloA)"/>
        <circle cx="1385" cy="344" r="124" fill="url(#haloB)"/>
        <circle cx="1110" cy="372" r="90" fill="url(#haloC)"/>
      </g>
      <g opacity="0.72">
        <path d="M905 152C981 152 1048 210 1077 296C1105 380 1170 440 1252 454" stroke="#0B3C91" stroke-width="2.2"/>
        <path d="M952 122C1024 122 1089 174 1127 250C1164 326 1238 389 1328 404" stroke="#61A6FF" stroke-width="1.8"/>
        <path d="M878 212C952 214 1019 272 1048 356C1069 416 1126 474 1196 503" stroke="#8ABEFF" stroke-width="1.4"/>
      </g>
      <g opacity="0.42">
        <path d="M976 118L1016 94L1056 118L1056 164L1016 188L976 164V118Z" stroke="#0B3C91" stroke-width="1.2"/>
        <path d="M1080 206L1112 188L1144 206L1144 244L1112 262L1080 244V206Z" stroke="#005BBB" stroke-width="1.2"/>
        <path d="M1180 280L1210 262L1240 280L1240 314L1210 332L1180 314V280Z" stroke="#61A6FF" stroke-width="1.2"/>
        <path d="M1272 176L1302 158L1332 176L1332 210L1302 228L1272 210V176Z" stroke="#61A6FF" stroke-width="1.2"/>
      </g>
      <g opacity="0.2">
        <path d="M860 104H1510" stroke="#005BBB" stroke-width="1"/>
        <path d="M860 142H1510" stroke="#005BBB" stroke-width="1"/>
        <path d="M860 180H1510" stroke="#005BBB" stroke-width="1"/>
        <path d="M860 218H1510" stroke="#005BBB" stroke-width="1"/>
        <path d="M860 256H1510" stroke="#005BBB" stroke-width="1"/>
        <path d="M860 294H1510" stroke="#005BBB" stroke-width="1"/>
        <path d="M860 332H1510" stroke="#005BBB" stroke-width="1"/>
        <path d="M860 370H1510" stroke="#005BBB" stroke-width="1"/>
        <path d="M860 408H1510" stroke="#005BBB" stroke-width="1"/>
        <path d="M860 446H1510" stroke="#005BBB" stroke-width="1"/>
      </g>
      <defs>
        <radialGradient id="haloA" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(1240 188) rotate(90) scale(170)">
          <stop stop-color="#C4DEFF"/>
          <stop offset="1" stop-color="#C4DEFF" stop-opacity="0"/>
        </radialGradient>
        <radialGradient id="haloB" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(1385 344) rotate(90) scale(124)">
          <stop stop-color="#A8CCFF"/>
          <stop offset="1" stop-color="#A8CCFF" stop-opacity="0"/>
        </radialGradient>
        <radialGradient id="haloC" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(1110 372) rotate(90) scale(90)">
          <stop stop-color="#DCEBFF"/>
          <stop offset="1" stop-color="#DCEBFF" stop-opacity="0"/>
        </radialGradient>
      </defs>
    </svg>
    """)
ARCHITECTURE_DIAGRAM_SVG = quote("""
    <svg width="860" height="1040" viewBox="0 0 860 1040" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="860" height="1040" rx="28" fill="#F8FBFF"/>
      <rect x="18" y="18" width="824" height="1004" rx="24" stroke="#D4E2F2" stroke-width="2"/>
      <text x="42" y="72" fill="#0F172A" font-family="Arial, Helvetica, sans-serif" font-size="30" font-weight="700">功能架构图</text>
      <text x="42" y="106" fill="#475569" font-family="Arial, Helvetica, sans-serif" font-size="17">围绕数据准备、模型训练、预测解释与结果输出的完整功能链路</text>

      <rect x="34" y="136" width="368" height="184" rx="28" fill="#F5F9FF" stroke="#D7E5F5" stroke-width="2"/>
      <rect x="458" y="136" width="368" height="406" rx="28" fill="#F3FAFF" stroke="#D4E7F7" stroke-width="2"/>
      <rect x="34" y="324" width="368" height="338" rx="28" fill="#F4F7FD" stroke="#D7E4F1" stroke-width="2"/>
      <rect x="34" y="674" width="792" height="338" rx="28" fill="#F7FAF4" stroke="#DCE7D5" stroke-width="2"/>

      <rect x="42" y="126" width="136" height="28" rx="14" fill="#2F69BF"/>
      <text x="58" y="145" fill="#FFFFFF" font-family="Arial, Helvetica, sans-serif" font-size="12" font-weight="700">阶段一  数据准备</text>
      <rect x="466" y="126" width="136" height="28" rx="14" fill="#0E7C86"/>
      <text x="482" y="145" fill="#FFFFFF" font-family="Arial, Helvetica, sans-serif" font-size="12" font-weight="700">阶段三  在线预测</text>
      <rect x="42" y="340" width="136" height="28" rx="14" fill="#6A5ACD"/>
      <text x="58" y="359" fill="#FFFFFF" font-family="Arial, Helvetica, sans-serif" font-size="12" font-weight="700">阶段二  离线建模</text>
      <rect x="42" y="664" width="152" height="28" rx="14" fill="#6A8E51"/>
      <text x="58" y="683" fill="#FFFFFF" font-family="Arial, Helvetica, sans-serif" font-size="12" font-weight="700">阶段四  解释与交付</text>

      <rect x="42" y="170" width="352" height="132" rx="24" fill="#FFFFFF" stroke="#C9D9EC" stroke-width="2"/>
      <text x="68" y="182" fill="#0B3C91" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700">数据准备</text>
      <text x="68" y="216" fill="#0F172A" font-family="Arial, Helvetica, sans-serif" font-size="16">公开数据与文献附表进入统一数据池</text>
      <text x="68" y="242" fill="#0F172A" font-family="Arial, Helvetica, sans-serif" font-size="16">完成字段筛选、单位统一、缺失值处理</text>
      <text x="68" y="268" fill="#0F172A" font-family="Arial, Helvetica, sans-serif" font-size="16">生成可训练样本与统一 schema</text>

      <rect x="466" y="170" width="352" height="132" rx="24" fill="#FFFFFF" stroke="#C9D9EC" stroke-width="2"/>
      <text x="492" y="182" fill="#0B3C91" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700">用户输入样本</text>
      <text x="492" y="216" fill="#0F172A" font-family="Arial, Helvetica, sans-serif" font-size="16">录入粒径、电位、剂量、暴露时间</text>
      <text x="492" y="242" fill="#0F172A" font-family="Arial, Helvetica, sans-serif" font-size="16">补充材料类型、细胞类型、检测方法</text>
      <text x="492" y="268" fill="#0F172A" font-family="Arial, Helvetica, sans-serif" font-size="16">形成单条待预测样本</text>

      <rect x="68" y="402" width="300" height="92" rx="22" fill="#EEF4FB" stroke="#BCD1E7" stroke-width="2"/>
      <text x="92" y="426" fill="#0B3C91" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700">毒性分类训练</text>
      <text x="92" y="454" fill="#0F172A" font-family="Arial, Helvetica, sans-serif" font-size="15">基于统一样本训练分类器</text>
      <text x="92" y="478" fill="#0F172A" font-family="Arial, Helvetica, sans-serif" font-size="15">学习高 / 中 / 低毒性等级边界</text>

      <rect x="68" y="518" width="300" height="92" rx="22" fill="#EEF4FB" stroke="#BCD1E7" stroke-width="2"/>
      <text x="92" y="544" fill="#0B3C91" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700">细胞活力回归训练</text>
      <text x="92" y="572" fill="#0F172A" font-family="Arial, Helvetica, sans-serif" font-size="15">训练回归器预测细胞活力百分比</text>
      <text x="92" y="596" fill="#0F172A" font-family="Arial, Helvetica, sans-serif" font-size="15">两类模型统一封装为 bundle 文件</text>

      <rect x="466" y="364" width="352" height="152" rx="24" fill="#DCF4F5" stroke="#78C7CD" stroke-width="3"/>
      <rect x="466" y="364" width="352" height="14" rx="24" fill="#0E7C86"/>
      <text x="492" y="398" fill="#0B3C91" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700">预测推理</text>
      <text x="682" y="398" fill="#0E7C86" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="700">核心推理节点</text>
      <text x="492" y="430" fill="#0F172A" font-family="Arial, Helvetica, sans-serif" font-size="16">加载模型文件并完成特征编码</text>
      <text x="492" y="456" fill="#0F172A" font-family="Arial, Helvetica, sans-serif" font-size="16">融合用户输入与训练得到的模型参数</text>
      <text x="492" y="482" fill="#0F172A" font-family="Arial, Helvetica, sans-serif" font-size="16">输出毒性等级、细胞活力、模型置信度</text>

      <rect x="42" y="714" width="352" height="132" rx="24" fill="#FFFFFF" stroke="#C9D9EC" stroke-width="2"/>
      <text x="68" y="738" fill="#0B3C91" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700">解释生成</text>
      <text x="68" y="770" fill="#0F172A" font-family="Arial, Helvetica, sans-serif" font-size="16">按特征权重整理主要影响因素</text>
      <text x="68" y="796" fill="#0F172A" font-family="Arial, Helvetica, sans-serif" font-size="16">拼接规则解释、建议与文献摘要</text>
      <text x="68" y="822" fill="#0F172A" font-family="Arial, Helvetica, sans-serif" font-size="16">形成可直接展示的模板解释</text>

      <rect x="466" y="714" width="352" height="64" rx="20" fill="#FFFFFF" stroke="#C9D9EC" stroke-width="2"/>
      <text x="492" y="728" fill="#0B3C91" font-family="Arial, Helvetica, sans-serif" font-size="17" font-weight="700">可选 LLM 改写</text>
      <text x="492" y="752" fill="#0F172A" font-family="Arial, Helvetica, sans-serif" font-size="15">仅改写解释文本，不参与模型预测</text>

      <rect x="466" y="790" width="352" height="64" rx="20" fill="#F7F1E3" stroke="#E4D4B5" stroke-width="2"/>
      <text x="492" y="808" fill="#B45309" font-family="Arial, Helvetica, sans-serif" font-size="17" font-weight="700">模板回退</text>
      <text x="492" y="832" fill="#0F172A" font-family="Arial, Helvetica, sans-serif" font-size="15">无可用配置或调用失败时返回原始解释</text>

      <rect x="42" y="878" width="220" height="108" rx="22" fill="#FFFFFF" stroke="#C9D9EC" stroke-width="2"/>
      <text x="66" y="912" fill="#0B3C91" font-family="Arial, Helvetica, sans-serif" font-size="17" font-weight="700">文献摘要库</text>
      <text x="66" y="940" fill="#475569" font-family="Arial, Helvetica, sans-serif" font-size="15">提供数据来源摘要</text>
      <text x="66" y="964" fill="#475569" font-family="Arial, Helvetica, sans-serif" font-size="15">提供方法依据与解释支撑</text>

      <rect x="290" y="878" width="528" height="120" rx="24" fill="#FFFFFF" stroke="#C9D9EC" stroke-width="2"/>
      <text x="318" y="912" fill="#0B3C91" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700">最终输出</text>
      <text x="318" y="942" fill="#0F172A" font-family="Arial, Helvetica, sans-serif" font-size="15">研究结论、毒性等级、预测细胞活力、模型置信度</text>
      <text x="318" y="966" fill="#0F172A" font-family="Arial, Helvetica, sans-serif" font-size="15">样本信息、特征影响、解释与依据、文献链接与下一步建议</text>

      <path d="M218 298V332" stroke="#0B3C91" stroke-width="4" stroke-linecap="round"/>
      <path d="M642 298V356" stroke="#0B3C91" stroke-width="4" stroke-linecap="round"/>
      <path d="M394 440H466" stroke="#0B3C91" stroke-width="4" stroke-linecap="round"/>
      <path d="M394 558H430V470H466" stroke="#0B3C91" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M642 516V650H218V700" stroke="#0B3C91" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M394 770H466" stroke="#0B3C91" stroke-width="4" stroke-linecap="round"/>
      <path d="M642 846V862H554V878" stroke="#0B3C91" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M262 932H290" stroke="#0B3C91" stroke-width="4" stroke-linecap="round"/>

      <circle cx="218" cy="332" r="6" fill="#0B3C91"/>
      <circle cx="642" cy="356" r="6" fill="#0B3C91"/>
      <circle cx="466" cy="440" r="6" fill="#0B3C91"/>
      <circle cx="466" cy="470" r="6" fill="#0B3C91"/>
      <circle cx="218" cy="700" r="6" fill="#0B3C91"/>
      <circle cx="466" cy="770" r="6" fill="#0B3C91"/>
      <circle cx="554" cy="878" r="6" fill="#0B3C91"/>
      <circle cx="290" cy="932" r="6" fill="#0B3C91"/>
    </svg>
    """)


def build_result_payload(result: PredictionResult, explanation: str) -> dict[str, str]:
    """把模型输出整理成前端展示更方便的字符串载荷。"""
    return {
        "toxicity_label": TOXICITY_TEXT.get(
            result.toxicity_level, result.toxicity_level
        ),
        "cell_viability_percent": f"{result.cell_viability_percent}%",
        "confidence": f"{result.confidence}",
        "explanation": explanation,
    }


def format_user_error(error: Exception) -> str:
    """把底层异常翻译成用户可理解的报错信息。"""
    if isinstance(error, FileNotFoundError):
        return f"数据或模型文件缺失：{error}"
    if isinstance(error, ValueError):
        return f"输入或数据格式错误：{error}"
    return f"预测失败：{error}"


def inject_styles() -> None:
    """集中注入单页应用使用的 CSS 主题与布局样式。"""
    styles = """
        <style>
        :root {
            --bg: #F4F7FB;
            --surface: #FFFFFF;
            --surface-alt: #EEF4FB;
            --border: #D3DEEA;
            --text: #020617;
            --muted: #475569;
            --primary: #0F172A;
            --secondary: #334155;
            --accent: #005BBB;
            --accent-strong: #0B3C91;
            --accent-soft: #DCEBFF;
            --low: #15803D;
            --medium: #B45309;
            --high: #B91C1C;
            --shadow: 0 18px 48px rgba(15, 23, 42, 0.08);
            --radius: 22px;
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(0, 91, 187, 0.12), transparent 24rem),
                radial-gradient(circle at left 20%, rgba(11, 60, 145, 0.08), transparent 30rem),
                linear-gradient(180deg, #FBFDFF 0%, var(--bg) 100%);
            color: var(--text);
        }

        .main .block-container {
            max-width: 1240px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero-card,
        .summary-card,
        .empty-card,
        div[data-testid="stForm"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
        }

        .hero-card {
            padding: 2rem 2.1rem 2.05rem;
            margin-bottom: 1.5rem;
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(242, 247, 255, 0.96) 100%);
            background-image:
                url("__HERO_BACKGROUND__"),
                linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(242, 247, 255, 0.96) 100%);
            background-position: right center, center;
            background-repeat: no-repeat, no-repeat;
            background-size: 58% auto, cover;
            position: relative;
            overflow: hidden;
        }

        .hero-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 6px;
            background: linear-gradient(90deg, var(--accent-strong) 0%, var(--accent) 55%, #61A6FF 100%);
        }

        .hero-card::after {
            content: "";
            position: absolute;
            top: 0;
            right: 0;
            width: 260px;
            height: 260px;
            background: radial-gradient(circle, rgba(0, 91, 187, 0.14) 0%, transparent 70%);
            pointer-events: none;
        }

        .eyebrow {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            color: var(--accent-strong);
            font-weight: 700;
        }

        .hero-title {
            font-size: 2.5rem;
            line-height: 1.08;
            color: var(--primary);
            font-weight: 900;
            margin: 0.55rem 0 0.8rem;
            max-width: 16ch;
            white-space: nowrap;
        }

        .hero-copy,
        .section-copy,
        .muted-text {
            color: var(--muted);
            line-height: 1.8;
            font-size: 1rem;
        }

        .hero-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 1.2rem;
        }

        .hero-chip {
            border: 1px solid #BCD0EA;
            background: rgba(255, 255, 255, 0.82);
            color: var(--secondary);
            border-radius: 999px;
            padding: 0.42rem 0.86rem;
            font-size: 0.84rem;
            font-weight: 600;
        }

        .summary-card,
        .empty-card,
        .detail-card {
            padding: 1.15rem;
            margin-bottom: 0.82rem;
        }

        .detail-card {
            background: #FFFFFF;
            border: 1px solid #D9E2EC;
            border-radius: 18px;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
        }

        .workspace-card-title {
            font-size: 1.95rem;
            line-height: 1.04;
            font-weight: 900;
            color: var(--primary);
            margin: 0.28rem 0 0.62rem;
        }

        .workspace-kicker {
            color: var(--accent-strong);
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.76rem;
            font-weight: 700;
        }

        div[data-testid="stForm"] {
            padding: 1.55rem 1.3rem 1.35rem;
            position: relative;
            overflow: hidden;
        }

        div[data-testid="stForm"]::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 5px;
            background: linear-gradient(90deg, var(--accent-strong), #67A8FF);
        }

        .input-group {
            border-top: 1px solid #D8E3F0;
            padding-top: 1rem;
            margin-top: 1rem;
        }

        .input-group:first-child {
            border-top: none;
            margin-top: 0;
            padding-top: 0;
        }

        div[data-testid="stForm"] h4 {
            margin-top: 0.35rem;
            font-size: 1.75rem;
            color: var(--primary);
            letter-spacing: -0.03em;
        }

        div[data-testid="stForm"] label,
        div[data-testid="stForm"] .stCheckbox p {
            color: var(--secondary);
            font-weight: 500;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input {
            border-radius: 16px !important;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {
            background: #F7FAFF !important;
            border: 1px solid #D8E3F0 !important;
            box-shadow: none !important;
        }

        div[data-baseweb="input"] input,
        div[data-baseweb="select"] input {
            color: var(--primary) !important;
            font-weight: 600;
        }

        div[data-testid="stForm"] button[kind="secondaryFormSubmit"] {
            min-height: 3.4rem;
            border-radius: 18px;
            border: 1px solid rgba(11, 60, 145, 0.15);
            background: #005BBB;
            color: #FFFFFF;
            font-weight: 800;
            font-size: 1.08rem;
            letter-spacing: 0.04em;
            box-shadow: 0 16px 30px rgba(0, 91, 187, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.18);
            transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
        }

        div[data-testid="stForm"] button[kind="secondaryFormSubmit"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 20px 34px rgba(0, 91, 187, 0.34), inset 0 1px 0 rgba(255, 255, 255, 0.22);
            filter: saturate(1.06);
        }

        div[data-testid="stForm"] button[kind="secondaryFormSubmit"]:focus,
        div[data-testid="stForm"] button[kind="secondaryFormSubmit"]:focus-visible {
            outline: none;
            box-shadow:
                0 0 0 4px rgba(97, 166, 255, 0.28),
                0 20px 34px rgba(0, 91, 187, 0.34),
                inset 0 1px 0 rgba(255, 255, 255, 0.22);
        }

        div[data-testid="stForm"] button[kind="secondaryFormSubmit"] p {
            font-weight: 800 !important;
            color: #FFFFFF !important;
            font-size: 1.08rem !important;
        }

        .panel-title {
            color: var(--primary);
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.32rem;
        }

        .section-strong-label {
            color: var(--primary);
            font-size: 1.08rem;
            font-weight: 800;
            margin: 0.2rem 0 0.4rem;
        }

        .summary-text {
            color: var(--secondary);
            line-height: 1.68;
            margin-bottom: 0.72rem;
        }

        .summary-card {
            background:
                linear-gradient(180deg, rgba(255,255,255,1) 0%, rgba(245,249,255,0.98) 100%);
            position: relative;
            overflow: hidden;
        }

        .summary-card::before,
        .detail-card::before {
            content: "";
            display: block;
            width: 4.25rem;
            height: 4px;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--accent-strong), var(--accent));
            margin-bottom: 0.78rem;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.7rem;
            margin: 0.05rem 0 0.78rem;
        }

        .metric-card {
            border: 1px solid #D9E5F2;
            border-radius: 18px;
            background: linear-gradient(180deg, #FFFFFF 0%, #F6FAFF 100%);
            padding: 0.82rem 0.92rem 0.88rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05);
        }

        .metric-label {
            font-size: 0.82rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.42rem;
        }

        .metric-value {
            font-size: 1.72rem;
            line-height: 1;
            font-weight: 900;
            color: var(--primary);
        }

        .metric-value.low { color: var(--low); }
        .metric-value.medium { color: var(--medium); }
        .metric-value.high { color: var(--high); }

        .input-summary-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.72rem;
            margin-top: 0.2rem;
        }

        .input-summary-item {
            background: var(--surface-alt);
            border: 1px solid #D7E3F0;
            border-radius: 16px;
            padding: 0.76rem 0.88rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.8rem;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7);
        }

        .input-summary-group {
            margin-top: 0.9rem;
        }

        .input-summary-group:first-of-type {
            margin-top: 0.25rem;
        }

        .input-summary-group-title {
            color: var(--primary);
            font-size: 0.98rem;
            font-weight: 800;
            margin-bottom: 0.52rem;
        }

        .input-summary-label {
            font-size: 0.86rem;
            color: var(--secondary);
            font-weight: 700;
            white-space: nowrap;
        }

        .input-summary-value {
            font-size: 0.94rem;
            font-weight: 800;
            color: var(--primary);
            text-align: right;
            word-break: break-word;
        }

        .explanation-line {
            padding: 0.48rem 0 0.54rem;
            border-top: 1px solid #E2EAF4;
            color: var(--secondary);
            line-height: 1.62;
        }

        .explanation-line:first-of-type {
            border-top: none;
            padding-top: 0;
        }

        .explanation-line strong {
            color: var(--primary);
        }

        .literature-item {
            padding: 0.6rem 0 0.68rem;
            border-top: 1px solid #E2EAF4;
            color: var(--secondary);
            line-height: 1.62;
        }

        .literature-link {
            color: var(--accent-strong);
            font-weight: 700;
            text-decoration: none;
            border-bottom: 1px solid rgba(11, 60, 145, 0.25);
        }

        .literature-link:hover {
            color: var(--accent);
            border-bottom-color: var(--accent);
        }

        .feature-row {
            margin-bottom: 0.82rem;
        }

        .feature-meta {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            font-size: 0.95rem;
            color: var(--secondary);
        }

        .feature-meta span:first-child {
            font-weight: 600;
            color: var(--primary);
        }

        .feature-bar {
            height: 10px;
            border-radius: 999px;
            background: #DFE7F2;
            overflow: hidden;
            margin-top: 0.32rem;
        }

        .feature-bar > span {
            display: block;
            height: 100%;
            background: linear-gradient(90deg, var(--accent-strong) 0%, #2D8CFF 100%);
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.15);
        }

        .loading-card {
            position: relative;
            overflow: hidden;
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 246, 255, 0.98) 100%);
        }

        .loading-card::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(
                110deg,
                transparent 0%,
                rgba(255, 255, 255, 0.35) 42%,
                rgba(137, 182, 255, 0.18) 50%,
                transparent 58%
            );
            transform: translateX(-120%);
            animation: loading-sheen 2.2s ease-in-out infinite;
            pointer-events: none;
        }

        .loading-title {
            font-size: 1.35rem;
            font-weight: 900;
            color: var(--primary);
            margin: 0.2rem 0 0.42rem;
        }

        .loading-copy {
            color: var(--secondary);
            line-height: 1.72;
            max-width: 42rem;
            margin-bottom: 0.9rem;
        }

        .loading-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-bottom: 1rem;
        }

        .loading-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.48rem 0.76rem;
            border-radius: 999px;
            border: 1px solid #CFE0F4;
            background: rgba(255, 255, 255, 0.82);
            color: var(--secondary);
            font-size: 0.84rem;
            font-weight: 700;
        }

        .loading-dot {
            width: 0.48rem;
            height: 0.48rem;
            border-radius: 999px;
            background: var(--accent);
            animation: loading-pulse 1.2s ease-in-out infinite;
        }

        .loading-pill:nth-child(2) .loading-dot { animation-delay: 0.16s; }
        .loading-pill:nth-child(3) .loading-dot { animation-delay: 0.32s; }

        .loading-bar {
            position: relative;
            height: 0.68rem;
            border-radius: 999px;
            background: rgba(188, 208, 234, 0.45);
            overflow: hidden;
        }

        .loading-bar > span {
            display: block;
            width: 38%;
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--accent-strong) 0%, #5EA6FF 100%);
            animation: loading-progress 1.5s ease-in-out infinite;
        }

        @keyframes loading-sheen {
            0% { transform: translateX(-120%); }
            100% { transform: translateX(120%); }
        }

        @keyframes loading-pulse {
            0%, 100% { transform: scale(0.85); opacity: 0.55; }
            50% { transform: scale(1.1); opacity: 1; }
        }

        @keyframes loading-progress {
            0% { transform: translateX(-120%); }
            50% { transform: translateX(85%); }
            100% { transform: translateX(210%); }
        }

        @media (max-width: 960px) {
            .main .block-container {
                padding-top: 1.25rem;
            }

            .hero-title,
            .workspace-card-title {
                font-size: 2rem;
            }

            .hero-title {
                max-width: none;
                white-space: normal;
            }

            .metric-grid,
            .input-summary-grid {
                grid-template-columns: 1fr;
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
        """
    styles = styles.replace(
        "__HERO_BACKGROUND__", f"data:image/svg+xml;utf8,{HERO_BACKGROUND_SVG}"
    )
    st.markdown(
        styles,
        unsafe_allow_html=True,
    )


def build_summary_text(payload: dict[str, str], result: PredictionResult) -> str:
    """按毒性等级生成一句可直接展示的结论摘要。"""
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
    """渲染页面顶部的品牌区与能力说明。"""
    st.markdown(
        """
        <div class="hero-card">
          <div class="eyebrow">Nano Toxicity Agent</div>
          <div class="hero-title">纳米材料毒性预测智能体</div>
          <div class="hero-copy">
            面向肿瘤纳米药物筛选与临床前研究判断的单页工作台，
            以研究结论、关键指标和解释依据为主轴。
          </div>
          <div class="hero-meta">
            <span class="hero-chip">输出毒性等级</span>
            <span class="hero-chip">细胞活力预测</span>
            <span class="hero-chip">解释与文献依据</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    """在用户尚未提交参数时展示默认结果工作区。"""
    st.markdown(
        f"""
        <div class="workspace-kicker">Result Workspace</div>
        <div class="workspace-card-title">等待生成研究结论</div>
        <div class="section-copy">提交预测后，右侧会切换成正式结果工作区。</div>
        <div class="empty-card">
          <div class="panel-title">工作区说明</div>
          <div class="summary-text">
            右侧区域用于承接预测后的主要输出，内容会按“研究结论、关键指标、解释与依据、特征影响”的顺序展开，
            便于在研究复核直接阅读。
          </div>
          <div style="margin-top: 1.5rem;">
            <img
              src="data:image/svg+xml;utf8,{ARCHITECTURE_DIAGRAM_SVG}"
              alt="功能架构图"
              style="width: 100%; display: block; border-radius: 20px; border: 1px solid #D7E2EE;"
            />
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_panel() -> tuple[PredictionInput, bool, bool]:
    """渲染左侧录入表单，并返回样本、LLM 开关和提交状态。"""
    st.markdown(
        """
        <div class="workspace-kicker">Parameter Workspace</div>
        <div class="workspace-card-title">参数录入</div>
        <div class="section-copy">按材料属性、暴露条件和实验对象填写参数后统一预测。</div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("prediction_form", clear_on_submit=False):
        st.markdown("#### 材料属性")
        material_col_left, material_col_right = st.columns(2, gap="medium")
        with material_col_left:
            particle_size_nm = st.number_input(
                "粒径 (nm)", min_value=1.0, value=90.0, step=1.0
            )
        with material_col_right:
            zeta_potential_mv = st.number_input("Zeta 电位 (mV)", value=-8.0, step=1.0)

        material_meta_col_left, material_meta_col_right = st.columns(2, gap="medium")
        with material_meta_col_left:
            material_type = st.selectbox(
                "材料类型",
                [
                    "liposome",
                    "polymer",
                    "metal_oxide",
                    "gold",
                    "silver",
                    "carbon",
                    "zinc_oxide",
                    "copper_oxide",
                ],
            )
        with material_meta_col_right:
            surface_modification = st.selectbox(
                "表面修饰", ["PEG", "unmodified", "citrate"]
            )

        st.markdown("#### 暴露条件")
        exposure_col_left, exposure_col_right = st.columns(2, gap="medium")
        with exposure_col_left:
            dose_ug_ml = st.number_input(
                "暴露剂量 (μg/mL)", min_value=0.0, value=50.0, step=5.0
            )
        with exposure_col_right:
            exposure_time_h = st.number_input(
                "暴露时间 (h)", min_value=1.0, value=24.0, step=1.0
            )

        st.markdown("#### 实验对象")
        experiment_col_left, experiment_col_right = st.columns(2, gap="medium")
        with experiment_col_left:
            cell_type = st.selectbox(
                "细胞类型", ["A549", "BEAS-2B", "HepG2", "THP-1", "NRK-52E"]
            )
        with experiment_col_right:
            species = st.selectbox("来源物种", ["Human", "Rat", "Mouse"])

        experiment_meta_col_left, experiment_meta_col_right = st.columns(
            2, gap="medium"
        )
        with experiment_meta_col_left:
            tissue = st.selectbox("组织来源", ["Lung", "Liver", "Blood", "Kidney"])
        with experiment_meta_col_right:
            assay_method = st.selectbox("检测方法", ["MTT", "LDH"])

        st.markdown("#### 解释增强")
        use_llm = st.checkbox("使用可选 LLM 改写解释", value=False)
        # st.markdown(
        #     '<div class="muted-text" style="margin-top: 0.75rem;">填写完成后统一提交，结果将在右侧工作区呈现。</div>',
        #     unsafe_allow_html=True,
        # )
        submitted = st.form_submit_button("预测", use_container_width=True)

    # 表单字段统一封装为 PredictionInput，便于直接进入推理流程。
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
    """把主要特征重要性渲染成条形卡片。"""
    if not top_features:
        st.markdown(
            dedent("""
            <div class="detail-card">
              <div class="workspace-kicker">Feature Signals</div>
              <div class="workspace-card-title">特征影响</div>
              <div class="summary-text">暂无可展示的主要影响因素。</div>
            </div>
            """),
            unsafe_allow_html=True,
        )
        return

    rows: list[str] = []
    max_weight = max(abs(weight) for _, weight in top_features) or 1.0
    for feature_name, weight in top_features:
        width = max(abs(weight) / max_weight * 100.0, 8.0)
        label = html.escape(clean_feature_name(feature_name))
        rows.append(dedent(f"""
                <div class="feature-row">
                  <div class="feature-meta">
                    <span>{label}</span>
                    <span>{weight:.3f}</span>
                  </div>
                  <div class="feature-bar"><span style="width: {width:.1f}%;"></span></div>
                </div>
                """).strip())
    st.markdown(
        dedent(f"""
        <div class="detail-card">
          <div class="workspace-kicker">Feature Signals</div>
          <div class="workspace-card-title">特征影响</div>
          {''.join(rows)}
        </div>
        """),
        unsafe_allow_html=True,
    )


def render_input_summary(sample: PredictionInput) -> None:
    """把用户输入按材料/暴露/实验对象分组回显。"""
    grouped_items = [
        (
            "材料属性",
            [
                ("粒径", f"{sample.particle_size_nm:.0f} nm"),
                ("Zeta 电位", f"{sample.zeta_potential_mv:.0f} mV"),
                ("材料类型", sample.material_type),
                ("表面修饰", sample.surface_modification),
            ],
        ),
        (
            "暴露条件",
            [
                ("暴露剂量", f"{sample.dose_ug_ml:.0f} μg/mL"),
                ("暴露时间", f"{sample.exposure_time_h:.0f} h"),
            ],
        ),
        (
            "实验对象",
            [
                ("细胞类型", sample.cell_type),
                ("来源物种", sample.species),
                ("组织来源", sample.tissue),
                ("检测方法", sample.assay_method),
            ],
        ),
    ]
    groups_markup = "".join(dedent(f"""
            <div class="input-summary-group">
              <div class="input-summary-group-title">{html.escape(group_title)}</div>
              <div class="input-summary-grid">
                {''.join(
                    dedent(
                        f'''
                        <div class="input-summary-item">
                          <div class="input-summary-label">{html.escape(label)}</div>
                          <div class="input-summary-value">{html.escape(value)}</div>
                        </div>
                        '''
                    ).strip()
                    for label, value in items
                )}
              </div>
            </div>
            """).strip() for group_title, items in grouped_items)
    st.markdown(
        dedent(f"""
        <div class="detail-card">
          <div class="workspace-kicker">Sample Snapshot</div>
          <div class="panel-title">样本信息</div>
          {groups_markup}
        </div>
        """),
        unsafe_allow_html=True,
    )


def normalize_explanation_line(line: str) -> str:
    """清理 Markdown 痕迹，便于把解释文本重新拆段展示。"""
    normalized = line.strip()
    normalized = re.sub(r"^#{1,6}\s*", "", normalized)
    normalized = re.sub(r"^\s*[-*]\s+", "", normalized)
    normalized = normalized.replace("**", "").replace("__", "")
    return normalized.strip()


def extract_explanation_sections(explanation: str) -> dict[str, list[str]]:
    """从模板解释或 LLM 改写结果中提取解释、建议、文献依据三段。"""
    sections = {"解释": [], "建议": [], "文献依据": []}
    current_section: str | None = None
    for raw_line in explanation.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        normalized = normalize_explanation_line(line)
        heading = normalized.rstrip("：:")
        if heading in sections:
            current_section = heading
            continue
        if normalized == "---":
            continue
        if current_section in sections:
            sections[current_section].append(normalized)
    return sections


def render_explanation_sections(
    explanation: str, literature_entries: list[dict[str, object]]
) -> None:
    """把解释文本和文献条目渲染成结构化说明卡片。"""
    sections = extract_explanation_sections(explanation)
    explanation_body = (
        "".join(
            f'<div class="explanation-line">{html.escape(line)}</div>'
            for line in sections["解释"]
        )
        or '<div class="explanation-line">暂无补充解释。</div>'
    )
    suggestion_body = (
        "".join(
            f'<div class="explanation-line">{html.escape(line)}</div>'
            for line in sections["建议"]
        )
        or '<div class="explanation-line">暂无补充建议。</div>'
    )
    literature_body_parts: list[str] = []
    for entry in literature_entries[:3]:
        title = html.escape(str(entry.get("title", "未命名文献")))
        url = html.escape(str(entry.get("url", "#")))
        key_points = entry.get("key_points")
        if isinstance(key_points, list) and key_points:
            points = "；".join(html.escape(str(point)) for point in key_points[:3])
        else:
            points = "该文献用于支撑本次解释。"
        literature_body_parts.append(dedent(f"""
                <div class="literature-item">
                  <a class="literature-link" href="{url}" target="_blank" rel="noopener noreferrer">{title}</a>
                  <div>{points}</div>
                </div>
                """).strip())
    literature_body = (
        "".join(literature_body_parts)
        or '<div class="explanation-line">无可用文献依据。</div>'
    )
    card_markup = (
        '<div class="detail-card">'
        '<div class="workspace-kicker">Interpretation</div>'
        '<div class="workspace-card-title">解释与依据</div>'
        '<div class="section-strong-label">解释</div>'
        f"{explanation_body}"
        '<div class="section-strong-label">建议</div>'
        f"{suggestion_body}"
        '<div class="section-strong-label">文献依据</div>'
        f"{literature_body}"
        "</div>"
    )
    st.markdown(
        card_markup,
        unsafe_allow_html=True,
    )


def build_llm_loading_markup() -> str:
    """生成调用 LLM 改写期间的占位提示卡片。"""
    return dedent("""
        <div class="detail-card loading-card">
          <div class="workspace-kicker">LLM Enhancement</div>
          <div class="loading-title">正在生成增强解释</div>
          <div class="loading-copy">
            正在调用 LLM 对研究解释进行改写与整理，页面会在保留核心结论不变的前提下更新为更适合阅读的结果文本。
          </div>
          <div class="loading-pills">
            <div class="loading-pill"><span class="loading-dot"></span>提取核心结论</div>
            <div class="loading-pill"><span class="loading-dot"></span>整理解释结构</div>
            <div class="loading-pill"><span class="loading-dot"></span>校验输出一致性</div>
          </div>
          <div class="loading-bar"><span></span></div>
        </div>
        """).strip()


def render_llm_loading_state(slot: st.delta_generator.DeltaGenerator) -> None:
    """在指定占位区展示 LLM 正在处理的状态。"""
    slot.markdown(build_llm_loading_markup(), unsafe_allow_html=True)


def build_metric_grid_markup(payload: dict[str, str], result: PredictionResult) -> str:
    """生成研究结论卡里的核心指标栅格。"""
    toxicity_tone = html.escape(result.toxicity_level)
    return dedent(f"""
        <div class="metric-grid">
          <div class="metric-card">
            <div class="metric-label">毒性等级</div>
            <div class="metric-value {toxicity_tone}">{html.escape(payload["toxicity_label"])}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">预测细胞活力</div>
            <div class="metric-value">{html.escape(payload["cell_viability_percent"])}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">模型置信度</div>
            <div class="metric-value">{html.escape(payload["confidence"])}</div>
          </div>
        </div>
        """).strip()


def render_result_panel(
    sample: PredictionInput,
    payload: dict[str, str],
    result: PredictionResult,
    literature_entries: list[dict[str, object]],
) -> None:
    """按固定顺序输出结论、样本信息、特征影响与解释依据。"""
    metric_grid = build_metric_grid_markup(payload, result)
    st.markdown(
        (
            '<div class="summary-card">'
            '<div class="workspace-kicker">Research Conclusion</div>'
            '<div class="workspace-card-title">研究结论</div>'
            f"{metric_grid}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    render_input_summary(sample)
    render_feature_bars(result.top_features)
    render_explanation_sections(payload["explanation"], literature_entries)


def main() -> None:
    """串起表单提交、模型推理与结果渲染的整页流程。"""
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

        llm_loading_slot = st.empty()
        try:
            # 预测链路：模型 -> 文献 -> 预测 -> 规则解释 -> 可选 LLM 改写 -> 前端渲染。
            if use_llm:
                render_llm_loading_state(llm_loading_slot)
            bundle = load_or_train_bundle(MODEL_PATH, DATA_PATH)
            literature = select_literature(
                load_literature(LITERATURE_PATH), "feature_explanation"
            )
            result = predict_toxicity(bundle, sample)
            explanation = build_explanation(sample, result, literature)
            if use_llm:
                explanation = generate_llm_response(explanation) or explanation
            llm_loading_slot.empty()
            payload = build_result_payload(result, explanation)
            render_result_panel(sample, payload, result, literature)
        except Exception as error:
            llm_loading_slot.empty()
            st.error(format_user_error(error))


if __name__ == "__main__":
    main()
