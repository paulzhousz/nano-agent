from __future__ import annotations

from typing import Any

from nano_tox_agent.schema import PredictionInput, PredictionResult

TOXICITY_TEXT = {"low": "低毒", "medium": "中毒", "high": "高毒"}


def _clean_feature_name(feature_name: str) -> str:
    cleaned = feature_name.removeprefix("numeric__").removeprefix("categorical__")
    return cleaned.replace("_", " ")


def _build_suggestion(result: PredictionResult) -> str:
    if result.toxicity_level == "high":
        return "建议降低暴露剂量，增加 PEG 等表面修饰，并补充正常细胞系对照验证。"
    if result.toxicity_level == "medium":
        return "建议增加剂量梯度实验，并比较癌细胞系与正常细胞系的敏感性差异。"
    return "可进入下一轮体外验证，并保留正常细胞系对照以确认安全性。"


def _format_literature(literature_entries: list[dict[str, Any]]) -> str:
    if not literature_entries:
        return "无可用文献依据。"

    lines: list[str] = []
    for entry in literature_entries[:3]:
        title = str(entry.get("title", "未命名文献"))
        key_points = entry.get("key_points")
        if key_points:
            points = "；".join(str(point) for point in key_points[:3])
        else:
            points = "该文献用于支撑本次解释。"
        lines.append(f"- {title}：{points}")
    return "\n".join(lines)


def build_explanation(
    sample: PredictionInput,
    result: PredictionResult,
    literature_entries: list[dict[str, Any]],
) -> str:
    del sample

    top_features = result.top_features[:5]
    if top_features:
        feature_lines = [
            f"- {_clean_feature_name(feature_name)}：{weight:.3f}"
            for feature_name, weight in top_features
        ]
    else:
        feature_lines = ["- 无可用主要影响因素"]

    explanation = [
        f"预测毒性等级：{TOXICITY_TEXT.get(result.toxicity_level, result.toxicity_level)}",
        f"预测细胞活力：{result.cell_viability_percent:.1f}%",
        f"模型置信度：{result.confidence:.1%}",
        "主要影响因素：",
        *feature_lines,
        "解释：",
        f"模型当前预测为{TOXICITY_TEXT.get(result.toxicity_level, result.toxicity_level)}，结合细胞活力与关键特征变化进行综合判断。",
        "建议：",
        _build_suggestion(result),
        "文献依据：",
        _format_literature(literature_entries),
    ]
    return "\n".join(explanation)
