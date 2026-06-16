# 纳米毒性预测智能体

用于纳米材料细胞毒性预测的可运行 Streamlit 原型。

## 范围

本应用用于预测候选纳米材料的体外细胞毒性风险。应用背景来自肿瘤纳米医学，但它不预测临床人体毒性，也不预测抗肿瘤疗效。

第一版模型使用 `data/processed/toxicity_clean.csv` 的完整公开数据清洗结果训练。报告和演示必须同时披露 `data/processed/toxicity_clean_metadata.json` 中的数据来源、清洗规则和记录数量。

## 安装

```bash
rtk uv venv --python 3.12
rtk uv sync --extra dev
```

## 测试

```bash
rtk uv run pytest -q
```

## 运行

```bash
rtk uv run streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

## 可选 LLM 层

设置 `LLM_API_KEY`、`LLM_API_BASE` 和可选 `LLM_MODEL`，即可启用 OpenAI-compatible 的解释改写。LLM 只允许改写解释文本，永远不决定也不修改毒性等级、预测细胞活力、模型置信度这些核心预测结论。当未配置、请求失败或改写不合规时，系统会自动回退到模板解释。

## 手动验收清单

1. 启动应用
2. 打开 `http://127.0.0.1:8501`
3. 保持默认值点击 `预测毒性`
4. 确认页面显示研究结论、毒性等级、预测细胞活力、模型置信度、解释与依据、特征影响。
5. 在不设置环境变量时启用 `使用可选 LLM 改写解释` 再次点击 `预测毒性`
6. 确认在未配置、请求失败或改写不合规时，预测仍通过模板 fallback 成功渲染，且毒性等级、预测细胞活力、模型置信度不被改动
