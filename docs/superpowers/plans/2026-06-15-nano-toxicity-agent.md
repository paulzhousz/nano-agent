# 纳米毒性预测智能体实现计划索引

> **给智能体执行者：** 必须使用子技能：`superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，按任务逐步执行本计划。步骤使用复选框（`- [ ]`）语法跟踪。

**目标：** 将纳米毒性预测智能体拆成一组可独立验证的实现子计划。

**架构：** 工作被拆成四个可交付切片。每个子计划都会产出可运行软件、配套测试和一次提交；后续子计划基于前序产物继续构建。

**技术栈：** Python 3.12、uv、pandas、numpy、scikit-learn、joblib、Streamlit、pytest。

---

## 子计划

按以下顺序执行：

1. [基础、Schema、数据与文献](2026-06-15-nano-toxicity-agent-01-foundation-data.md)
   - 创建项目打包配置、领域 schema、公开数据清洗脚本、完整公开数据清洗结果、核心文献完整摘要生成脚本、核心文献 JSON 和测试。
   - 验证：`rtk uv run pytest tests/test_schema.py tests/test_public_data.py tests/test_literature.py -q`
   - 审核加固：通过自动测试锁定数据来源 metadata。

2. [模型训练与预测服务](2026-06-15-nano-toxicity-agent-02-modeling-prediction.md)
   - 创建预处理、模型训练、模型持久化和预测服务。
   - 验证：`rtk uv run pytest tests/test_train_predict.py -q`
   - 审核加固：先计算 holdout 指标，再用完整公开数据清洗结果训练最终预测模型，并用确定性预测测试锁定模型服务行为。

3. [解释引擎与可选 LLM 层](2026-06-15-nano-toxicity-agent-03-explanation-llm.md)
   - 创建确定性解释文本、实验建议和无 key 时的 LLM fallback。
   - 验证：`rtk uv run pytest tests/test_explain_llm.py -q`

4. [Streamlit 应用与端到端验证](2026-06-15-nano-toxicity-agent-04-app-e2e.md)
   - 创建 Streamlit UI、README、CLI 验证路径和最终全量检查。
   - 验证：`rtk uv run pytest -q` 和 `rtk uv run streamlit run app.py --server.address 127.0.0.1 --server.port 8501`
   - 审核加固：在手动浏览器验收前，用 Streamlit `AppTest` smoke 检查覆盖默认预测渲染和无 key LLM fallback。

## 范围

本计划只实现 spec 中的第一版原型：

```text
Streamlit UI + scikit-learn model + cleaned public nanotoxicity dataset + complete local core-literature summaries + template/optional LLM explanation
```

完整 Agent 框架、PDF RAG、SHAP/LIME 解释、多来源 schema 合并和多终点毒理预测均保留为后续项目。

## 自检

Spec 覆盖情况：

- 项目边界和第一版原型范围：由本索引和子计划 04 的 README 覆盖。
- 数据 schema、阈值、校验、公开数据来源优先级和完整公开数据清洗结果：由子计划 01 覆盖。
- 核心文献完整摘要库：由子计划 01 覆盖。
- ML 分类器、回归器、持久化和模型拥有的预测结论：由子计划 02 覆盖。
- 确定性解释和可选 LLM fallback：由子计划 03 覆盖。
- Streamlit UI、完整测试套件、CLI 预测、本地应用启动：由子计划 04 覆盖。
- 工程审核 P1/P2/P3 修复：完整公开数据清洗脚本、完整公开数据清洗结果及 metadata 测试、核心文献完整摘要生成脚本、可追踪文献 URL、确定性模型行为测试、OpenAI-compatible LLM fallback、UI 错误处理、自动 Streamlit smoke 检查和手动浏览器验收，均由子计划 01-04 覆盖。

按设计延后：

- 完整 Agent 框架、RAG、SHAP/LIME、多来源 schema 合并和多终点毒理预测已在 spec 中记录为未来改进，不属于第一版原型任务。

占位符扫描：

- 每个子计划都包含具体文件、测试、命令、预期结果和提交点。

类型一致性：

- 共享类型在子计划 01 中定义，然后才被子计划 02-04 使用。
- `ModelBundle`、`predict_toxicity` 和 `load_or_train_bundle` 在子计划 02 中定义，然后才在子计划 04 的 UI 中使用。
- `build_explanation`、`TOXICITY_TEXT` 和 `generate_llm_response` 在子计划 03 中定义，然后才在子计划 04 的 UI 中使用。
- LLM 配置统一使用 `LLM_API_KEY`、`LLM_API_BASE` 和可选的 `LLM_MODEL`。

## 交接

计划已完成，并拆分为可独立验证的子计划。两种执行方式：

1. **子智能体驱动（推荐）** - 每个子计划派发一个新的 subagent，子计划之间做审核，快速迭代。
2. **内联执行** - 在当前会话中使用 `executing-plans` 执行，每个子计划后设置检查点。

选择哪种方式？
