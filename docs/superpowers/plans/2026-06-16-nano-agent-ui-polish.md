# 纳米毒性预测页面美化实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `app.py` 重构为“学术展示感”的科研工作台页面，在不改动预测与解释逻辑的前提下，完成顶部导语、双栏布局、结果双层结构、空状态和特征强度展示。

**Architecture:** 保持所有业务逻辑仍集中在现有 `app.py` 调用链：`load_or_train_bundle()`、`predict_toxicity()`、`build_explanation()`、`generate_llm_response()` 不变。页面层通过新增少量渲染 helper 和样式注入函数重组布局，并用 `streamlit.testing.v1.AppTest` 锁定关键 UI 文案、空状态与预测后结构。

**Tech Stack:** Python 3.12、Streamlit、pytest、streamlit.testing.v1、现有 `nano_tox_agent` 预测/解释模块。

---

## 文件结构

- 修改：`app.py`
  - 继续承担单文件 Streamlit 入口职责
  - 新增样式注入、导语区、输入表单、空状态、结果区等渲染 helper
- 修改：`tests/test_app_payload.py`
  - 保留现有 payload / 报错测试
  - 增加空状态、预测后结论区、特征区新结构的界面断言
- 可选修改：`README.md`
  - 仅当手动验收步骤需要补充新的界面预期时更新

## Task 1: 用测试锁定新的页面结构

**Files:**
- Modify: `tests/test_app_payload.py`
- Modify: `app.py`

- [ ] **Step 1: 写失败的空状态与新结果结构测试**

在 `tests/test_app_payload.py` 追加以下测试：

```python
def test_streamlit_default_view_shows_empty_state_guidance():
    app = AppTest.from_file(str(APP_FILE))
    app.run()
    assert not app.exception

    markdown_values = [node.value for node in app.markdown]
    assert any("纳米材料毒性评估与解释" in value for value in markdown_values)
    assert any("当前页面可完成什么任务" in value for value in markdown_values)
    assert any("预测后将输出哪些结果" in value for value in markdown_values)


def test_streamlit_prediction_flow_renders_summary_and_feature_bars():
    app = AppTest.from_file(str(APP_FILE))
    app.run()
    assert not app.exception

    app.button[0].click().run()
    assert not app.exception

    metric_labels = [metric.label for metric in app.metric]
    assert "毒性等级" in metric_labels
    assert "预测细胞活力" in metric_labels
    assert "模型置信度" in metric_labels

    markdown_values = [node.value for node in app.markdown]
    assert any("研究结论" in value for value in markdown_values)
    assert any("解释与依据" in value for value in markdown_values)
    assert any("特征影响" in value for value in markdown_values)
```

- [ ] **Step 2: 运行测试确认失败**

运行：`rtk uv run pytest tests/test_app_payload.py -q`

预期：新增测试失败，因为当前页面没有“空状态说明”“研究结论”“解释与依据”“特征影响”等新结构文案。

- [ ] **Step 3: 保留旧断言中仍然有效的行为约束**

确认 `tests/test_app_payload.py` 仍保留以下现有测试，不删除：

```python
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
```

- [ ] **Step 4: 提交测试基线**

```bash
rtk git add tests/test_app_payload.py
rtk git commit -m "test: lock nano agent polished ui structure"
```

## Task 2: 重构页面骨架、样式变量和输入工作区

**Files:**
- Modify: `app.py`
- Test: `tests/test_app_payload.py`

- [ ] **Step 1: 在 `app.py` 中引入页面样式与导语区 helper**

在 `app.py` 的 `format_user_error()` 后新增以下函数骨架：

```python
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


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card" style="padding: 1.4rem 1.5rem 1.5rem;">
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
```

- [ ] **Step 2: 在 `app.py` 中加入输入表单 helper，替换 sidebar 布局**

继续在 `app.py` 中新增输入表单函数：

```python
def render_input_panel() -> tuple[PredictionInput, bool, bool]:
    st.markdown('<div class="section-card" style="padding: 1.25rem;">', unsafe_allow_html=True)
    st.markdown("### 参数录入")
    st.markdown('<div class="section-copy">按材料属性、暴露条件和实验对象填写参数后统一预测。</div>', unsafe_allow_html=True)

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
```

- [ ] **Step 3: 最小化改造 `main()` 以接入导语区和双栏骨架**

将 `main()` 重写为以下骨架：

```python
def main() -> None:
    st.set_page_config(page_title="纳米毒性预测智能体", layout="wide")
    inject_styles()
    render_hero()

    left_col, right_col = st.columns([0.95, 1.25], gap="large")
    with left_col:
        sample, use_llm, submitted = render_input_panel()
    with right_col:
        st.markdown("<!-- result panel placeholder -->", unsafe_allow_html=True)
```

- [ ] **Step 4: 运行测试，确认仍然失败但骨架已接入**

运行：`rtk uv run pytest tests/test_app_payload.py -q`

预期：仍有结果区相关断言失败，但应用正常加载，说明骨架和表单改造没有引入异常。

- [ ] **Step 5: 提交页面骨架**

```bash
rtk git add app.py
rtk git commit -m "feat: scaffold academic ui shell for streamlit app"
```

## Task 3: 实现空状态、研究结论区、解释区和特征强度条

**Files:**
- Modify: `app.py`
- Test: `tests/test_app_payload.py`

- [ ] **Step 1: 在 `app.py` 中新增结果 helper**

在 `app.py` 中新增以下函数：

```python
def build_summary_text(result: PredictionResult) -> str:
    label = TOXICITY_TEXT.get(result.toxicity_level, result.toxicity_level)
    if result.toxicity_level == "low":
        return f"样本呈{label}性倾向，适合作为后续候选材料继续体外验证。"
    if result.toxicity_level == "medium":
        return f"样本呈{label}风险，建议结合剂量梯度和对照实验进一步复核。"
    return f"样本呈{label}风险，建议优先优化剂量或表面修饰后再进入下一轮筛选。"


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-card" style="padding: 1.25rem 1.35rem;">
          <div class="eyebrow">Result Workspace</div>
          <h3 style="margin-top: 0.35rem;">当前页面可完成什么任务</h3>
          <div class="section-copy">
            输入纳米材料与实验参数后，页面会输出毒性等级、预测细胞活力、模型置信度、
            智能体解释和特征影响，便于科研筛选与演示汇报。
          </div>
          <h4 style="margin-top: 1rem;">预测后将输出哪些结果</h4>
          <div class="section-copy">研究结论摘要、三项关键指标、解释与依据、特征影响强度。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_bars(top_features: list[tuple[str, float]]) -> None:
    st.markdown("#### 特征影响")
    if not top_features:
        st.markdown('<div class="muted-text">暂无可用特征影响结果。</div>', unsafe_allow_html=True)
        return
    max_score = max(score for _, score in top_features) or 1.0
    for feature_name, score in top_features:
        width = max(8.0, round(score / max_score * 100, 1))
        st.markdown(
            f"""
            <div style="margin-bottom: 0.85rem;">
              <div style="display:flex;justify-content:space-between;gap:1rem;">
                <strong>{feature_name}</strong>
                <span class="muted-text">{score:.4f}</span>
              </div>
              <div class="feature-bar"><span style="width:{width}%;"></span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_result_panel(payload: dict[str, str], result: PredictionResult) -> None:
    st.markdown(
        f"""
        <div class="summary-card" style="padding: 1.25rem 1.35rem; margin-bottom: 1rem;">
          <div class="eyebrow">Research Summary</div>
          <h3 style="margin: 0.45rem 0 0.35rem;">研究结论</h3>
          <div class="section-copy">{build_summary_text(result)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("毒性等级", payload["toxicity_label"])
    metric_col2.metric("预测细胞活力", payload["cell_viability_percent"])
    metric_col3.metric("模型置信度", payload["confidence"])

    st.markdown('<div class="section-card" style="padding: 1.15rem 1.25rem; margin-top: 1rem;">', unsafe_allow_html=True)
    st.markdown("#### 解释与依据")
    for paragraph in payload["explanation"].splitlines():
        if paragraph.strip():
            st.markdown(paragraph)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card" style="padding: 1.15rem 1.25rem; margin-top: 1rem;">', unsafe_allow_html=True)
    render_feature_bars(result.top_features)
    st.markdown("</div>", unsafe_allow_html=True)
```

- [ ] **Step 2: 用新 helper 完成 `main()` 的结果分支**

把 `main()` 右栏补全为以下逻辑：

```python
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
        except Exception as error:
            st.error(format_user_error(error))
            return

        render_result_panel(payload, result)
```

- [ ] **Step 3: 调整现有 LLM fallback 测试以匹配新渲染方式**

将 `tests/test_app_payload.py` 中末尾断言：

```python
    assert any("智能体解释" in subheader.value for subheader in app.subheader)
    assert len(app.text) == 1
    assert app.text[0].value == expected_explanation
```

改为：

```python
    markdown_values = [node.value for node in app.markdown]
    assert any("解释与依据" in value for value in markdown_values)
    assert any(expected_explanation.splitlines()[0] in value for value in markdown_values)
```

- [ ] **Step 4: 运行测试，确认全部通过**

运行：`rtk uv run pytest tests/test_app_payload.py -q`

预期：通过，显示 `6 passed` 或更高，且无 `app.exception` 失败。

- [ ] **Step 5: 提交结果区实现**

```bash
rtk git add app.py tests/test_app_payload.py
rtk git commit -m "feat: render academic result workspace for nano agent"
```

## Task 4: 全量验证与验收收尾

**Files:**
- Modify: `README.md`（仅当实际界面文案变化需要同步）
- Modify: `app.py`
- Modify: `tests/test_app_payload.py`

- [ ] **Step 1: 跑全量自动测试**

运行：`rtk uv run pytest -q`

预期：全部通过，无新增回归。

- [ ] **Step 2: 启动 Streamlit 做手动验收**

运行：`rtk uv run streamlit run app.py --server.address 127.0.0.1 --server.port 8501`

手动检查：

```text
1. 首屏可见顶部导语卡和左右双栏结构。
2. 未预测时右栏展示空状态说明，不是空白。
3. 点击“预测毒性”后先看到“研究结论”和三项指标。
4. 下方出现“解释与依据”和“特征影响”两个分区。
5. 勾选“使用可选 LLM 改写解释”但未配置环境变量时，页面仍能正常展示 fallback 解释。
6. 窗口缩窄到移动端宽度时，无水平滚动。
```

- [ ] **Step 3: 只有在 README 与实际界面预期不一致时再更新 README**

如需修改 `README.md`，将手动验收清单第 4 步替换为：

```markdown
4. 确认页面显示研究结论、毒性等级、预测细胞活力、模型置信度、解释与依据、特征影响。
```

- [ ] **Step 4: 运行最终状态检查**

运行：

```bash
rtk git status --short
rtk git log --oneline -n 3
```

预期：

- `git status` 只显示本次任务预期文件改动
- 最近 3 个提交包含测试基线、页面骨架、结果区实现

- [ ] **Step 5: 提交收尾改动**

```bash
rtk git add app.py tests/test_app_payload.py README.md
rtk git commit -m "chore: finalize nano agent ui polish verification"
```
