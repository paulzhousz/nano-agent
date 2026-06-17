# 纳米毒性预测页面 Spec 对齐实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将当前“已可用但视觉完成度不足”的 Streamlit 页面继续打磨到更接近 `docs/superpowers/specs/2026-06-16-nano-agent-ui-polish-design.md` 的学术展示感版本。

**Architecture:** 保持预测、解释、文献和 LLM fallback 调用链不变，只重构 `app.py` 中的展示层和少量测试。重点是加强导语区、空状态、输入卡片层级、结果摘要卡和解释版式，同时补一层能锁定真实结果态切换的 UI 测试。

**Tech Stack:** Python 3.12、Streamlit、streamlit.testing.v1、pytest、现有 `nano_tox_agent` 预测/解释模块。

---

## 文件结构

- Modify: `app.py`
  - 继续作为单文件页面入口
  - 细化样式系统、导语区、空状态、结果摘要、解释版式和输入卡片层级
- Modify: `tests/test_app_payload.py`
  - 锁定更明确的结果态与空状态语义
  - 增加对结果摘要、输入摘要、解释段落的断言
- Modify: `README.md`（仅当手动验收步骤需要补充时）

## Task 1: 锁定 spec 缺口对应的失败测试

**Files:**
- Modify: `tests/test_app_payload.py`
- Modify: `app.py`

- [ ] **Step 1: 追加失败测试，锁定“学术展示感”缺口**

在 `tests/test_app_payload.py` 追加以下测试：

```python
def test_streamlit_empty_state_uses_guidance_not_result_language():
    app = AppTest.from_file(str(APP_FILE))
    app.run()
    assert not app.exception

    rendered = "\n".join(markdown_values(app))
    assert "当前页面可完成什么任务" in rendered
    assert "预测后将输出哪些结果" in rendered
    assert "研究结论" not in rendered


def test_streamlit_prediction_flow_renders_input_summary_card():
    sample, result, explanation = build_default_prediction_context()
    app = AppTest.from_file(str(APP_FILE))
    app.run()
    assert not app.exception

    app.button[0].click().run()
    assert not app.exception

    rendered = "\n".join(markdown_values(app))
    assert f"粒径 {sample.particle_size_nm:.0f} nm" in rendered
    assert f"剂量 {sample.dose_ug_ml:.0f} μg/mL" in rendered
    assert sample.cell_type in rendered
    assert "输入摘要" in rendered


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
```

- [ ] **Step 2: 运行测试确认先失败**

运行：`rtk uv run pytest tests/test_app_payload.py -q`

预期：

- 新增 2-3 条测试失败
- 原有测试继续通过
- 失败原因应集中在当前空状态语义、输入摘要缺失、解释区层级不够

- [ ] **Step 3: 提交测试基线**

```bash
rtk git add tests/test_app_payload.py
rtk git commit -m "test: lock remaining nano agent ui spec gaps"
```

## Task 2: 重做导语区、空状态和输入工作区层级

**Files:**
- Modify: `app.py`
- Test: `tests/test_app_payload.py`

- [ ] **Step 1: 调整全局样式，强化学术展示感**

在 `app.py` 的 `inject_styles()` 中，将相关样式改为更明确的展示层级：

```python
.hero-card {
    padding: 1.75rem 1.9rem 1.8rem;
    margin-bottom: 1.4rem;
    background:
        linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(248, 250, 252, 0.94) 100%);
    position: relative;
    overflow: hidden;
}

.hero-card::after {
    content: "";
    position: absolute;
    top: 0;
    right: 0;
    width: 180px;
    height: 180px;
    background: radial-gradient(circle, rgba(3, 105, 161, 0.12) 0%, transparent 70%);
    pointer-events: none;
}

.hero-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-top: 1rem;
}

.hero-chip {
    border: 1px solid #CBD5E1;
    background: #F8FAFC;
    color: var(--secondary);
    border-radius: 999px;
    padding: 0.3rem 0.7rem;
    font-size: 0.82rem;
}

.section-card {
    padding: 1.35rem;
}

.input-group {
    border-top: 1px solid #E2E8F0;
    padding-top: 0.9rem;
    margin-top: 0.9rem;
}
```

- [ ] **Step 2: 重写 `render_hero()`，补导语摘要标签**

将 `render_hero()` 调整为：

```python
def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
          <div class="eyebrow">Academic Screening Interface</div>
          <div class="hero-title">纳米材料毒性评估与解释</div>
          <div class="hero-copy">
            面向肿瘤纳米药物筛选与临床前研究判断的单页工作台，
            以研究结论、关键指标和解释依据为主轴，适合日常复核与汇报截图。
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
```

- [ ] **Step 3: 重写 `render_empty_state()`，去掉结果态语义污染**

将 `render_empty_state()` 改为说明卡，不再使用“研究结论”标题：

```python
def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-card">
          <div class="panel-title">工作区说明</div>
          <div class="summary-text">
            右侧区域会在提交预测后切换为正式结果工作区，按“研究结论、关键指标、解释与依据、特征影响”的顺序展示结果。
          </div>
          <div class="panel-title">当前页面可完成什么任务</div>
          <div class="summary-text">
            你可以填写纳米材料属性、暴露条件和实验对象信息，快速完成一次科研筛选式毒性预测。
          </div>
          <div class="panel-title">预测后将输出哪些结果</div>
          <div class="summary-text">
            页面会输出研究结论、毒性等级、预测细胞活力、模型置信度，以及解释与依据和特征影响两块分析内容。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
```

- [ ] **Step 4: 调整 `render_input_panel()` 的分组呈现**

在 `st.form(...)` 内给每组包上统一容器：

```python
st.markdown('<div class="input-group">', unsafe_allow_html=True)
st.markdown("#### 材料属性")
# ...当前这组字段
st.markdown("</div>", unsafe_allow_html=True)
```

对四个分组都这样处理，并把按钮前增加一条辅助说明：

```python
st.markdown(
    '<div class="muted-text" style="margin-top: 0.75rem;">填写完成后统一提交，结果将在右侧工作区呈现。</div>',
    unsafe_allow_html=True,
)
```

- [ ] **Step 5: 运行测试确认空状态与输入摘要类断言转绿**

运行：`rtk uv run pytest tests/test_app_payload.py -q`

预期：

- 空状态语义测试通过
- 其他与结果摘要/解释区相关的新测试可能仍失败

- [ ] **Step 6: 提交导语与输入区改造**

```bash
rtk git add app.py
rtk git commit -m "feat: strengthen nano agent academic page framing"
```

## Task 3: 重做结果摘要卡、输入摘要卡和解释版式

**Files:**
- Modify: `app.py`
- Test: `tests/test_app_payload.py`

- [ ] **Step 1: 新增输入摘要 helper**

在 `app.py` 中新增：

```python
def render_input_summary(sample: PredictionInput) -> None:
    st.markdown(
        f"""
        <div class="detail-card">
          <div class="panel-title">输入摘要</div>
          <div class="summary-text">
            粒径 {sample.particle_size_nm:.0f} nm<br>
            剂量 {sample.dose_ug_ml:.0f} μg/mL<br>
            细胞 {html.escape(sample.cell_type)}<br>
            检测方法 {html.escape(sample.assay_method)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
```

- [ ] **Step 2: 重做 `render_result_panel()` 的上层布局**

将 `render_result_panel()` 上层改为“结论卡 + 输入摘要卡”双列，然后再显示三项指标：

```python
summary_left, summary_right = st.columns([1.2, 0.8], gap="medium")
with summary_left:
    st.markdown('<div class="summary-card">', unsafe_allow_html=True)
    st.markdown("### 研究结论")
    st.markdown(
        f'<div class="summary-text">{build_summary_text(payload, result)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
with summary_right:
    render_input_summary(sample)
```

同时把 `render_result_panel()` 签名改为：

```python
def render_result_panel(sample: PredictionInput, payload: dict[str, str], result: PredictionResult) -> None:
```

并在 `main()` 中同步改调用：

```python
render_result_panel(sample, payload, result)
```

- [ ] **Step 3: 将解释文本转为段落块展示**

新增 helper：

```python
def render_explanation_sections(explanation: str) -> None:
    st.markdown('<div class="detail-card">', unsafe_allow_html=True)
    st.markdown("### 解释与依据")
    for line in explanation.splitlines():
        if not line.strip():
            continue
        st.markdown(line)
    st.markdown("</div>", unsafe_allow_html=True)
```

然后在 `render_result_panel()` 中用它替代当前的：

```python
st.markdown(payload["explanation"])
```

- [ ] **Step 4: 微调特征影响视觉层级**

在 `inject_styles()` 中补：

```python
.detail-card {
    background: #FFFFFF;
    border: 1px solid #D9E2EC;
    border-radius: 16px;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
}

.feature-meta span:first-child {
    font-weight: 600;
    color: var(--primary);
}
```

- [ ] **Step 5: 运行测试确认结果态测试全部转绿**

运行：`rtk uv run pytest tests/test_app_payload.py -q`

预期：全部通过，尤其包括：

- `test_streamlit_prediction_flow_renders_new_summary_sections`
- `test_streamlit_prediction_flow_renders_input_summary_card`
- `test_streamlit_prediction_flow_renders_explanation_as_sections`

- [ ] **Step 6: 提交结果区视觉对齐**

```bash
rtk git add app.py tests/test_app_payload.py
rtk git commit -m "feat: align nano agent result workspace with design spec"
```

## Task 4: 全量验证与浏览器验收

**Files:**
- Modify: `README.md`（仅当手动验收描述需要补充）
- Modify: `app.py`
- Modify: `tests/test_app_payload.py`

- [ ] **Step 1: 跑全量自动测试**

运行：`rtk uv run pytest -q`

预期：全部通过，无回归。

- [ ] **Step 2: 启动 Streamlit 页面做浏览器验收**

运行：`rtk uv run streamlit run app.py --server.address 127.0.0.1 --server.port 8501`

人工检查：

```text
1. 首屏导语区应有标题、副标题和 3 个摘要标签。
2. 空状态不应出现“研究结论”作为结果态标题。
3. 输入区四个分组应有明确分隔，而不是一整串默认控件。
4. 提交预测后，上层应先出现“研究结论”和“输入摘要”两张卡。
5. 关键指标、解释与依据、特征影响三段应有清晰卡片边界。
6. 桌面端整体应更像学术汇报页，而不是默认工具页。
```

- [ ] **Step 3: 只有在 README 需要补充时再更新**

若要补充 README，可将手动验收第 4 步扩展为：

```markdown
4. 确认页面显示研究结论、输入摘要、毒性等级、预测细胞活力、模型置信度、解释与依据、特征影响。
```

- [ ] **Step 4: 做最终状态检查**

运行：

```bash
rtk git status --short
rtk git log --oneline -n 3
```

预期：

- 只存在本任务预期改动
- 最近提交能清晰区分“导语/输入区对齐”和“结果区对齐”

- [ ] **Step 5: 提交收尾验证**

```bash
rtk git add app.py tests/test_app_payload.py README.md
rtk git commit -m "chore: verify nano agent spec-aligned UI polish"
```
