# Nano Toxicity Agent Implementation Plan Index

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the nano toxicity prediction agent as a set of independently verifiable implementation subplans.

**Architecture:** The work is split into four deliverable slices. Each subplan creates working software with its own tests and commit, while later subplans build on the artifacts from earlier ones.

**Tech Stack:** Python 3.11+, pandas, numpy, scikit-learn, joblib, Streamlit, pytest.

---

## Subplans

Execute in this order:

1. [Foundation, Schema, Data, and Literature](2026-06-15-nano-toxicity-agent-01-foundation-data.md)
   - Creates project packaging, domain schema, demo CSV, fixed literature JSON, and tests.
   - Verification: `rtk pytest tests/test_schema.py tests/test_literature.py -q`
   - Review hardening: metadata provenance is locked by an automated test.

2. [Model Training and Prediction Service](2026-06-15-nano-toxicity-agent-02-modeling-prediction.md)
   - Creates preprocessing, model training, model persistence, and prediction service.
   - Verification: `rtk pytest tests/test_train_predict.py -q`
   - Review hardening: final prediction models train on full demo data after holdout metric calculation, making the risk-direction test deterministic.

3. [Explanation Engine and Optional LLM Layer](2026-06-15-nano-toxicity-agent-03-explanation-llm.md)
   - Creates deterministic explanation text, experiment suggestions, and no-key LLM fallback.
   - Verification: `rtk pytest tests/test_explain_llm.py -q`

4. [Streamlit App and End-to-End Verification](2026-06-15-nano-toxicity-agent-04-app-e2e.md)
   - Creates the Streamlit UI, README, CLI verification path, and final full-suite checks.
   - Verification: `rtk pytest -q` and `rtk streamlit run app.py --server.address 127.0.0.1 --server.port 8501`
   - Review hardening: Streamlit `AppTest` smoke checks cover default prediction rendering and no-key LLM fallback before manual browser acceptance.

## Scope

This plan implements only the first prototype from the spec:

```text
Streamlit UI + scikit-learn model + local literature summary library + template/optional LLM explanation
```

The full Agent framework, RAG over PDFs, SHAP/LIME explanations, multi-source schema merging, and multi-endpoint toxicology prediction remain follow-up projects.

## Self-Review

Spec coverage:

- Project boundary and first prototype scope: covered by this index and subplan 04 README.
- Data schema, thresholds, validation, demo data: covered by subplan 01.
- Local fixed literature library: covered by subplan 01.
- ML classifier, regressor, persistence, and model-owned prediction: covered by subplan 02.
- Deterministic explanation and optional LLM fallback: covered by subplan 03.
- Streamlit UI, full test suite, CLI prediction, local app launch: covered by subplan 04.
- Engineering review P1/P2/P3 fixes: data provenance with metadata tests, traceable literature URLs, deterministic model behavior tests, OpenAI-compatible LLM fallback, UI error handling, automated Streamlit smoke checks, and manual browser acceptance are covered by subplans 01-04.

Deferred by design:

- Full Agent framework, RAG, SHAP/LIME, multi-source schema merging, and multi-endpoint toxicology prediction are documented as future improvements in the spec, not first-prototype tasks.

Placeholder scan:

- Each subplan includes concrete files, tests, commands, expected results, and commit points.

Type consistency:

- Shared types are defined in subplan 01 before being used by subplans 02-04.
- `ModelBundle`, `predict_toxicity`, and `load_or_train_bundle` are defined in subplan 02 before UI use in subplan 04.
- `build_explanation`, `TOXICITY_TEXT`, and `generate_llm_response` are defined in subplan 03 before UI use in subplan 04.
- LLM configuration consistently uses `LLM_API_KEY`, `LLM_API_BASE`, and optional `LLM_MODEL`.

## Handoff

Plan complete and split into independently verifiable subplans. Two execution options:

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per subplan, review between subplans, fast iteration.
2. **Inline Execution** - execute subplans in this session using executing-plans, with checkpoints after each subplan.

Which approach?
