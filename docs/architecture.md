# Architecture

This project implements a research-oriented ESC-guideline RAG pipeline for structured clinical reasoning.

```text
Patient case document
    |
    v
DOCX parser and Query A / Query B splitter
    |
    v
Pinned and case-triggered retrieval queries
    |
    v
FAISS retrieval over ESC sections and recommendations
    |
    v
Cross-encoder reranking
    |
    v
Structured JSON generation
    |
    v
Citation and schema verification
    |
    v
Workbook code mapping and prognosis calibration
    |
    v
Excel population and model evaluation
```

## Query A

Query A is the pre-diagnosis task. It produces:

- ranked differential diagnosis
- initial tests
- admission/discharge disposition
- prognosis for each differential
- safety-critical flags
- workbook-ready Excel codes

## Query B

Query B is the post-diagnosis task. It produces:

- final prognosis
- likely trigger of decompensation
- cardiology vs CICU level of care
- cause workup algorithm
- treatment plan by day
- follow-up plan
- hospitalization duration score
- intervention and investigation codes

## Guardrails

The pipeline separates:

- guideline-supported recommendations, which may cite ESC excerpts
- patient-specific clinical reasoning, which must remain uncited

This prevents the model from presenting patient-specific interpretation as if it were directly guideline-derived.

## Calibration

The prognosis scale is ordinal from 1 to 8. The calibrated system allows score 5 for prolonged or complex admission even without ICU features, while scores 6-8 require objective instability, intubation risk, or mortality-risk physiology.
