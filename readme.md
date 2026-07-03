# ESC-Guideline RAG with Structured Verification Loop

A research-oriented Retrieval-Augmented Generation (RAG) system for structured clinical reasoning using **ESC (European Society of Cardiology)** guideline text.

> **Research use only. Not for real patient care.**

---

## Overview

This system is designed to:

- Generate **structured clinical assessments** (**Query A** & **Query B** formats)
- Ground **all guideline-level claims** in retrieved ESC excerpts
- Explicitly separate **guideline-supported statements** from **patient-specific clinical reasoning**
- Enforce **internal consistency** via an automated verification loop
- Reduce hallucination and over-escalation (e.g., unnecessary ICU assignment)
- Populate the research Excel workbook with RAG-only structured codes

## Repository Contents

```text
rag_core.py                  # Main RAG pipeline, prompts, retrieval, verification, calibration
batch_fill_rag_excel.py      # Private workbook population workflow
esc_rag/                     # Public, dependency-light helpers for metrics/codebooks/calibration
scripts/evaluate_models.py   # Workbook prognosis benchmark report
scripts/run_demo.py          # Public synthetic demo, no API or patient data
tests/                       # Public unit tests for calibration, codebooks, metrics
examples/                    # Synthetic, non-patient demo cases and model comparison data
docs/                        # Architecture, evaluation, and data privacy notes
```

The public demo and tests do not require private patient data or an OpenAI API key.

```bash
python -m unittest discover -s tests
python scripts/run_demo.py
```

---

## Current Defaults

- Default OpenAI model: `gpt-5.4-mini`
- Prognosis scale: **1-8**, matching the research Excel workbook
- Batch mode: **single-pass by default** for cost control
- Verification loop: optional via `--verified`
- Dynamic LLM query generation: off by default; pinned ESC/HFrEF queries are used to reduce API calls
- Excel coding: the RAG prompts now request explicit `excel_codes` so workbook population does not rely only on prose parsing

## Current Benchmark Position

In the latest calibrated workbook, the RAG system is strongest where guideline grounding and structured output matter most:

- **Best ordinal final-prognosis error**: RAG had the lowest mean absolute error for final prognosis (`0.77`), meaning its wrong answers were less wrong on the 1-8 scale.
- **Best post-diagnosis department assignment**: RAG had the highest agreement for cardiology vs CICU after diagnosis (`70.7%`).
- **Best treatment/workup macro-average**: RAG had the best overall exact agreement across the checked management fields (`81.7%`).
- **Better final-prognosis exact accuracy than DEEP and C+** after high-risk calibration: RAG `41.3%`, DEEP `25.3%`, C+ `26.7%`.
- **RAG remained below CHAT for exact final prognosis**: CHAT `48.0%`, RAG `41.3%`. However, RAG's ordinal error was better, so it was usually closer when it missed.

Model labels in the research workbook:

- **CHAT** = non-paid ChatGPT version used externally.
- **C+** = the user's paid ChatGPT / premium model run.
- **DEEP** = DeepSeek-style comparison run.
- **RAG** = this ESC-guideline retrieval-augmented system using structured prompts, citation rules, code validation, and prognosis calibration.

The current positioning is therefore not simply "RAG wins every column." The more accurate claim is:

> RAG gives more auditable, guideline-grounded answers, beats the comparison models on structured management decisions, beats DEEP/C+ on calibrated final prognosis, and has the lowest ordinal prognosis error even when CHAT has higher exact-category accuracy.

## What Makes This System Different

Most medical RAG systems:

1. Retrieve text
2. Generate an answer
3. Hope citations are correct

This system adds **structural constraints + post-generation validation + safety gating**.

---

## 1) Structured Clinical Output (Not Free Text)

The system outputs strict JSON schemas.

### Query A (Pre-diagnosis)

- **Ranked differential** (exactly 3)
- **Tests with timing**
- **Disposition** (admit vs discharge + department)
- **Prognosis** (1-8 scale)
- **Safety-critical flags**
- **Excel codes** for RAG diagnostic/prognosis/test/disposition columns

### Query B (Post-diagnosis)

- **Prognosis** (1-8 scale)
- **Most likely trigger**
- **Admission level**
- **Cause workup algorithm** (6–10 steps)
- **Treatment plan by day**
- **Follow-up plan**
- **Estimated duration** (HDS-5 scale)
- **Safety-critical issues**
- **Excel codes** for RAG final prognosis, trigger, department, duration, therapies, investigations, and interventions

This makes outputs:

- Reproducible
- Verifiable
- Auditable
- Suitable for research comparison

---

## 2) Citation-Constrained Reasoning

The system enforces strict separation between guideline-backed statements and patient-specific reasoning.

| Type of statement | May cite ESC? |
|---|---|
| Guideline recommendations | ✅ Yes |
| Definitions | ✅ Yes |
| Algorithms | ✅ Yes |
| Patient labs, ECG interpretation, CXR interpretation | ❌ No |
| Clinical judgment | ❌ No |

This prevents:

- False “evidence-backed” patient-specific claims
- Overstating guideline authority
- Hallucinated citations

If no supporting excerpt is retrieved, the system explicitly outputs:

> "No direct guideline citation retrieved for this point."

---

## 3) Retrieval + Reranking Architecture

### Pipeline

1. Pinned and case-triggered ESC/HFrEF retrieval queries
2. Dense retrieval (**E5 embeddings + FAISS**)
3. Cross-encoder reranking
4. Context bundling (top-K final passages)

This ensures:

- High recall
- Reduced irrelevant context
- Stable evidence grounding
- Lower cost than generating retrieval queries with the LLM for every patient

Dynamic LLM query generation can be re-enabled when needed:

```bash
export ESC_RAG_DYNAMIC_QUERYGEN=1
```

Top-K is configurable per query type.

---

## 4) Verification Loop (Hallucination Guard)

After draft generation, an optional verifier model checks:

- Citation validity
- Missing support
- Format violations
- Probability consistency
- Structural constraints

If issues exist:

1. Targeted retrieval queries are generated
2. Additional passages are retrieved
3. Answer is revised
4. Output is re-validated

This creates a **closed-loop reasoning system** when `--verified` is used. For full-cohort Excel population, single-pass is the default because it is substantially cheaper.

Reduces:

- Unsupported claims
- Incorrect citations
- Format drift
- Logical inconsistencies

---

## 5) Level-of-Care Guardrails (Over-ICU Prevention)

Many LLM systems over-escalate. This system includes **rule-based gating**.

### ICU requires (trigger-based)

- Hypotension
- Elevated lactate
- Severe hypoxemia despite high FiO₂
- Altered mentation

### Otherwise

- CICU or cardiology admission
- No automatic ICU inflation

Additional rules:

- Prognosis calibration is ordinal: **5** is allowed for prolonged/complex admission even without ICU triggers, while **6–8** require instability, intubation risk, or mortality-risk physiology
- High-risk prognosis rescue promotes compressed score-6 cases when objective markers are present, including very high lactate, low GCS with renal failure, end-stage/inotrope-dependent HF, high-grade AV block with ventricular rate around 30/min, or combined lactate/renal/CICU physiology
- Hyperkalemia, troponin rise, and high FiO₂ add safety flags but do not force ICU

Prevents:

- Severity inflation
- Unrealistic triage decisions

---

## 6) Explicit Prognostic Scales

Two calibrated scales are used.

### Prognosis (1-8)

- **1** = Safe discharge  
- **4** = Standard admission  
- **5** = Prolonged/complex admission
- **6** = High ICU risk
- **7** = High intubation risk
- **8** = High in-hospital mortality

### Hospitalization Duration Scale (HDS-5)

- **1** = <24h  
- **3** = 3–5 days  
- **5** = >10 days  

Allows:

- Structured outcome modeling
- Future validation studies
- Quantitative evaluation

---

## 7) Differential–Prognosis Consistency Enforcement

Hard rules:

- Exactly **3** differentials
- Probabilities sum to **1.0**
- Prognosis dx must match differential dx order
- Max **7** tests
- Workup algorithm length constrained

These constraints:

- Prevent generative drift
- Force disciplined reasoning
- Improve reproducibility

---

## 8) Safety-Critical Detection Layer

Automatically flags:

- Hyperkalemia
- Respiratory deterioration risk
- Possible ACS
- Renal deterioration
- PE suspicion

Each safety item includes:

- Severity (**HIGH / MED / LOW**)
- Action
- Rationale
- Optional guideline support

Ensures high-risk elements are never buried in prose.

---

## Architecture Overview

```text
Case Text
    ↓
Pinned / Case-Triggered Retrieval Queries
    ↓
FAISS Retrieval (Sections + Recommendations)
    ↓
Cross-Encoder Rerank
    ↓
Context Bundle
    ↓
Draft Generation (JSON Structured)
    ↓
Guardrails Injection
    ↓
Verification Loop
    ↓
Final Validated Output
```

---

## Model And Cost Strategy

The default model is `gpt-5.4-mini`, selected as the cost-conscious default for full-cohort structured RAG extraction.

Recommended use:

- Full cohort / first pass: `gpt-5.4-mini`, single-pass
- Borderline or failed patients: rerun selected patients with `--verified`
- Highest-stakes audit cases: optionally override `--model` with a stronger model

Set the model explicitly:

```bash
export ESC_RAG_MODEL=gpt-5.4-mini
```

Or pass it to the batch runner:

```bash
.venv/bin/python batch_fill_rag_excel.py --model gpt-5.4-mini
```

The script skips already-complete RAG rows unless `--overwrite` or `--rerun-complete` is provided. This avoids paying twice for rows that are already populated.

---

## Excel Batch Population

The batch runner is:

```text
batch_fill_rag_excel.py
```

It:

- Reads patient `.docx` files from the SwissTransfer folder
- Splits each document into Query A and Query B inputs
- Removes old model-answer sections before sending text to RAG
- Runs RAG Query A and Query B
- Writes only RAG columns in the Excel workbook
- Saves raw per-patient JSON logs under `batch_runs/<timestamp>/`
- Stops cleanly on authentication or quota errors
- Retries transient API/network/rate-limit failures with exponential backoff

### Smoke Test

Run a small test before spending on the whole cohort:

```bash
cd /Users/zak/research/med/RAG_AI/rag2.0

ESC_RAG_MODEL=gpt-5.4-mini .venv/bin/python batch_fill_rag_excel.py \
  --excel "../Final AI & HFrEF patients list_RAG_filled_20260504-141028.xlsx" \
  --output "../smoke_RAG_refilled_gpt54mini.xlsx" \
  --only 1 15 \
  --overwrite \
  --rerun-complete
```

### Refill All RAG Rows

Use this when previous RAG rows may have mapping issues and should be recomputed:

```bash
cd /Users/zak/research/med/RAG_AI/rag2.0

ESC_RAG_MODEL=gpt-5.4-mini .venv/bin/python batch_fill_rag_excel.py \
  --excel "../Final AI & HFrEF patients list_RAG_refilled_gpt54mini_highrisk_calibrated.xlsx" \
  --output "../Final AI & HFrEF patients list_RAG_refilled_gpt54mini_rerun.xlsx" \
  --overwrite \
  --rerun-complete \
  --max-retries 5 \
  --retry-sleep 30
```

### Resume Missing Rows Only

If the run stops because of quota, add credits and rerun against the latest output workbook without `--overwrite`:

```bash
ESC_RAG_MODEL=gpt-5.4-mini .venv/bin/python batch_fill_rag_excel.py \
  --excel "../Final AI & HFrEF patients list_RAG_refilled_gpt54mini.xlsx" \
  --output "../Final AI & HFrEF patients list_RAG_refilled_gpt54mini_resumed.xlsx" \
  --max-retries 5 \
  --retry-sleep 30
```

### Higher-Accuracy Rerun For Selected Patients

Use verification only for selected rows because it costs more:

```bash
ESC_RAG_MODEL=gpt-5.4-mini .venv/bin/python batch_fill_rag_excel.py \
  --excel "../Final AI & HFrEF patients list_RAG_refilled_gpt54mini.xlsx" \
  --output "../selected_verified_RAG.xlsx" \
  --only 15 20 47 54 59 \
  --verified \
  --overwrite \
  --rerun-complete
```

---

## Excel Coding Notes

The RAG prompts now request an `excel_codes` object. The batch script uses these codes first and falls back to conservative local parsing only if older logs or malformed responses lack `excel_codes`.

Important coding rule:

- Code a therapy/investigation as positive only when recommended for the patient.
- Do **not** code generic escalation language as positive, e.g. "intubate if worsening" or "CTPA if ADHF uncertain", unless the workbook has a matching "possible/conditional" code.
- The batch mapper now validates Query B codes before writing Excel. Binary fields such as `vasodilators_rag`, `niv_rag`, and `coro_rag` are clamped to `0/1`; `vasodilators=2` is treated as an active patient-specific consideration when vasodilators are discussed, while `niv=2` is written as `1` only if NIV is definitely planned.
- Coronary angiography (`coro_rag`) is kept binary and should be positive only for a patient-specific invasive coronary angiography recommendation.

This change was added because narrative parsing overcalled some actions in prior runs, especially:

- NIV
- possible intubation
- coronary angiography
- vasodilators

---

## Key Design Principles

- Evidence must match claim type.
- Patient-specific reasoning must remain uncited.
- ICU is a triggered state, not a default escalation.
- Structured output > narrative output.
- Verification > blind trust.

---

## Intended Research Use Cases

- Studying LLM + RAG reliability in cardiology
- Comparing guideline-grounded vs non-grounded reasoning
- Measuring hallucination rates with and without verification
- Evaluating triage calibration
- Structured AI clinical reasoning benchmarking

---

## Not Designed For

- Real-time clinical decision support
- Medication dosing recommendations
- Replacing physician judgment
- Emergency deployment systems

---

## Strengths Compared to Typical Medical RAG

| Feature | Typical RAG | This system |
|---|---:|---:|
| Structured output | ❌ | ✅ |
| Citation enforcement | Partial | Strict |
| Patient/guideline separation | ❌ | Explicit |
| Verifier loop | ❌ | ✅ |
| ICU gating | ❌ | ✅ |
| Prognosis calibration | ❌ | Structured |
| Format validation | ❌ | Hard constraints |

---

## Reproducibility

The system logs:

- Retrieval queries
- Retrieved `chunk_id`s
- Final context
- Verifier reports
- Case snapshot parsing

This enables:

- Auditable reasoning trails
- Research reproducibility
- Error analysis

---

## Disclaimer

This system is for research purposes only and is not intended for clinical decision-making.
