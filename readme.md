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

---

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
- **Prognosis** (1–7 scale)
- **Safety-critical flags**

### Query B (Post-diagnosis)

- **Prognosis** (1–7 scale)
- **Most likely trigger**
- **Admission level**
- **Cause workup algorithm** (6–10 steps)
- **Treatment plan by day**
- **Follow-up plan**
- **Estimated duration** (HDS-5 scale)
- **Safety-critical issues**

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

1. Dense retrieval (**E5 embeddings + FAISS**)
2. Cross-encoder reranking
3. Context bundling (top-K final passages)

This ensures:

- High recall
- Reduced irrelevant context
- Stable evidence grounding

Top-K is configurable per query type.

---

## 4) Verification Loop (Hallucination Guard)

After draft generation, a verifier model checks:

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

This creates a **closed-loop reasoning system**, not a single-pass generator.

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

- Prognosis scores **≥ 5** are capped unless ICU triggers exist
- Hyperkalemia, troponin rise, and high FiO₂ add safety flags but do not force ICU

Prevents:

- Severity inflation
- Unrealistic triage decisions

---

## 6) Explicit Prognostic Scales

Two calibrated scales are used.

### Prognosis (1–7)

- **1** = Safe discharge  
- **4** = Standard admission  
- **7** = High in-hospital mortality

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
Query Generation
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




