# rag_core.py
from __future__ import annotations

import os
import re
import json
import time
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME_DEFAULT = os.environ.get("ESC_RAG_MODEL", "gpt-5.2")
EMBED_MODEL_NAME = os.environ.get("ESC_RAG_EMBED_MODEL", "intfloat/e5-large-v2")
RERANK_MODEL_NAME = os.environ.get("ESC_RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

ARTIFACT_DIR = os.environ.get("ESC_RAG_INDEX_DIR", "esc_rag_artifacts")

FAISS_SEC_PATH = os.path.join(ARTIFACT_DIR, "esc_sections.faiss")
FAISS_REC_PATH = os.path.join(ARTIFACT_DIR, "esc_recommendations.faiss")
SEC_META_PATH = os.path.join(ARTIFACT_DIR, "esc_sections_meta.parquet")
REC_META_PATH = os.path.join(ARTIFACT_DIR, "esc_recommendations_meta.parquet")

# ============================================================
# OPENAI CLIENT + UTILITIES
# ============================================================

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def extract_json(text: str) -> dict:
    """
    Robust JSON extraction: grabs the first {...} block.
    """
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return json.loads(m.group(0))


def _safe_extract_json(raw: str) -> dict:
    try:
        return extract_json(raw)
    except Exception as e:
        head = raw[:2000] if isinstance(raw, str) else str(raw)[:2000]
        raise ValueError(f"Model did not return valid JSON. Raw head:\n{head}") from e


def call_gpt(
    system_prompt: str,
    user_prompt: str,
    model: str = MODEL_NAME_DEFAULT,
    temperature: float = 0.0,
    json_mode: bool = True,
) -> str:
    """
    Compatibility-first OpenAI call:
    - Tries Responses API with response_format (newer SDKs).
    - Falls back to Responses API without response_format (older SDKs).
    - If json_mode=True, we still enforce "Output ONLY JSON" in prompts + robust JSON extraction downstream.
    """
    client = get_client()

    input_payload = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if json_mode:
        # Newer SDKs: responses.create(..., response_format={"type":"json_object"})
        try:
            resp = client.responses.create(
                model=model,
                temperature=temperature,
                input=input_payload,
                response_format={"type": "json_object"},
            )
            return resp.output_text
        except TypeError:
            # Older SDK: no response_format kwarg.
            pass
        except Exception:
            # Any other transient error: fall through to basic call.
            pass

    # Basic call (works broadly)
    resp = client.responses.create(
        model=model,
        temperature=temperature,
        input=input_payload,
    )
    return resp.output_text


# ============================================================
# SCALE TEXT (IN-PROMPT)
# ============================================================

PROGNOSIS_SCALE_TEXT_1_7 = """
PROGNOSIS SCALE (1–7):
1 = Safe for Discharge from ER — Fully stable; no further inpatient management needed.
2 = Short Observation (<24h) — Brief monitoring/diagnostic workup; likely home next day.
3 = Short Admission (1–3 days) — Needs treatment/monitoring; can be discharged within ~72h.
4 = Standard Admission (3–7 days) — Requires therapy, imaging, gradual stabilization; moderate severity.
5 = Prolonged Admission (>7 days) — Complex management; likely complications; slow response.
6 = High Risk of ICU/Intubation — Hemodynamic or respiratory instability; ICU likely.
7 = High Risk of Death (In-hospital) — Critical illness; multiorgan risk or very poor reserve.
""".strip()

HDS_SCALE_TEXT_1_5 = """
STANDARDIZED HOSPITALIZATION DURATION SCALE (HDS-5):
1 = Discharged from ER (<24h) — Treated and discharged same day.
2 = Short Stay (1–2 days) — Brief observation or uncomplicated course.
3 = Standard Stay (3–5 days) — Typical admission for acute decompensation/new diagnosis.
4 = Prolonged Stay (6–10 days) — Slow response; additional evaluation or complex management.
5 = Extended Stay (>10 days) — Complicated hospitalization; often ICU step-down or social issues.
""".strip()

# ============================================================
# SYSTEM PROMPTS
# ============================================================

QUERYGEN_SYSTEM = """
You generate short retrieval queries for ESC guideline passages.
Output only JSON: {"queries":[...]}.
""".strip()

DRAFT_SYSTEM = """
You are evaluating clinical reasoning for research only (NOT real patient care).

CRITICAL CITATION RULES:
- ESC guideline excerpts can support ONLY: definitions, recommended actions/tests, algorithms, risk stratification frameworks, monitoring recommendations.
- Do NOT cite ESC excerpts for patient-specific facts (labs/vitals/imaging/ECG/ABG interpretations, e.g., "cephalization", "respiratory alkalosis", "rising troponin").
- If you mention patient-specific facts, keep them UNCITED in one_liner/justification/clinical_judgment.
- Use citations [chunk_id] ONLY in:
  (a) guideline_support fields
  (b) evidence arrays when the item is truly a guideline-recommendation/definition claim.

FIELD RULES (TO REDUCE VERIFIER FAILS):
- differential[].one_liner: patient-specific reasoning ONLY; default evidence=[] for differential items.
- prognosis[].justification: patient-specific reasoning ONLY; default evidence=[].
- disposition.guideline_support: cite ONLY if the excerpt explicitly states admission/discharge/level-of-care criteria. Otherwise:
  "No direct guideline citation retrieved for this point."
- tests[].reason: guideline-oriented; evidence allowed if excerpt directly supports the test/timing.

STYLE:
- Clinician-readable, short.
- Output MUST be a single JSON object and nothing else.
""".strip()

VERIFY_SYSTEM = """
You are a strict verifier for a clinical RAG system (research only, not patient care).

You will receive:
(1) Retrieved ESC guideline excerpts (each starts with [chunk_id])
(2) A candidate JSON answer

SCOPE OF VERIFICATION:
- Only verify claims explicitly presented as ESC-supported, i.e.:
  (a) any text inside fields named "guideline_support"
  (b) any chunk_id listed inside any "evidence" array
- Do NOT penalize uncited patient-specific reasoning in:
  "one_liner", "justification", "clinical_judgment" (unless they contain citations).

CITATION CHECK:
- If guideline_support contains citations [chunk_id], the cited excerpt must directly support the claim.
- If an evidence array contains chunk_ids, the associated sentence must be a guideline claim supported by those chunks.
- If a field is guideline-like but has no valid support, require:
  guideline_support = "No direct guideline citation retrieved for this point."

CONSISTENCY / FORMAT:
- Query A:
  - exactly 3 differential items; probabilities sum to 1.0 (±0.01)
  - prognosis: exactly 3 items; dx match differential dx in same order
  - tests <= 7
  - prognosis score: 1–7
- Query B:
  - cause_workup_algorithm length 6–10
  - prognosis score: 1–7
  - HDS score: 1–5

Output ONLY JSON:
{
  "ok": true/false,
  "issues": [
    {
      "type": "citation_missing|citation_wrong|unsupported_claim|format_violation|consistency_violation",
      "path": "json.pointer.path",
      "problem": "string",
      "suggestion": "string",
      "missing_info_query": "short retrieval query to find support (optional)"
    }
  ]
}
""".strip()

REVISE_SYSTEM = """
You revise a clinical RAG JSON answer (research only, not patient care).

Inputs:
- Retrieved ESC guideline excerpts
- Original candidate JSON
- Verifier issues

Hard rules:
1) Fix all format/consistency violations.
2) Never cite ESC excerpts for patient-specific facts.
3) For any guideline-like claim lacking direct support, either:
   - add correct citations [chunk_id] IF present, OR
   - set guideline_support to: "No direct guideline citation retrieved for this point."
4) Prefer leaving evidence arrays empty unless you are citing a true ESC recommendation/definition.
5) Output ONLY the final JSON object and nothing else.
""".strip()

# ============================================================
# CASE SNAPSHOT (LIGHT PARSER)
# ============================================================

def parse_case_snapshot(case_text: str) -> dict:
    t = case_text.lower()

    def find_bp():
        m = re.search(r"blood pressure\s*=\s*(\d+)\s*/\s*(\d+)", t)
        return {"sys": int(m.group(1)), "dia": int(m.group(2))} if m else None

    def find_hr():
        m = re.search(r"(heart rate|hr)\s*[:=]\s*(\d+)\s*(bpm)?", t)
        if not m:
            # some lines: "BPM= 108/min"
            m2 = re.search(r"bpm\s*=\s*(\d+)", t)
            return int(m2.group(1)) if m2 else None
        return int(m.group(2))

    def find_spo2():
        m = re.search(r"spo2\s*=\s*(\d+)", t)
        return int(m.group(1)) if m else None

    def find_fio2():
        # Accept "FiO2 = 50%" or "fiO2: 21%" etc.
        m = re.search(r"fio2\s*[:=]\s*([0-9.]+)\s*%?", t)
        if not m:
            return None
        v = float(m.group(1))
        return v / 100.0 if v > 1.5 else v  # handle 50 vs 0.50

    def find_k():
        m = re.search(r"potassium\s*=\s*([0-9.]+)", t)
        return float(m.group(1)) if m else None

    def find_creat():
        m = re.search(r"creatinine\s*=\s*([0-9.]+)", t)
        return float(m.group(1)) if m else None

    def find_lactate():
        m = re.search(r"lactate\s*=\s*([0-9.]+)", t)
        return float(m.group(1)) if m else None

    def find_mental():
        m = re.search(r"gcs\s*=\s*(\d+)\s*/\s*(\d+)", t)
        if not m:
            return None
        return int(m.group(1))

    snap = {
        "bp": find_bp(),
        "hr": find_hr(),
        "spo2": find_spo2(),
        "fio2": find_fio2(),
        "k": find_k(),
        "creatinine": find_creat(),
        "lactate": find_lactate(),
        "gcs": find_mental(),
        "lbbb": "lbbb" in t,
        "cad_terms": any(x in t for x in ["nstemi", "stemi", "pci", "cabg", "coronary", "angina"]),
        "pe_terms": any(x in t for x in ["pulmonary embol", "pe", "ctpa", "d-dimer", "d-dimers"]),
        "hf_terms": any(x in t for x in ["heart failure", "hfr", "hfmr", "hfpef", "pulmonary oedema", "pulmonary edema"]),
        "red_flags": [],
    }

    if snap["k"] is not None and snap["k"] >= 6.0:
        snap["red_flags"].append("hyperkalemia>=6.0")
    if snap["spo2"] is not None and snap["spo2"] < 90:
        snap["red_flags"].append("hypoxemia_spo2<90")
    if snap["bp"] and snap["bp"]["sys"] < 90:
        snap["red_flags"].append("hypotension_sys<90")
    if snap["lactate"] is not None and snap["lactate"] >= 2.0:
        snap["red_flags"].append("lactate>=2.0")
    if snap["gcs"] is not None and snap["gcs"] < 15:
        snap["red_flags"].append("altered_mentation")

    return snap


def needs_icu_by_triggers(snap: dict) -> bool:
    """
    Conservative ICU triggers.
    ICU only if shock/impending respiratory failure markers exist.
    """
    if snap.get("bp") and snap["bp"]["sys"] < 90:
        return True
    if snap.get("lactate") is not None and snap["lactate"] >= 2.0:
        return True
    if snap.get("gcs") is not None and snap["gcs"] < 15:
        return True

    spo2 = snap.get("spo2")
    fio2 = snap.get("fio2")
    if spo2 is not None and spo2 < 90:
        return True
    if fio2 is not None and fio2 >= 0.50 and spo2 is not None and spo2 < 93:
        return True

    return False


def severity_assessment(snap: dict) -> dict:
    """
    Generalized severity tier for guardrails.
    This should not "force ICU"; it only helps nudge monitoring and safety-critical.
    """
    tier = "LOW"

    k = snap.get("k")
    cr = snap.get("creatinine")
    fio2 = snap.get("fio2")
    spo2 = snap.get("spo2")

    if needs_icu_by_triggers(snap):
        tier = "HIGH"
    else:
        # High monitoring but not necessarily ICU
        if (k is not None and k >= 5.8) or (cr is not None and cr >= 1.5):
            tier = "HIGH"
        elif (fio2 is not None and fio2 >= 0.40) or (spo2 is not None and spo2 < 93):
            tier = "MED"

    return {"tier": tier}


# ============================================================
# QUERY GENERATION (pinned coverage queries)
# ============================================================

PINNED_QUERIES_A = [
    "acute heart failure criteria for hospital admission ESC",
    "acute heart failure ICU admission indications ESC",
    "NSTE-ACS admission criteria ECG changes troponin ESC",
    "pulmonary embolism diagnostic algorithm D-dimer rule-out ESC",
    "pulmonary embolism anticoagulation while awaiting diagnosis ESC",
]

PINNED_QUERIES_B = [
    "acute heart failure in-hospital monitoring level of care ESC",
    "acute heart failure identify precipitants triggers guideline ESC",
    "pulmonary embolism outpatient vs inpatient management criteria ESC",
    "acute heart failure oxygen therapy indications CPAP NIV ESC",
]


def generate_queries(case_text: str, question_set: str, model: str = MODEL_NAME_DEFAULT) -> List[str]:
    prompt = f"""
Case:
{case_text}

Question set: {question_set}

Generate 10 retrieval queries to pull ESC guideline text for:
- acute heart failure evaluation (ED/inpatient), diagnostics, treatment
- admission vs discharge and monitoring level (ward vs CICU/ICU) if explicitly stated
- ACS/NSTE-ACS workup pathways and monitoring if relevant
- PE diagnostic + risk stratification algorithm if relevant

Rules:
- Queries 8–16 words.
- Use words like: recommended, should, initial assessment, algorithm, admission, discharge, monitoring.
Return JSON only: {{"queries":[...]}}.
"""
    raw = call_gpt(QUERYGEN_SYSTEM, prompt, model=model, temperature=0.0, json_mode=True)
    data = _safe_extract_json(raw)
    qs = list(data.get("queries", []))

    if question_set.upper() == "A":
        qs += PINNED_QUERIES_A
    else:
        qs += PINNED_QUERIES_B

    # de-dup preserving order
    seen = set()
    out = []
    for q in qs:
        q = (q or "").strip()
        k = q.lower()
        if q and k not in seen:
            seen.add(k)
            out.append(q)
    return out


# ============================================================
# FAISS LOAD + SEARCH
# ============================================================

_faiss = None
_sec_index = None
_rec_index = None
_sec_meta: Optional[pd.DataFrame] = None
_rec_meta: Optional[pd.DataFrame] = None

_embedder = None
_reranker = None


def _load_faiss():
    global _faiss
    if _faiss is None:
        import faiss  # type: ignore
        _faiss = faiss
    return _faiss


def _load_indices_and_meta():
    global _sec_index, _rec_index, _sec_meta, _rec_meta
    faiss = _load_faiss()

    if _sec_index is None:
        if not os.path.exists(FAISS_SEC_PATH):
            raise FileNotFoundError(f"Missing FAISS sections index: {FAISS_SEC_PATH}")
        _sec_index = faiss.read_index(FAISS_SEC_PATH)

    if _rec_index is None:
        if not os.path.exists(FAISS_REC_PATH):
            raise FileNotFoundError(f"Missing FAISS recommendations index: {FAISS_REC_PATH}")
        _rec_index = faiss.read_index(FAISS_REC_PATH)

    if _sec_meta is None:
        if not os.path.exists(SEC_META_PATH):
            raise FileNotFoundError(f"Missing sections metadata: {SEC_META_PATH}")
        _sec_meta = pd.read_parquet(SEC_META_PATH)

    if _rec_meta is None:
        if not os.path.exists(REC_META_PATH):
            raise FileNotFoundError(f"Missing recommendations metadata: {REC_META_PATH}")
        _rec_meta = pd.read_parquet(REC_META_PATH)


def _load_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def _embed_query(text: str) -> np.ndarray:
    v = _load_embedder().encode([f"query: {text}"], normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(v, dtype=np.float32)


def faiss_search(index, meta: pd.DataFrame, query: str, k: int = 25) -> pd.DataFrame:
    qv = _embed_query(query)
    D, I = index.search(qv, k)
    rows = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0:
            continue
        m = meta.iloc[idx].to_dict()
        m["faiss_score"] = float(score)
        rows.append(m)
    return pd.DataFrame(rows)


# ============================================================
# RERANK
# ============================================================

def _load_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL_NAME)
    return _reranker


def rerank(query: str, candidates: pd.DataFrame, top_k: int = 16) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    pairs = [(query, t) for t in candidates["text"].tolist()]
    scores = _load_reranker().predict(pairs)
    out = candidates.copy()
    out["rerank_score"] = scores
    out = out.sort_values("rerank_score", ascending=False).head(top_k).reset_index(drop=True)
    return out


# ============================================================
# CONTEXT BUNDLE
# ============================================================

def build_context_bundle_from_queries(queries: List[str], top_k_final: int = 16) -> dict:
    _load_indices_and_meta()
    assert _sec_meta is not None and _rec_meta is not None

    all_hits = []
    for q in queries:
        a = faiss_search(_sec_index, _sec_meta, q, k=30)
        b = faiss_search(_rec_index, _rec_meta, q, k=35)
        if not a.empty:
            a["source_index"] = "sections"
            a["query"] = q
            all_hits.append(a)
        if not b.empty:
            b["source_index"] = "recommendations"
            b["query"] = q
            all_hits.append(b)

    if not all_hits:
        raise RuntimeError("No retrieval results. Check indices/metadata and embedding model match.")

    cand = pd.concat(all_hits, ignore_index=True)
    cand = cand.sort_values("faiss_score", ascending=False).drop_duplicates("chunk_id").reset_index(drop=True)
    top = rerank(" | ".join(queries), cand, top_k=top_k_final)

    context_blocks = [f"[{r.chunk_id}]\n{r.text}" for r in top.itertuples(index=False)]
    return {
        "retrieval_queries": queries,
        "top_table": top,
        "context_text": "\n\n---\n\n".join(context_blocks),
    }


# ============================================================
# PROMPT BUILDERS (keep your exact I/O style)
# ============================================================

def build_prompt_query_A(case_text: str, bundle: dict) -> str:
    snap = parse_case_snapshot(case_text)
    return f"""
{PROGNOSIS_SCALE_TEXT_1_7}

CASE_TEXT:
{case_text}

CASE_SNAPSHOT (auto-extracted):
{json.dumps(snap, indent=2)}

RETRIEVED_ESC_GUIDELINE_EXCERPTS:
{bundle["context_text"]}

TASK:
Return JSON with this exact structure:

{{
  "differential": [
    {{"dx":"string","probability":0.0,"one_liner":"string",
      "evidence":[{{"chunk_id":"..."}}],"confidence_0_to_1":0.0}}
  ],
  "tests": [
    {{"test":"string","timing":"ER Day0 | Inpatient Day1 | Discharge/Outpatient",
      "reason":"string","evidence":[{{"chunk_id":"..."}}]}}
  ],
  "disposition": {{
    "answer":"ADMIT | DISCHARGE | OBSERVE",
    "department":"CARDIOLOGY DEPARTMENT | CICU | RESPIRATORY DEPARTMENT (PULMONOLOGISTS) | INTERNAL MEDICINE | ICU | SURGICAL",
    "confidence_0_to_1":0.0,
    "guideline_support":"ESC-only statement + [chunk_id] citations OR 'No direct guideline citation retrieved for this point.'",
    "clinical_judgment":"patient-specific reasoning, uncited"
  }},
  "prognosis": [
    {{"dx":"string","score_1_to_7":1,"confidence_0_to_1":0.0,
      "justification":"patient-specific reasoning, uncited",
      "evidence":[{{"chunk_id":"..."}}]}}
  ],
  "safety_critical": [
    {{"item":"string","action":"string","why":"string","evidence":[{{"chunk_id":"..."}}],"severity":"HIGH|MED|LOW"}}
  ]
}}

Constraints:
- differential exactly 3; probabilities sum to 1.0
- tests max 7
- prognosis exactly 3; dx match differential dx order
- scores 1–7
- Default evidence=[] for differential and prognosis unless citing a PURE guideline definition.
- Do NOT use ESC citations for patient-specific labs/ECG/CXR/ABG.
Return ONLY JSON.
""".strip()


def build_prompt_query_B(case_text_post: str, bundle: dict) -> str:
    snap = parse_case_snapshot(case_text_post)
    return f"""
{PROGNOSIS_SCALE_TEXT_1_7}

{HDS_SCALE_TEXT_1_5}

CASE_TEXT (post-diagnosis):
{case_text_post}

CASE_SNAPSHOT (auto-extracted):
{json.dumps(snap, indent=2)}

RETRIEVED_ESC_GUIDELINE_EXCERPTS:
{bundle["context_text"]}

TASK:
Return JSON with this exact structure:

{{
  "prognosis": {{
    "score_1_to_7":1,
    "confidence_0_to_1":0.0,
    "guideline_support":"ESC-only statement + [chunk_id] citations OR 'No direct guideline citation retrieved for this point.'",
    "clinical_judgment":"patient-specific reasoning, uncited"
  }},
  "most_likely_trigger": {{
    "answer":"string",
    "confidence_0_to_1":0.0,
    "guideline_support":"ESC-only statement + [chunk_id] citations OR 'No direct guideline citation retrieved for this point.'",
    "clinical_judgment":"patient-specific reasoning, uncited"
  }},
  "admission_level": {{
    "department":"CICU | CARDIOLOGY DEPARTMENT",
    "confidence_0_to_1":0.0,
    "guideline_support":"ESC-only statement + [chunk_id] citations OR 'No direct guideline citation retrieved for this point.'",
    "clinical_judgment":"patient-specific reasoning, uncited"
  }},
  "cause_workup_algorithm": [
    {{
      "step":1,"action":"string","timing":"Day 0 | Day 1 | Day 2 | Discharge/Outpatient","purpose":"string",
      "guideline_support":"ESC-only statement + [chunk_id] citations OR 'No direct guideline citation retrieved for this point.'",
      "clinical_judgment":"uncited"
    }}
  ],
  "treatment_plan_by_day":[
    {{
      "day":"Day 0","actions":["string"],"guideline_support":"ESC-only + [chunk_id] OR 'No direct guideline citation retrieved for this point.'",
      "clinical_judgment":"uncited"
    }}
  ],
  "followup":{{
    "in_hospital":[{{"item":"string","purpose":"string","guideline_support":"ESC-only + [chunk_id] OR 'No direct guideline citation retrieved for this point.'","clinical_judgment":"uncited"}}],
    "after_discharge":[{{"item":"string","purpose":"string","guideline_support":"ESC-only + [chunk_id] OR 'No direct guideline citation retrieved for this point.'","clinical_judgment":"uncited"}}]
  }},
  "estimated_duration": {{
    "hds_score_1_to_5":1,
    "confidence_0_to_1":0.0,
    "guideline_support":"ESC-only + [chunk_id] OR 'No direct guideline citation retrieved for this point.'",
    "clinical_judgment":"uncited"
  }},
  "safety_critical":[
    {{
      "item":"string","action":"string","why":"string","severity":"HIGH|MED|LOW",
      "guideline_support":"ESC-only + [chunk_id] OR 'No direct guideline citation retrieved for this point.'",
      "clinical_judgment":"uncited"
    }}
  ]
}}

Constraints:
- cause_workup_algorithm 6–10 steps
- prognosis 1–7
- HDS 1–5
- No dosing.
- Do NOT cite ESC for patient-specific facts.
Return ONLY JSON.
""".strip()


# ============================================================
# VERIFICATION LOOP
# ============================================================

def verify_answer(context_text: str, candidate_json: dict, model: str = MODEL_NAME_DEFAULT) -> dict:
    user_prompt = f"""
RETRIEVED_ESC_GUIDELINE_EXCERPTS:
{context_text}

CANDIDATE_JSON:
{json.dumps(candidate_json, ensure_ascii=False)}

Return verifier JSON only.
"""
    raw = call_gpt(VERIFY_SYSTEM, user_prompt, model=model, temperature=0.0, json_mode=True)
    return _safe_extract_json(raw)


def revise_answer(context_text: str, candidate_json: dict, verifier_report: dict, model: str = MODEL_NAME_DEFAULT) -> dict:
    user_prompt = f"""
RETRIEVED_ESC_GUIDELINE_EXCERPTS:
{context_text}

ORIGINAL_CANDIDATE_JSON:
{json.dumps(candidate_json, ensure_ascii=False)}

VERIFIER_REPORT:
{json.dumps(verifier_report, ensure_ascii=False)}

Return revised final JSON only.
"""
    raw = call_gpt(REVISE_SYSTEM, user_prompt, model=model, temperature=0.0, json_mode=True)
    return _safe_extract_json(raw)


def generate_targeted_queries_from_issues(issues: List[dict], max_new: int = 6) -> List[str]:
    qs: List[str] = []
    for it in issues:
        q = (it.get("missing_info_query") or "").strip()
        if q:
            qs.append(q)
        if len(qs) >= max_new:
            break
    # de-dup
    seen = set()
    out: List[str] = []
    for q in qs:
        k = q.lower()
        if k not in seen:
            seen.add(k)
            out.append(q)
    return out


# ============================================================
# HARD VALIDATORS
# ============================================================

def validate_query_A_shape(resp: dict) -> None:
    if len(resp.get("differential", [])) != 3:
        raise ValueError("Query A: differential must have exactly 3 items.")
    probs = [float(x.get("probability", 0.0)) for x in resp["differential"]]
    if abs(sum(probs) - 1.0) > 0.01:
        raise ValueError(f"Query A: probabilities must sum to 1.0; got {sum(probs):.3f}")
    if len(resp.get("prognosis", [])) != 3:
        raise ValueError("Query A: prognosis must have exactly 3 items.")
    dxs = [d.get("dx") for d in resp["differential"]]
    pdxs = [p.get("dx") for p in resp["prognosis"]]
    if pdxs != dxs:
        raise ValueError("Query A: prognosis dx must match differential dx in the same order.")
    if len(resp.get("tests", [])) > 7:
        raise ValueError("Query A: tests must be <= 7 items.")
    for p in resp["prognosis"]:
        s = int(p.get("score_1_to_7", 0))
        if not (1 <= s <= 7):
            raise ValueError("Query A: prognosis score must be 1–7.")


def validate_query_B_shape(resp: dict) -> None:
    algo = resp.get("cause_workup_algorithm", [])
    if not (6 <= len(algo) <= 10):
        raise ValueError("Query B: cause_workup_algorithm must be 6–10 steps.")
    s = int(resp.get("prognosis", {}).get("score_1_to_7", 0))
    if not (1 <= s <= 7):
        raise ValueError("Query B: prognosis score must be 1–7.")
    h = int(resp.get("estimated_duration", {}).get("hds_score_1_to_5", 0))
    if not (1 <= h <= 5):
        raise ValueError("Query B: HDS score must be 1–5.")


# ============================================================
# GUARDRAILS (generalized, avoids making other patients worse)
# ============================================================

def _ensure_list(x):
    return x if isinstance(x, list) else []


def inject_severity_guardrails_queryA(resp: dict, snap: dict) -> dict:
    sev = severity_assessment(snap)

    # --- Add or upgrade safety-critical items for hyperK / high O2 use (patient-specific, uncited) ---
    sc = _ensure_list(resp.get("safety_critical"))

    def has_item(substr: str) -> bool:
        return any(substr.lower() in (it.get("item", "").lower()) for it in sc)

    k = snap.get("k")
    if k is not None and k >= 5.8 and not has_item("hyperkal"):
        sc.insert(
            0,
            {
                "item": "Severe hyperkalemia",
                "action": "Immediate hyperkalemia protocol with continuous cardiac monitoring per local protocol; urgent escalation if refractory or worsening renal failure.",
                "why": "Hyperkalemia increases risk of malignant arrhythmias, especially with conduction disease or renal dysfunction.",
                "evidence": [],
                "severity": "HIGH",
            },
        )

    fio2 = snap.get("fio2")
    if fio2 is not None and fio2 >= 0.50 and not has_item("oxygen requirement"):
        sc.insert(
            0,
            {
                "item": "High supplemental oxygen requirement / respiratory deterioration risk",
                "action": "Close monitoring of oxygenation and work of breathing; escalate respiratory support per local protocol if worsening.",
                "why": "High oxygen requirement suggests limited reserve; respiratory failure can evolve rapidly in acute cardiopulmonary decompensation.",
                "evidence": [],
                "severity": "HIGH" if needs_icu_by_triggers(snap) else "MED",
            },
        )

    resp["safety_critical"] = sc[:6]

    # --- Prevent "ICU" unless hard ICU triggers exist ---
    disp = resp.get("disposition", {}) or {}
    dept = (disp.get("department") or "").upper()

    if "ICU" in dept and not needs_icu_by_triggers(snap):
        disp["department"] = "CICU"
        disp["clinical_judgment"] = (
            (disp.get("clinical_judgment", "") or "").strip()
            + " Level-of-care gate: high-risk features warrant monitored cardiac care (CICU), but no clear shock or refractory hypoxemia criteria for general ICU are present in the provided data."
        ).strip()
        resp["disposition"] = disp

    # --- If severity high and they chose INTERNAL MEDICINE, nudge to CARDIOLOGY (not ICU) ---
    if sev["tier"] == "HIGH":
        dept2 = (resp.get("disposition", {}).get("department") or "").upper()
        if "INTERNAL MEDICINE" in dept2:
            disp["department"] = "CARDIOLOGY DEPARTMENT"
            disp["clinical_judgment"] = (
                (disp.get("clinical_judgment", "") or "").strip()
                + " Severity-gate: high-risk features favor cardiology-led monitored admission rather than a general ward."
            ).strip()
            resp["disposition"] = disp

    return resp


def cap_queryA_prognosis_if_no_icu_triggers(resp: dict, snap: dict) -> dict:
    """
    Avoid inflating to score 5 just because things look scary but stable.
    If NO ICU triggers, cap score at 4.
    """
    if not needs_icu_by_triggers(snap):
        for p in _ensure_list(resp.get("prognosis")):
            s = int(p.get("score_1_to_7", 0))
            if s >= 5:
                p["score_1_to_7"] = 4
                p["confidence_0_to_1"] = min(float(p.get("confidence_0_to_1", 0.6)), 0.65)
                p["justification"] = (
                    (p.get("justification", "") or "").strip()
                    + " No clear shock/respiratory-failure triggers provided; prognosis score capped to standard admission."
                ).strip()
    return resp


def strip_evidence_from_differential_and_prognosis(resp: dict) -> dict:
    """
    Default to evidence=[] in differential/prognosis (reduces verifier noise and citation misuse).
    Tests and guideline_support can still cite.
    """
    for d in _ensure_list(resp.get("differential")):
        d["evidence"] = []
    for p in _ensure_list(resp.get("prognosis")):
        p["evidence"] = []
    return resp


# ============================================================
# ANSWER FUNCTIONS
# ============================================================

def answer_query_A_with_verification(
    case_text: str,
    model: str = MODEL_NAME_DEFAULT,
    max_rounds: int = 2,
    top_k_round0: int = 16,
    top_k_reretrieval: int = 18,
) -> Tuple[dict, dict]:
    snap = parse_case_snapshot(case_text)

    base_queries = generate_queries(case_text, question_set="A", model=model)
    bundle = build_context_bundle_from_queries(base_queries, top_k_final=top_k_round0)

    draft_prompt = build_prompt_query_A(case_text, bundle)
    raw = call_gpt(DRAFT_SYSTEM, draft_prompt, model=model, temperature=0.0, json_mode=True)
    candidate = _safe_extract_json(raw)

    # guardrails (safe defaults)
    candidate = strip_evidence_from_differential_and_prognosis(candidate)
    candidate = inject_severity_guardrails_queryA(candidate, snap)
    candidate = cap_queryA_prognosis_if_no_icu_triggers(candidate, snap)

    verifier_reports = []
    final_queries = list(base_queries)

    for _ in range(max_rounds):
        try:
            validate_query_A_shape(candidate)
            report = verify_answer(bundle["context_text"], candidate, model=model)
        except Exception as e:
            report = {
                "ok": False,
                "issues": [
                    {
                        "type": "format_violation",
                        "path": "/",
                        "problem": str(e),
                        "suggestion": "Fix output structure/constraints exactly.",
                        "missing_info_query": "",
                    }
                ],
            }

        verifier_reports.append(report)

        if report.get("ok") is True:
            break

        new_queries = generate_targeted_queries_from_issues(report.get("issues", []), max_new=6)
        if new_queries:
            final_queries = base_queries + new_queries
            bundle = build_context_bundle_from_queries(final_queries, top_k_final=top_k_reretrieval)

        candidate = revise_answer(bundle["context_text"], candidate, report, model=model)

        # re-apply safe guardrails after revision
        candidate = strip_evidence_from_differential_and_prognosis(candidate)
        candidate = inject_severity_guardrails_queryA(candidate, snap)
        candidate = cap_queryA_prognosis_if_no_icu_triggers(candidate, snap)

    validate_query_A_shape(candidate)

    debug = {
        "base_queries": base_queries,
        "final_queries": final_queries,
        "top_table": bundle["top_table"],
        "verifier_reports": verifier_reports,
        "case_snapshot": snap,
    }
    return candidate, debug


def answer_query_B_with_verification(
    case_text_post: str,
    model: str = MODEL_NAME_DEFAULT,
    max_rounds: int = 2,
    top_k_round0: int = 18,
    top_k_reretrieval: int = 20,
) -> Tuple[dict, dict]:
    snap = parse_case_snapshot(case_text_post)

    base_queries = generate_queries(case_text_post, question_set="B", model=model)
    bundle = build_context_bundle_from_queries(base_queries, top_k_final=top_k_round0)

    draft_prompt = build_prompt_query_B(case_text_post, bundle)
    raw = call_gpt(DRAFT_SYSTEM, draft_prompt, model=model, temperature=0.0, json_mode=True)
    candidate = _safe_extract_json(raw)

    verifier_reports = []
    final_queries = list(base_queries)

    for _ in range(max_rounds):
        try:
            validate_query_B_shape(candidate)
            report = verify_answer(bundle["context_text"], candidate, model=model)
        except Exception as e:
            report = {
                "ok": False,
                "issues": [
                    {
                        "type": "format_violation",
                        "path": "/",
                        "problem": str(e),
                        "suggestion": "Fix output structure/constraints exactly.",
                        "missing_info_query": "",
                    }
                ],
            }

        verifier_reports.append(report)

        if report.get("ok") is True:
            break

        new_queries = generate_targeted_queries_from_issues(report.get("issues", []), max_new=6)
        if new_queries:
            final_queries = base_queries + new_queries
            bundle = build_context_bundle_from_queries(final_queries, top_k_final=top_k_reretrieval)

        candidate = revise_answer(bundle["context_text"], candidate, report, model=model)

    validate_query_B_shape(candidate)

    debug = {
        "base_queries": base_queries,
        "final_queries": final_queries,
        "top_table": bundle["top_table"],
        "verifier_reports": verifier_reports,
        "case_snapshot": snap,
    }
    return candidate, debug


def answer_query_A_single_pass(
    case_text: str,
    model: str = MODEL_NAME_DEFAULT,
    top_k_final: int = 16
) -> Tuple[dict, dict]:
    snap = parse_case_snapshot(case_text)
    queries = generate_queries(case_text, question_set="A", model=model)
    bundle = build_context_bundle_from_queries(queries, top_k_final=top_k_final)
    prompt = build_prompt_query_A(case_text, bundle)
    raw = call_gpt(DRAFT_SYSTEM, prompt, model=model, temperature=0.0, json_mode=True)
    resp = _safe_extract_json(raw)

    resp = strip_evidence_from_differential_and_prognosis(resp)
    resp = inject_severity_guardrails_queryA(resp, snap)
    resp = cap_queryA_prognosis_if_no_icu_triggers(resp, snap)

    validate_query_A_shape(resp)
    debug = {"queries": queries, "top_table": bundle["top_table"], "case_snapshot": snap}
    return resp, debug


def answer_query_B_single_pass(
    case_text_post: str,
    model: str = MODEL_NAME_DEFAULT,
    top_k_final: int = 18
) -> Tuple[dict, dict]:
    snap = parse_case_snapshot(case_text_post)
    queries = generate_queries(case_text_post, question_set="B", model=model)
    bundle = build_context_bundle_from_queries(queries, top_k_final=top_k_final)
    prompt = build_prompt_query_B(case_text_post, bundle)
    raw = call_gpt(DRAFT_SYSTEM, prompt, model=model, temperature=0.0, json_mode=True)
    resp = _safe_extract_json(raw)
    validate_query_B_shape(resp)
    debug = {"queries": queries, "top_table": bundle["top_table"], "case_snapshot": snap}
    return resp, debug


# ============================================================
# PRETTY PRINTERS (DOCTOR READABLE)
# ============================================================

PROGNOSIS_SCALE_1_7 = {
    1: "Safe for Discharge from ER",
    2: "Short Observation (<24h)",
    3: "Short Admission (1–3 days)",
    4: "Standard Admission (3–7 days)",
    5: "Prolonged Admission (>7 days)",
    6: "High Risk of ICU/Intubation",
    7: "High Risk of Death (In-hospital)",
}

HDS_SCALE_1_5 = {
    1: "Discharged from ER (<24h)",
    2: "Short Stay (1–2 days)",
    3: "Standard Stay (3–5 days)",
    4: "Prolonged Stay (6–10 days)",
    5: "Extended Stay (>10 days)",
}


def pretty_query_A(resp: dict) -> str:
    lines = []
    lines.append("=== QUERY A REPORT ===\n")

    lines.append("1) Differential (ranked)")
    for i, d in enumerate(resp["differential"], 1):
        lines.append(f"  {i}. {d['dx']}  (p={d['probability']:.2f}, conf={d['confidence_0_to_1']:.2f})")
        lines.append(f"     {d['one_liner']}")
        ev = [e.get("chunk_id") for e in d.get("evidence", []) if e.get("chunk_id")]
        if ev:
            lines.append(f"     Evidence: {', '.join(ev)}")
    lines.append("")

    lines.append("2) Tests + timing")
    for i, t in enumerate(resp["tests"], 1):
        lines.append(f"  {i}. {t['test']} — {t['timing']}")
        lines.append(f"     Reason: {t['reason']}")
        ev = [e.get("chunk_id") for e in t.get("evidence", []) if e.get("chunk_id")]
        if ev:
            lines.append(f"     Evidence: {', '.join(ev)}")
    lines.append("")

    disp = resp["disposition"]
    lines.append("3) Disposition")
    lines.append(f"  Decision: {disp['answer']} → {disp['department']}  (conf={disp['confidence_0_to_1']:.2f})")
    lines.append(f"  Guideline support: {disp['guideline_support']}")
    lines.append(f"  Clinical judgment: {disp['clinical_judgment']}")
    lines.append("")

    lines.append("4) Prognosis (1–7)")
    for p in resp["prognosis"]:
        label = PROGNOSIS_SCALE_1_7.get(p["score_1_to_7"], "Unknown")
        lines.append(f"  - {p['dx']}: {p['score_1_to_7']} — {label} (conf={p['confidence_0_to_1']:.2f})")
        lines.append(f"    {p['justification']}")
    lines.append("")

    lines.append("5) Safety-critical")
    if resp.get("safety_critical"):
        for s in resp["safety_critical"]:
            lines.append(f"  - [{s['severity']}] {s['item']}")
            lines.append(f"    Action: {s['action']}")
            lines.append(f"    Why: {s['why']}")
    else:
        lines.append("  (none)")

    return "\n".join(lines)


def pretty_query_B(resp: dict) -> str:
    lines = []
    lines.append("=== QUERY B REPORT (Post-diagnosis) ===\n")

    prog = resp["prognosis"]
    lines.append("6) Prognosis")
    lines.append(f"  Score (1–7): {prog['score_1_to_7']}  (conf={prog['confidence_0_to_1']:.2f})")
    lines.append(f"  Guideline support: {prog['guideline_support']}")
    lines.append(f"  Clinical judgment: {prog['clinical_judgment']}")
    lines.append("")

    trg = resp["most_likely_trigger"]
    lines.append("7) Most likely trigger of decompensation")
    lines.append(f"  Answer: {trg['answer']}  (conf={trg['confidence_0_to_1']:.2f})")
    lines.append(f"  Guideline support: {trg['guideline_support']}")
    lines.append(f"  Clinical judgment: {trg['clinical_judgment']}")
    lines.append("")

    adm = resp["admission_level"]
    lines.append("8) Admit to")
    lines.append(f"  Department: {adm['department']}  (conf={adm['confidence_0_to_1']:.2f})")
    lines.append(f"  Guideline support: {adm['guideline_support']}")
    lines.append(f"  Clinical judgment: {adm['clinical_judgment']}")
    lines.append("")

    lines.append("9) Workup to identify the cause (algorithm)")
    for s in resp["cause_workup_algorithm"]:
        lines.append(f"  Step {s['step']}: {s['action']} — {s['timing']}")
        lines.append(f"     Purpose: {s['purpose']}")
        lines.append(f"     Guideline support: {s['guideline_support']}")
        lines.append(f"     Clinical judgment: {s['clinical_judgment']}")
    lines.append("")

    lines.append("10) Treatment plan by day")
    for dayblock in resp["treatment_plan_by_day"]:
        lines.append(f"  {dayblock['day']}:")
        for a in dayblock["actions"]:
            lines.append(f"    • {a}")
        lines.append(f"    Guideline support: {dayblock['guideline_support']}")
        lines.append(f"    Clinical judgment: {dayblock['clinical_judgment']}")
    lines.append("")

    lines.append("11) Follow-up")
    lines.append("  In-hospital:")
    for item in resp["followup"]["in_hospital"]:
        lines.append(f"    • {item['item']} — {item['purpose']}")
        lines.append(f"      Guideline support: {item['guideline_support']}")
        lines.append(f"      Clinical judgment: {item['clinical_judgment']}")
    lines.append("  After discharge:")
    for item in resp["followup"]["after_discharge"]:
        lines.append(f"    • {item['item']} — {item['purpose']}")
        lines.append(f"      Guideline support: {item['guideline_support']}")
        lines.append(f"      Clinical judgment: {item['clinical_judgment']}")
    lines.append("")

    dur = resp["estimated_duration"]
    label = HDS_SCALE_1_5.get(dur["hds_score_1_to_5"], "Unknown")
    lines.append("12) Estimated duration of hospitalization")
    lines.append(f"  HDS-5: {dur['hds_score_1_to_5']} — {label}  (conf={dur['confidence_0_to_1']:.2f})")
    lines.append(f"  Guideline support: {dur['guideline_support']}")
    lines.append(f"  Clinical judgment: {dur['clinical_judgment']}")
    lines.append("")

    lines.append("Safety-critical")
    if resp.get("safety_critical"):
        for s in resp["safety_critical"]:
            lines.append(f"  - [{s['severity']}] {s['item']}")
            lines.append(f"    Action: {s['action']}")
            lines.append(f"    Why: {s['why']}")
            lines.append(f"    Guideline support: {s['guideline_support']}")
            lines.append(f"    Clinical judgment: {s['clinical_judgment']}")
    else:
        lines.append("  (none)")

    return "\n".join(lines)