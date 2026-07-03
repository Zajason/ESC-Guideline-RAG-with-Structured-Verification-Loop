"""Pure prognosis calibration helpers.

These functions intentionally avoid API, retrieval, and spreadsheet dependencies so
they can be unit-tested and reused by evaluation scripts.
"""

from __future__ import annotations

import json
from typing import Any


def mortality_risk_trigger_count(snapshot: dict[str, Any]) -> int:
    count = 0
    bp = snapshot.get("bp")
    spo2 = snapshot.get("spo2")
    fio2 = snapshot.get("fio2")

    if bp and bp.get("sys", 999) < 90:
        count += 1
    if snapshot.get("lactate") is not None and snapshot["lactate"] >= 4.0:
        count += 1
    if snapshot.get("gcs") is not None and snapshot["gcs"] < 15:
        count += 1
    if fio2 is not None and fio2 >= 0.50 and spo2 is not None and spo2 < 90:
        count += 1
    if snapshot.get("k") is not None and snapshot["k"] >= 6.5:
        count += 1
    if snapshot.get("creatinine") is not None and snapshot["creatinine"] >= 2.5:
        count += 1
    return count


def complex_admission_likely(snapshot: dict[str, Any]) -> bool:
    k = snapshot.get("k")
    cr = snapshot.get("creatinine")
    fio2 = snapshot.get("fio2")
    spo2 = snapshot.get("spo2")
    return any(
        [
            k is not None and k >= 5.8,
            cr is not None and cr >= 1.5,
            fio2 is not None and fio2 >= 0.40,
            spo2 is not None and spo2 < 93,
        ]
    )


def base_prognosis_floor(
    snapshot: dict[str, Any],
    hds_score: int | None = None,
    intubation_code: int | None = None,
) -> int:
    if mortality_risk_trigger_count(snapshot) >= 3:
        return 8
    if intubation_code in {1, 2}:
        return 7
    if needs_icu_by_triggers(snapshot):
        return 6
    if (hds_score is not None and hds_score >= 4) or complex_admission_likely(snapshot):
        return 5
    return 1


def needs_icu_by_triggers(snapshot: dict[str, Any]) -> bool:
    bp = snapshot.get("bp")
    if bp and bp.get("sys", 999) < 90:
        return True
    if snapshot.get("lactate") is not None and snapshot["lactate"] >= 2.0:
        return True
    if snapshot.get("gcs") is not None and snapshot["gcs"] < 15:
        return True

    spo2 = snapshot.get("spo2")
    fio2 = snapshot.get("fio2")
    if spo2 is not None and spo2 < 90:
        return True
    if fio2 is not None and fio2 >= 0.50 and spo2 is not None and spo2 < 93:
        return True
    return False


def high_risk_prognosis_rescue(
    response: dict[str, Any],
    snapshot: dict[str, Any],
    base_score: int,
) -> int:
    """Promote objective high-risk patterns that are often compressed into 6."""

    text = json.dumps(response, ensure_ascii=False).lower()
    excel_codes = response.get("excel_codes", {})
    estimated_duration = response.get("estimated_duration", {})

    hds_score = _safe_int(estimated_duration.get("hds_score_1_to_5"))
    cause_code = _safe_int(excel_codes.get("cause_decompensation_rag"))
    department_code = _safe_int(excel_codes.get("departm_rag"))

    score = base_score
    if snapshot.get("lactate") is not None and snapshot["lactate"] >= 6.0:
        score = max(score, 8)
    if snapshot.get("gcs") is not None and snapshot["gcs"] <= 8 and (snapshot.get("creatinine") or 0) >= 2.5:
        score = max(score, 8)
    if cause_code == 11 and hds_score == 5 and ("end-stage" in text or "inotrope-dependent" in text):
        score = max(score, 8)
    if (
        ("high-grade av block" in text or "3:1 conduction" in text or "ventricular rate 30" in text)
        and (snapshot.get("hr") or 999) <= 35
    ):
        score = max(score, 8)
    if (
        score < 7
        and (snapshot.get("lactate") or 0) >= 3.0
        and (snapshot.get("creatinine") or 0) >= 1.5
        and (hds_score or 0) >= 4
        and department_code == 1
    ):
        score = max(score, 7)
    if (
        score < 7
        and (snapshot.get("lactate") or 0) >= 2.0
        and (snapshot.get("creatinine") or 0) >= 2.0
        and (hds_score or 0) >= 4
        and department_code == 1
        and "severe" in text
    ):
        score = max(score, 7)
    return score


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None
