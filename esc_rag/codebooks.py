"""Workbook codebook helpers used by tests and evaluation utilities."""

from __future__ import annotations

from typing import Any


QUERY_A_ALLOWED = {
    "diag1": set(range(1, 16)),
    "diag2": set(range(1, 16)),
    "diag3": set(range(1, 16)),
    "new_tni": {0, 1},
    "er_echo": {0, 1},
    "er_ct": {0, 1, 2, 3},
    "er_ctpa": {0, 1},
    "er_us": {0, 1, 2, 3, 4},
    "discharge": {0, 1},
    "department_a": {1, 2, 3, 4, 5, 6},
    "prog_a1": set(range(1, 9)),
    "prog_a2": set(range(1, 9)),
    "prog_a3": set(range(1, 9)),
}

QUERY_B_ALLOWED = {
    "prog_b": set(range(1, 9)),
    "cause": set(range(1, 13)),
    "department_b": {0, 1},
    "hds": set(range(1, 6)),
    "diuretics": {0, 1, 2},
    "vasodilators": {0, 1},
    "intubation": {0, 1, 2},
    "niv": {0, 1},
    "abx": {0, 1},
    "inotropes": {0, 1, 2, 3, 4},
    "ace_arb_arni": {0, 1},
    "bb": {0, 1},
    "sglt2": {0, 1},
    "mra": {0, 1},
    "antiarrhythmic": {0, 1},
    "echo_b": {0, 1},
    "coro": {0, 1},
    "ctca": {0, 1},
    "ct_b": {0, 1},
    "ctpa_b": {0, 1, 2},
    "mri": {0, 1},
    "mri_in_hosp": {0, 1},
    "icd": {0, 1, 2, 3, 4, 5, 6},
    "icd_in_hosp": {0, 1},
    "interrogation": {0, 1},
    "holter": {0, 1},
    "holter_in_hosp": {0, 1},
    "aortic_valve": {0, 1, 2, 3},
    "aortic_method": {1, 2},
    "mitral_valve": {0, 1, 2, 3},
    "mitral_method": {1, 2},
}


def as_int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def clamp_int(value: Any, allowed: set[int], default: int | None = None) -> int | None:
    ivalue = as_int_or_none(value)
    if ivalue in allowed:
        return ivalue
    return default


def binary(value: Any) -> int:
    ivalue = as_int_or_none(value)
    return 1 if ivalue is not None and ivalue > 0 else 0


def invalid_codes(values: dict[str, Any], allowed: dict[str, set[int]]) -> dict[str, Any]:
    bad = {}
    for key, allowed_values in allowed.items():
        if key not in values:
            continue
        value = values[key]
        if value is None or value == "":
            continue
        if value not in allowed_values:
            bad[key] = value
    return bad
