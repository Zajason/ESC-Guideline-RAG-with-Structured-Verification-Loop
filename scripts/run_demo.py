#!/usr/bin/env python3
"""Run the public synthetic metrics demo.

This does not call the OpenAI API and does not require private patient data.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from esc_rag.metrics import prognosis_metrics


def main() -> int:
    path = Path(__file__).resolve().parents[1] / "examples" / "synthetic_model_comparison.csv"
    rows = list(csv.DictReader(path.open()))
    models = ["CHAT", "DEEP", "C+", "RAG"]
    print(f"Synthetic demo rows: {len(rows)}")
    print(f"{'Model':<8} {'Exact':>8} {'MAE':>7} {'Within1':>8}")
    for model in models:
        pairs = [(int(row[model]), int(row["doctor_result"])) for row in rows]
        metrics = prognosis_metrics(pairs)
        print(f"{model:<8} {metrics['exact_accuracy']:>8.3f} {metrics['mae']:>7.3f} {metrics['within_1']:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
