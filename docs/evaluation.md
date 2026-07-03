# Evaluation

The latest internal workbook compared four model columns against the doctor/result prognosis column.

## Final Prognosis

| Model | Exact accuracy | MAE | RMSE | Within +/-1 | Bias |
|---|---:|---:|---:|---:|---:|
| CHAT | 48.0% | 0.85 | 1.33 | 77.3% | +0.05 |
| RAG | 41.3% | 0.77 | 1.08 | 82.7% | +0.40 |
| C+ | 26.7% | 1.35 | 1.76 | 60.0% | +1.11 |
| DEEP | 25.3% | 1.51 | 1.98 | 60.0% | +1.05 |

CHAT had the highest exact-category accuracy. RAG had the best ordinal error profile: lowest MAE, lowest RMSE, and highest within-one-category accuracy.

## High-Risk Recall

| Doctor category | CHAT recall | RAG recall |
|---|---:|---:|
| ICU risk / ICU outcome | 0/2 | 2/2 |
| Intubation | 1/3 | 2/3 |
| In-hospital death | 0/5 | 4/5 |

## Interpretation

The strongest claim is not that RAG wins every metric. The stronger and more defensible claim is:

> The ESC-guideline RAG system produces auditable, guideline-grounded answers, improves structured management decisions, beats DEEP and C+ on calibrated final prognosis, and is less wrong on ordinal prognosis even when CHAT has better exact-category accuracy.

## Reproducing Metrics

For the private research workbook:

```bash
python scripts/evaluate_models.py "../Final AI & HFrEF patients list_RAG_refilled_gpt54mini_rerun.xlsx"
```

For the public synthetic demo:

```bash
python scripts/run_demo.py
```
