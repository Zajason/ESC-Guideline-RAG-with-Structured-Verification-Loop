# Contributing

Thank you for your interest in this research project.

## Ground Rules

- Do not submit patient data, screenshots with identifiers, or private clinical documents.
- Keep examples synthetic.
- Keep medical claims clearly separated from code and benchmark claims.
- Add or update tests for calibration, workbook code mapping, and metric logic when changing behavior.

## Local Checks

```bash
python -m unittest discover -s tests
python scripts/run_demo.py
```

## Pull Requests

Please include:

- summary of changes
- whether outputs or calibration metrics changed
- tests run
- privacy confirmation that no patient data was added
