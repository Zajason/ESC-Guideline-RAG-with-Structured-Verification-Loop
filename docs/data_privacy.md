# Data Privacy

This repository is designed to be public-safe, but the original research workflow used private clinical documents and workbooks.

Do not commit:

- patient DOCX files
- original Excel workbooks
- generated patient-level output workbooks
- raw batch logs
- API keys or `.env` files
- screenshots or exports containing patient identifiers

The `.gitignore` excludes these by default.

## Public Demonstration Data

Use files under `examples/` for public demos. These are synthetic and are not derived from a real patient.

## Research Disclaimer

This project is for retrospective research and benchmarking only. It is not a clinical decision support system and must not be used for real patient care.
