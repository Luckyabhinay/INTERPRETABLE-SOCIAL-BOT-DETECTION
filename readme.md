# Interpretable Social Bot Detection

Research project for interpretable social bot detection, benchmark evaluation, and live account analysis. The repository keeps the Cresci-15 dashboard and the TwiBot-20 research workflow separate: they use different datasets, feature schemas, and model packages.

## What is included

- **Cresci-15 dashboard and API** — existing 34-feature analysis, explanations, and manual feature simulator.
- **TwiBot-20 research workflow** — dataset audits, feature extraction, preprocessing, locked model evaluation, explainability, and robustness analyses.
- **TwiBot-20 live dashboard** — profile-and-text scoring with a 50,072-feature model package. The live model does not use network features.
- **Research reports and validation tests** — dataset, feature, evaluation, explainability, and robustness documentation.

## Run the dashboard

Use Python with the dependencies in `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn app:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) for the TwiBot-20 dashboard. The Cresci-15 dashboard is available at [http://127.0.0.1:8000/cresci15](http://127.0.0.1:8000/cresci15).

Live account analysis requires either `XFETCH_API_KEY` or `X_BEARER_TOKEN`. Set credentials in your local environment; never put them in source files or commit them:

```powershell
$env:XFETCH_API_KEY = "your-key"
```

The application returns explicit service or feature-validation errors when live data or required features are unavailable. It does not fabricate network features.

## TwiBot-20 live model

The packaged live-compatible model uses **56 profile features, 16 tweet-summary features, and 50,000 TF-IDF features**. It was selected using development data and evaluated on the official held-out test split. See [the model package report](./twis20/phase4_live_profile_text_run01/model_package_report.md) for the recorded metrics and limitations.

The live dashboard displays both raw class scores as **uncalibrated model scores**. They are not calibrated probabilities or verified confidence. `probability` and `confidence` remain null in the API response.

The live-compatible model intentionally excludes graph/network features. Live scoring is experimental: current X accounts, available fields, and recent posts can differ substantially from the benchmark.

## Repository map

```text
app.py                              FastAPI routes and dashboard serving
twis20/
  dashboard.html                   TwiBot-20 dashboard
  *_audit.md                       Dataset and graph audit reports
  *_plan.md                        Feature, preprocessing, and evaluation plans
  live_network/                    Read-only benchmark graph provider
  phase3_features_phase4/           Reproducible Phase 3 feature artifacts
  phase4_live_profile_text_run01/   Live-compatible model package
  phase4_results_run03/             Locked Phase 4 experiment results
  phase5_explainability/            SHAP outputs and validation
  phase6_robustness/                Robustness outputs and validation
  tests/                            TwiBot-20 regression tests
```

Original benchmark archives, credentials, local backups, and generated scratch files are excluded from Git. Some training and audit workflows require the original datasets locally; consult [the dataset audit](./twis20/dataset_audit.md) for provenance and expected inputs. Do not modify official source archives or split assignments.

## API routes

| Route | Purpose |
|---|---|
| `GET /` | TwiBot-20 dashboard |
| `POST /twibot20/analyze` | TwiBot-20 live profile-and-text analysis |
| `GET /cresci15` | Existing Cresci-15 dashboard |
| `POST /analyze_x` | Existing Cresci-15 live analysis |
| `POST /predict_explain` | Existing Cresci-15 manual feature prediction |

## Validation

Run the TwiBot-20 tests:

```powershell
python -m pytest twis20/tests -q
```

Run the API and transport regression tests:

```powershell
python -m pytest test_xfetch_transport.py test_x_analysis.py test_xfetch_integration.py test_safe_live_prediction.py test_empirical_ood_suite.py -q
```

## Research integrity

- Official train/dev/test assignments and labels are preserved.
- The official test split is not used for model selection or threshold tuning.
- Preprocessing is fitted on training data and reused for inference.
- Account identifiers are kept separate from model features.
- Network features are not invented for live accounts.
- Benchmark scores do not establish accuracy on contemporary live accounts.
