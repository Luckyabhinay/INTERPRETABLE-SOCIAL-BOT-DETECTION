# TwiBot-20 live-account validation protocol

## Status and scope

The runner is ready, but no live evaluation has been conducted. Running one requires a user-provided 30–50-account registry with independently supported labels and a configured local X provider. Do not treat successful fetches, model scores, or the user's chosen sample as independent ground truth.

The runner calls the existing local `POST /twibot20/analyze` endpoint sequentially. The endpoint may use configured X API/XFetch credentials and may consume provider quota. The runner does not read, save, or print credentials; it saves usernames and submitted evidence references in the result CSV, so keep that file private. It does not save profile biographies, tweet text, or raw API responses.

## Registry

Start from [`live_account_registry_template.csv`](./live_account_registry_template.csv). Supply 30–50 unique accounts and complete every column:

- `record_id`: stable, non-identifying study row key.
- `username`: X handle or `x.com`/`twitter.com` profile URL.
- `reference_class`: `genuine_verified`, `bot_verified`, or `suspicious_unverified`.
- `reference_source`: the independent source or adjudication method.
- `evidence_reference`: a citation, study identifier, or private evidence locator sufficient for later review.
- `reviewer`: human reviewer identifier.
- `reviewed_at`: ISO-8601 review timestamp.

The runner validates syntax, uniqueness, class values, and completeness. It cannot verify that evidence is independent, current, or correctly interpreted. Only independently supported `genuine_verified` and `bot_verified` labels can enter error counts. Keep `suspicious_unverified` observations as a separate descriptive category; they are not ground truth and are excluded from false-positive/false-negative metrics. If labels are adjudicated by multiple reviewers, record the method and disagreement resolution in `reference_source`/`evidence_reference`.

## Reproducible execution

Start the local FastAPI service with its normal project instructions and configured credentials, then explicitly authorize live lookups:

```powershell
python -m twis20.live_validation `
  --registry "C:\private\study\live_accounts.csv" `
  --output "C:\private\study\live_results.csv" `
  --base-url http://127.0.0.1:8000 `
  --confirm-live-lookups
```

The runner writes a result CSV and adjacent `_summary.json`, refuses to overwrite either, records transport/HTTP/schema failures per account, and prints record IDs only. Failed requests remain visible and are excluded from successful-analysis metrics. No retry loop is used. API `probability` and `confidence` remain null; class scores are retained only as explicitly uncalibrated scores.

## Reported measures and limitations

Report registry size, successful analyses, failures, suspicious-unverified counts, and the confusion counts for successful independently labelled accounts. The CSV preserves all outcomes; the summary's confusion counts are not normalized metrics and must be accompanied by their denominator. Avoid presenting a small or convenience-sampled registry as an estimate of population accuracy. Do not claim population-level live accuracy or represent model scores as calibrated probabilities.

## Benchmark explainability

The dashboard's benchmark explainability panel reads the existing Phase 5 artifacts. It explicitly identifies the locked 50,080-feature profile+text+network benchmark model and the development split. It is not an explanation of the 50,072-feature live profile+text model. SHAP is a feature-contribution method, not causal evidence. No model or SHAP artifacts are recomputed by the dashboard.
