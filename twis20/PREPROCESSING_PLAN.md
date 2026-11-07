# TwiBot-20 Preprocessing Plan

## Scope

Implemented in `preprocessing.py` and orchestrated by `pipeline.py`. It preserves every official account row and train/dev/test assignment. No model is trained by these modules.

## Row identity and ordering

1. `records.py` streams the original archive's `train.json`, `dev.json`, and `test.json` as the authoritative source.
2. Keep `ID` as an opaque string in a separate sidecar. Never cast it to a number, use it as a feature, or normalize it beyond the documented source check.
3. Keep official split membership and row order separately. Before/after each transformation, assert one output row per input account and exact ID order equality.
4. Keep target labels in a separate target vector. Assert values are only `"0"`/`"1"`; never pass them into a feature transformer.
5. Support accounts are graph topology nodes only for the planned network block. They are not labeled training examples.

## Missing and invalid data

- Treat absent keys, JSON `null`, and trimmed case-insensitive empty/`none`/`null`/`nan` strings as missing, consistent with the dataset audit.
- For numeric profile counts, parse finite numeric values only. Non-numeric, nonfinite, and negative count values are invalid: map to missing, preserve a missing/invalid indicator, and report per-field counts. Do not clip or winsorize valid values.
- Fit a median imputer for each numeric feature using training rows only. Preserve an explicit missingness indicator even when the training column has no missing values. If an entire training feature is missing, retain a stable all-missing indicator and use a documented constant placeholder rather than dropping/reordering the column.
- Boolean fields accept only trimmed case-insensitive `true`/`1` and `false`/`0`. Missing/invalid values map to an explicit unknown state; never silently convert unknown to false.
- Parse `created_at` using the source format(s) verified by EDA. Invalid dates become missing plus an indicator. Do not compute age.
- Tweet missing/non-list values and invalid tweet items are counted separately from empty strings. An unavailable list yields an empty lexical document and explicit availability/count features.
- For description-derived and tweet-derived counts, distinguish missing source text from a valid empty string with a presence/availability indicator.

## Text and identifiers

- Preserve original tweet strings and list order. Do not normalize text by deleting punctuation, URLs, retweet markers, or repeated posts. Any normalization needed by the fixed vectorizer is specified in the feature schema.
- Keep account IDs, profile identity strings, split labels, and target labels outside the matrix. Join only via string IDs and assert one-to-one coverage.
- Do not include raw account IDs, `id_str`, screen names, profile names, or URLs as features. Text vocabulary is learned from training-account tweet documents only.

## Fit/transform boundaries

1. Split rows according to official membership before any learned transform is fit.
2. `FeaturePreprocessor` fits numeric medians on train only; the true/false/unknown mappings are fixed by the schema and are not learned from held-out data. Scaling is not performed.
3. Fit TF-IDF vocabulary and IDF weights on train-only account documents. Transform dev/test with the frozen training vocabulary; unseen tokens are ignored by that vectorizer.
4. Graph degree counts are deterministic transforms of the documented graph topology, not label-fitted transforms. The graph is transductive as specified in [FEATURE_ENGINEERING_PLAN.md](./FEATURE_ENGINEERING_PLAN.md).
5. Freeze and serialize all fitted transforms before validation/test transform. Validation can guide model selection; test is not used for fitting, feature selection, tuning, or threshold choice.

## Reproducibility and persisted artifacts (implementation phase)

Persist, without embedding IDs in feature columns:

- A feature manifest with schema version, exact output feature names/order, dtype, source field definitions, missing-value rules, vectorizer configuration/vocabulary, and transform versions.
- The fitted training-only preprocessor/vectorizer state.
- A separate account-ID/split/label sidecar with row order (labels restricted to target/evaluation use).
- Source archive/member identifiers and cryptographic hashes, code revision, dependency versions, and fixed random seeds for any stochastic operation.
- Audit checks: exact ID and row counts, no duplicate matrix columns, finite dense values after imputation, valid sparse matrix shape, no labels/identifiers in feature names or values by construction.

Feature matrices are generated only when `save_feature_dataset` is explicitly called with a new destination. The save operation keeps four feature blocks separate and writes IDs, labels, split membership, and overlap flags to `metadata.json`, never into matrices. It also writes `feature_manifest.json` with source hashes, package versions, imputation medians, graph protocol, and exact feature-name order. It refuses to overwrite an existing destination.
