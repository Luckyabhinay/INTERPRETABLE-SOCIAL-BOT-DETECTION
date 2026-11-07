# TwiBot-20 Evaluation Plan

## Evaluation boundaries

Use the official split assignments without moving, removing, or relabeling accounts:

- Train: 8,278 accounts.
- Validation (`dev`): 2,365 accounts.
- Test: 1,183 accounts.

Train labels are the only supervised fitting labels. Use validation labels for model and threshold selection. Lock all choices before evaluating once on test labels. Do not tune features, preprocessing, thresholds, or models from test performance. Report account-level results and the bot-positive convention (`label="1"`).

No experiment is run in this phase. Model training begins only after approval.

## Required independent experiments

Run each experiment using the same official account rows, train-only preprocessing rules, validation-based selection, and locked test protocol:

1. **Profile only** — the profile block in [feature_schema.md](./feature_schema.md).
2. **Text only** — tweet count summaries plus train-fitted tweet TF-IDF.
3. **Profile + text** — concatenate the two independent blocks.
4. **Profile + network** — concatenate profile and topology-only network blocks.
5. **Profile + text + network** — concatenate all blocks.

Report per-experiment validation and test metrics, confusion matrix, class support, and uncertainty estimates computed without test-based selection. Use the same metric definitions and threshold policy across experiments; any threshold is selected on validation only. Report at minimum accuracy, precision, recall, F1, ROC-AUC, and PR-AUC, with the positive class explicitly defined as bot.

## Network protocol

The planned network block uses all available graph `friend`/`follow` topology, including dev/test users and unlabeled support endpoints. This is a transductive assumption; label access remains restricted. Do not train on support labels (none are provided), propagate any target label, or derive neighborhood label statistics. Explicitly identify this setting in every network result; do not present it as inductive/deployment-realistic without a separate approved protocol.

## Text-overlap sensitivity

Keep the official test split intact and use the exact-text audit inventory in [text_overlap_audit.md](./text_overlap_audit.md) to define a predeclared diagnostic. `text_features.overlap_exposure` prepares exact and conservative-normalized exposure flags as sidecar columns for every row; train rows are marked not applicable. It does not filter or reorder data.

- For validation and, after all choices are frozen, test accounts, use `text_overlap_exact_train` to mark whether at least one of their exact tweet strings occurs in a training account's tweets.
- Use `text_overlap_normalized_train` for the separate normalized-exposure flag using the audit's conservative normalization.
- Report the frozen model's metrics for exposed and unexposed cohorts, along with cohort sizes and (for final reporting only) class support. No account is reassigned or silently dropped; subgroup metrics are supplementary and do not replace the all-test primary result.
- Do not choose models, thresholds, or features based on the test subgroup results. The test subgroup computation is post-hoc diagnostics only, using text content but not test labels to form the groups.
- If an additional training-text-masking sensitivity is desired, it must be a separately named and reported variant with explicit counts and must not alter official membership; obtain approval before adding it.

Exact and normalized text collisions do not prove that the same author posted the same content, that a post is a retweet, or that labels leaked. Interpret cohort differences as evidence about content exposure, not causal attribution.

## Reporting

Record source hashes, feature schema/order, preprocessing state, model/configuration, random seeds, validation selection procedure, test access point, and overlap-cohort definitions. Report whether each result uses profile, text, network, or combinations; state that no engagement features exist in these records. Preserve the official split-based all-test result as primary.

## Phase 4 runner safeguards

The explicit Phase 4 runner is `train_phase4.py`; invoking it starts model fitting and is outside feature extraction and validation. When separately approved, direct it to a new, unused output directory. It refuses to overwrite an existing destination and does not resume incomplete runs. LightGBM uses one thread by default; `--lightgbm-n-jobs` accepts only `1` or `2`. Random Forest and XGBoost are bounded to one thread. The LightGBM candidate set is capped at 100 estimators/15 leaves and 200 estimators/31 leaves.

Each candidate writes a start record, then persists its final status, configuration, timestamps, duration, error (if any), and development metrics immediately. Successful candidate models are saved independently; the selected model is loaded only after development-only selection. `candidate_events.jsonl` and `run_state.json` record progress and interruption status. Input matrices remain CSR sparse matrices. A candidate interrupted while fitting is marked `interrupted`, and subsequent candidates are not started. These checkpoints are for auditability, not automatic resume.

The runner reads train and development labels for fitting and selection only. It must persist and validate a selection lock affirming `selection_split="dev"` and `test_metrics_used=false` before its sole test-label access. Do not edit the official splits or labels, use test results for selection, or target existing Phase 4 output directories.

## Live-compatible profile+text inference

The live route uses the separate package in `phase4_live_profile_text_run01`,
not the locked run03 profile+text+network model. The package reuses the already
completed `profile_text` candidates from run03, selects among those ten
candidates using development macro F1, PR-AUC, then duration, and stores the
selected XGBoost candidate with the matching train-fitted Phase 3 preprocessor.
It does not refit a new model. A package-specific selection lock was written
before the official test metrics were computed. The live input order is exactly
the Phase 3 profile, text-summary, and TF-IDF blocks (50,072 features); no
network feature is filled, approximated, or required by this model.

This makes scoring possible for arbitrary live accounts with available profile
and tweet data, but it does not remove benchmark-to-live distribution shift.
The benchmark test metrics in `phase4_live_profile_text_run01/model_package_report.md`
are not a guarantee of live performance. Preserve the full run03 model and its
network-enabled results as a separate benchmark experiment.
