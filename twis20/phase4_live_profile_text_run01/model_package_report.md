# TwiBot-20 live-compatible profile+text model

This deployment package reuses a completed Phase 4 `profile_text` candidate,
trained using the official train split and selected only by development
macro F1, PR-AUC, and duration. It was copied into a separate directory;
the locked run03 network-enabled model and its files were not changed.

## Selected candidate

- Candidate: `xgboost / depth_8`
- Development threshold: `0.49`
- Development macro F1: `0.855533`
- Development PR-AUC: `0.930755`
- Input features: `50072` (profile + tweet summary + TF-IDF)
- Network features: not used; no graph values are fabricated or required.

## Official test result

- Accuracy: `0.854607`
- Macro F1: `0.851200`
- ROC-AUC: `0.933494`
- PR-AUC: `0.937715`
- TN/FP/FN/TP: `416/127/45/595`

The test labels were loaded only after `locked_selection_before_test.json`
affirmed development-only selection and no test-metric use. These held-out
benchmark results do not guarantee live X performance; current live fields,
account availability, and domain shift may differ from the benchmark.
