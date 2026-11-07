# Phase 6 — Robustness, Fairness and Generalization

## Integrity
The fixed Phase 4 XGBoost model, locked selection, official metrics, Phase 3 inputs and Phase 5 report had identical hashes before/after. No model fitting occurred and threshold remained 0.56.

## Overlap
`overlap_sensitivity.csv` reports all official-test overlap cohorts at the locked threshold. These are exposure diagnostics: copied/public text can affect benchmark performance but does not prove account-level leakage or memorization.

## Fixed-model comparison
`feature_ablation_fixed_models.csv` compares best stored Phase 4 candidates by feature configuration. This is a model comparison, not causal ablation; true removal needs retraining and is deferred.

## Missingness/noise
Only documented profile numeric stress/noise scenarios were executed on copies. Raw-text/profile missingness and network removal were unavailable without valid raw reconstruction and are not fabricated.

## Fairness and generalization
Demographic fairness cannot be established: no demographic/protected labels are available, and none were inferred. Generalization is limited by text dependence, verification contribution, text overlap sensitivity, low network contribution in Phase 5, transductive topology, and TwiBot-20 representativeness. Correlations and SHAP attributions are not causal claims.
