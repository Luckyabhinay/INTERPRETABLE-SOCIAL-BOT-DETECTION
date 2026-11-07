# Phase 5 — SHAP Explainability Report

## Provenance
Explained the unchanged locked Phase 4 XGBoost candidate `profile_text_network__xgboost__depth_8.joblib`. The model has 50080 inputs and the reconstructed CSR matrix has 50080 ordered, unique manifest names. All Phase 3 hashes matched run03 provenance before explanation and remained unchanged after.

## Method
TreeExplainer used raw XGBoost margins. A reproducible stratified development sample of 200 accounts (seed 20261106) was explained. Full data remained CSR; only the 200-row explanation sample was converted to dense for plotting. Maximum additivity residual: 6.06e-06.

## Global findings
See `global_shap_importance.csv`, `top20_features.csv`, `top30_features.csv`, and `feature_group_importance.csv`. SHAP values describe this model's associations, not causal effects or proof that an account is a bot/genuine.

## Individual and error analysis
Individual output intentionally omits account IDs. Error analysis is descriptive on the already locked official test result at threshold 0.56; it did not tune any selection.

## Limitations
Text overlap can inflate benchmark performance for exposed cohorts; it does not by itself prove leakage or memorization. Sparse lexical features make individual text attributions high-dimensional and sensitive to correlated/repeated content. Network features are transductive topology-only signals.

## Validation
Model/input count, name uniqueness/order, CSR acceptance, finite SHAP values, additivity, non-negative global absolute importance, and immutable fingerprints passed.
