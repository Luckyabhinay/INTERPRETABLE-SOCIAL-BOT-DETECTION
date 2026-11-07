# Phase 5 — SHAP Explainability Report

## Provenance
Explained the locked Phase 4 candidate `profile_text_network__xgboost__depth_8.joblib` (SHA-256 `72d1a3374efebf032cf4cf84fe35b30e682e0c587e6eae7f13867bb692dc4df7`). The model and reconstructed CSR matrix each have 50080 ordered feature inputs. All five Phase 3 files matched Phase 4 run03 provenance; their hashes, the selected model, selection lock, and official test-metrics file were unchanged across this run.

## Method
TreeExplainer used raw XGBoost margins. A reproducible stratified development sample of 200 accounts (seed 20261106) was explained. Full data remained CSR; only the explanation sample was densified for plots. SHAP output shape: `[200, 50080]`. Maximum additivity residual: 6.06e-06. The locked development-selected probability threshold was 0.560000.

## Official Phase 4 test results
These are the existing locked-model metrics, not values used to tune this explanation run.

| accuracy | precision_bot | recall_bot | f1_bot   | roc_auc  | pr_auc   | tn  | fp  | fn | tp  |
| -------- | ------------- | ---------- | -------- | -------- | -------- | --- | --- | -- | --- |
| 0.848690 | 0.828816      | 0.907813   | 0.866518 | 0.930257 | 0.933582 | 423 | 120 | 59 | 581 |

## Top 20 global features
Ranked by mean absolute SHAP over the development explanation sample. SHAP indicates model contribution, not causality.

| feature                   | mean_abs_shap | feature_group |
| ------------------------- | ------------- | ------------- |
| profile.verified__true    | 1.971541      | profile       |
| profile.verified__false   | 0.404025      | profile       |
| tweet.median_char_count   | 0.269457      | tweet_summary |
| tweet.url_count           | 0.151094      | tweet_summary |
| tweet.tfidf.rt            | 0.112638      | tfidf         |
| tweet.tfidf.for           | 0.102513      | tfidf         |
| tweet.tfidf.the           | 0.094832      | tfidf         |
| tweet.mention_count       | 0.092804      | tweet_summary |
| tweet.hashtag_count       | 0.091125      | tweet_summary |
| profile.listed_count      | 0.083997      | profile       |
| profile.favourites_count  | 0.066196      | profile       |
| tweet.tfidf.http          | 0.063501      | tfidf         |
| profile.followers_count   | 0.060801      | profile       |
| tweet.tfidf.by            | 0.060500      | tfidf         |
| tweet.tfidf.and           | 0.059844      | tfidf         |
| profile.geo_enabled__true | 0.053263      | profile       |
| profile.friends_count     | 0.052491      | profile       |
| tweet.tfidf.to            | 0.052095      | tfidf         |
| tweet.tfidf.me            | 0.051017      | tfidf         |
| tweet.tfidf.there         | 0.050017      | tfidf         |

## Aggregate feature-group importance
Sum of per-feature mean absolute SHAP values over the same sample.

| feature_group | mean_abs_shap |
| ------------- | ------------- |
| tfidf         | 7.178991      |
| profile       | 2.891218      |
| tweet_summary | 0.761054      |
| network       | 0.042402      |

## False-positive and false-negative findings
Descriptive analysis of the already locked official test predictions; it did not change selection or threshold. Error totals cover the complete test split. Attribution summaries use a reproducible sample of up to 30 errors per type. Positive signed mean SHAP values move toward the bot class.

### False Positive (120)

Mean signed SHAP below is computed over 30 reproducibly sampled errors; positive values move toward the bot class.

| feature                 | mean_shap | mean_abs_shap |
| ----------------------- | --------- | ------------- |
| profile.verified__true  | 0.778523  | 0.778523      |
| profile.verified__false | 0.148356  | 0.148356      |
| tweet.median_char_count | -0.087569 | 0.087569      |
| tweet.tfidf.rt          | -0.042648 | 0.042648      |
| tweet.url_count         | -0.032588 | 0.032588      |
### False Negative (59)

Mean signed SHAP below is computed over 30 reproducibly sampled errors; positive values move toward the bot class.

| feature                 | mean_shap | mean_abs_shap |
| ----------------------- | --------- | ------------- |
| profile.verified__true  | 0.663843  | 0.663843      |
| tweet.tfidf.for         | -0.207098 | 0.207098      |
| tweet.median_char_count | -0.173926 | 0.173926      |
| tweet.tfidf.the         | -0.143579 | 0.143579      |
| profile.verified__false | 0.129428  | 0.129428      |

## Other outputs
Individual explanations omit account IDs. See `individual_explanations.csv`, `global_shap_importance.csv`, and `plots/` for per-row details and visualizations.

## Limitations
Text overlap can inflate benchmark performance for exposed cohorts; it does not by itself prove leakage or memorization. Sparse lexical features are high-dimensional, correlated, and may give unstable individual token attributions; mean absolute SHAP is specific to the sampled development cohort. Network features are transductive topology-only signals and are not causal evidence. Error-attribution averages use at most 30 randomly sampled errors per class and should be treated as descriptive, not a population explanation.

## Validation
Feature order and model input dimensions matched; matrices remained CSR; SHAP values were finite; raw-margin additivity met the 1e-4 tolerance; feature/global/group outputs were generated; and Phase 3, model, selection-lock, and test-metric hashes remained unchanged.
