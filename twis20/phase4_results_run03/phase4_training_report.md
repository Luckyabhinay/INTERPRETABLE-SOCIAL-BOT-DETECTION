# TwiBot-20 Phase 4 ? Training Report

## Protocol

- Fixed random seed: `20261106`.
- Supervised fitting used official training labels only.
- Development labels selected the model setting and probability threshold.
- The selection lock was saved before official test evaluation.
- Test labels were not used for model or threshold tuning.
- Network features use friend/follow topology, including unlabeled support accounts.
- Sparse matrices were retained during model fitting.
- All 50 candidate model checkpoints were saved.

## Locked selection

- Experiment: `profile_text_network`
- Algorithm / setting: `xgboost` / `depth_8`
- Development threshold: `0.56`
- Development macro F1: `0.855963`

## Official test evaluation

| experiment           | configuration          | algorithm   | setting   |   threshold |   accuracy |   precision_bot |   recall_bot |   f1_bot |   macro_f1 |   balanced_accuracy |   roc_auc |   pr_auc |   genuine_recall |   false_positive_rate |   tn |   fp |   fn |   tp |
|:---------------------|:-----------------------|:------------|:----------|------------:|-----------:|----------------:|-------------:|---------:|-----------:|--------------------:|----------:|---------:|-----------------:|----------------------:|-----:|-----:|-----:|-----:|
| profile_text_network | E_profile_text_network | xgboost     | depth_8   |        0.56 |    0.84869 |        0.828816 |     0.907812 | 0.866518 |   0.845942 |            0.843409 |  0.930257 | 0.933582 |         0.779006 |              0.220994 |  423 |  120 |   59 |  581 |

## Overlap-aware sensitivity

`overlap_aware_test_sensitivity.csv` partitions official test accounts without removing or reassigning them.

## Saved artifacts

- Model checkpoints: 50
- Candidate records: 50
- Development candidate rows: 50
- Completed candidate events: 50
