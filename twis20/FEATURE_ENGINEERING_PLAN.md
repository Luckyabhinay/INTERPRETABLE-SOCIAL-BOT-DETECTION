# TwiBot-20 Phase 3 Feature Engineering Plan

## Status and scope

Implemented as feature extraction and preprocessing modules. No model has been trained. The original archives, official account assignments, and labels remain unchanged. The authoritative labeled records are `train.json`, `dev.json` (validation), and `test.json` in the original `Twibot-20.zip` archive.

The only allowed inputs are fields confirmed in the Phase 1 audit:

- Per-account `profile` fields in the labeled JSON records.
- `tweet`, a variable-length list of plain text strings.
- Account-level graph topology from the converted `edge.csv` (`friend` and `follow`) and roster from `split.csv`.

The raw `neighbor` lists are sampled/capped and are not the source of the network feature group. `label` is the target only. No per-tweet engagement counters are present; likes, replies, reposts/retweets, impressions, and engagement rates will not be inferred or fabricated.

## Independent feature groups

The groups are built separately and combined only in the experiments listed in [EVALUATION_PLAN.md](./EVALUATION_PLAN.md). Each account retains one row, in official split-file order. Account IDs and labels are stored in sidecar metadata, not in the feature matrix.

### 1. Profile

Use the five observed count fields `followers_count`, `friends_count`, `statuses_count`, `favourites_count`, and `listed_count`; parse numeric strings deliberately and preserve missing/invalid status. Use the observed Boolean-like fields `verified`, `geo_enabled`, `protected`, `default_profile`, `default_profile_image`, `contributors_enabled`, `has_extended_profile`, `is_translation_enabled`, `is_translator`, `profile_background_tile`, and `profile_use_background_image`, with explicit true/false/unknown parsing.

Derive `created_at_year` only from the observed profile creation date. Do not compute account age because no defensible collection/reference timestamp is provided. Include `description_present`, description character/word counts, and description hashtag/mention/URL counts as deterministic profile-description summaries. Do not include raw IDs, names, screen names, profile URLs, location strings, color codes, or serialized `entities` as model inputs. The complete raw profile field inventory and exclusions are recorded in [FEATURE_SCHEMA.md](./FEATURE_SCHEMA.md).

### 2. Tweet/text

For each account, aggregate its list-valued `tweet` strings in original order. Keep repeated entries, do not remove/merge duplicate tweets, and do not assume a fixed number of tweets. A missing/non-list field yields an empty document plus explicit availability/count features; a non-string item is treated as invalid/missing and counted, not stringified.

Create deterministic account-level count summaries (tweet count, nonempty count, total/mean/median character and word counts, exact unique-text count and duplicate fraction, and hashtag/mention/URL counts). In addition, fit a word TF-IDF representation on training-account documents only: lowercase, Unicode-aware word tokens, unigrams and bigrams within each tweet, `min_df=2`, `max_features=50,000`, sublinear term frequency, and L2 normalization. Tweet order and repeated occurrences are retained; n-grams do not cross tweet boundaries. Save the fitted vocabulary and ordered output names. This representation is subject to the overlap sensitivity protocol and does not establish that a repeated post was a retweet.

### 3. Network

Derive topology-only features from `friend` and `follow` rows in `edge.csv`, using source and target literally. Count distinct target neighbors (out) and source neighbors (in), separately for each relation and across the union of the two relations. Distinct counts avoid counting repeated rows twice; relation-specific and combined counts are kept separate. `post` edges are excluded.

The graph roster includes 11,826 labeled users and 217,754 unlabeled support users. Support nodes are included as unlabeled topology: they can contribute to degree counts and connectivity, but never receive labels, training rows, or target-derived aggregates. Do not use neighbor labels, label proportions, homophily, label propagation, supervised embeddings, or graph features derived from any split's labels.

The planned network experiment is **transductive**: all available `friend`/`follow` edges and roster membership, including dev/test endpoint topology, are visible when computing these topology-only features. This assumption must be stated with results. No labels from dev or test may enter feature construction or training. If the full topology is unavailable in a deployment setting, these results do not establish inductive performance.

## Leakage and reproducibility controls

- Train labels alone provide supervision. Validation labels are used only for model/threshold selection after train-only preprocessing; test labels are accessed only for the final locked evaluation.
- Exclude `ID`, profile `id`/`id_str`, `screen_name`, names, and other direct identity strings from predictors. Keep the official account ID in a separate keyed sidecar to join predictions and features.
- Do not derive predictors from labels, split-specific label aggregates, or test performance. Split membership is used only for row assignment, not as a feature.
- Fit learned preprocessing and TF-IDF vocabulary only on training rows. Do not fit on support, dev, or test.
- Fix and record normalization rules, source archive/member names, feature schema/version, package versions, feature-name order, and any random seed. Hash outputs and source files in the implementation phase.
- Preserve all account rows and tweet occurrences. Any explicitly defined sensitivity cohort is reporting-only and does not change official split membership.

## Deferred

The modules stop after feature extraction, train-only preprocessing, and optional feature artifact serialization. Model fitting, hyperparameter tuning, and test evaluation are out of scope and require separate approval.
