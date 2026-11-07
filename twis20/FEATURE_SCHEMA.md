# TwiBot-20 Feature Schema

This is the implemented ordered schema. It describes only observed source fields and deterministic derivatives. Feature extraction and preprocessing do not train models.

## Row key and target (kept outside features)

| Name | Source | Use |
|---|---|---|
| `account_id` | top-level `ID` | Sidecar key only; never a model feature |
| `split` | Official archive member (`train`, `dev`, `test`) | Sidecar/evaluation routing only |
| `label` | top-level `label` (`"0"` human, `"1"` bot) | Target only; never a feature |

Preserve IDs as strings. Do not join predictors by row position across files without an explicit ID check.

## Profile feature block (ordered)

### Numeric source metrics

Each raw numeric metric is parsed as finite numeric data. Output its value with the preprocessing described in [PREPROCESSING_PLAN.md](./PREPROCESSING_PLAN.md), followed by an explicit missing/invalid indicator:

1. `profile.followers_count`
2. `profile.friends_count`
3. `profile.statuses_count`
4. `profile.favourites_count`
5. `profile.listed_count`

### Boolean-like source flags

Emit true, false, and unknown one-hot indicators per field, in this order:

1. `profile.verified`
2. `profile.geo_enabled`
3. `profile.protected`
4. `profile.default_profile`
5. `profile.default_profile_image`
6. `profile.contributors_enabled`
7. `profile.has_extended_profile`
8. `profile.is_translation_enabled`
9. `profile.is_translator`
10. `profile.profile_background_tile`
11. `profile.profile_use_background_image`

Accepted true tokens: case-insensitive trimmed `true` and `1`; false tokens: `false` and `0`. Other nonempty tokens are invalid/unknown and must not silently map to false.

### Derived profile fields

In the following fixed order:

1. `profile.created_at_year` (parsed from the source date; no account-age calculation)
2. `profile.description_present`
3. `profile.description_char_count`
4. `profile.description_word_count`
5. `profile.description_hashtag_count`
6. `profile.description_mention_count`
7. `profile.description_url_count`

`description` counts use the same Unicode-aware `\w`-based hashtag/mention patterns and URL pattern specified for tweet text. Missing description has `description_present=0`, missing numeric derived counts, and matching missing indicators.

## Tweet/text feature block (ordered)

### Account-level numeric summaries

1. `tweet.list_available`
2. `tweet.item_count`
3. `tweet.valid_string_count`
4. `tweet.nonempty_count`
5. `tweet.total_char_count`
6. `tweet.mean_char_count`
7. `tweet.median_char_count`
8. `tweet.total_word_count`
9. `tweet.mean_word_count`
10. `tweet.median_word_count`
11. `tweet.unique_exact_text_count`
12. `tweet.exact_duplicate_fraction` = `(valid_string_count - unique_exact_text_count) / valid_string_count`, or 0 when there are no valid strings; pair with `tweet.list_available` and counts
13. `tweet.hashtag_count`
14. `tweet.mention_count`
15. `tweet.url_count`
16. `tweet.empty_or_invalid_item_count`

Counts are account aggregates, not engagement measurements. Empty strings are counted as empty; exact duplicate entries are counted and retained in the lexical document. Hashtags match `(?<!\w)#[\w]+`, mentions match `(?<![\w@])@[\w]+`, and URLs match `(?:https?://|www\.)\S+` case-insensitively. These are explicit pattern counts, not canonical platform parsing.

### Sparse lexical block

`tweet.tfidf.<token>` is a sparse word-TF-IDF block trained only on training-account documents. Fixed settings: Unicode-aware `(?u)\b\w\w+\b` word tokens, lowercase, word unigrams and bigrams, `min_df=2`, `max_features=50000`, `sublinear_tf=True`, L2 normalization. The analyzer generates n-grams within each tweet so a bigram cannot cross tweet boundaries. Repeated tweet occurrences remain repeated tokens. The exact output column names and order are the fitted vectorizer's `get_feature_names_out()` order, saved verbatim in the manifest.

## Network feature block (ordered)

Use distinct account IDs in `friend` and `follow` edges, with edge direction interpreted literally as `source_id -> target_id`. `post` is excluded.

1. `network.friend_in_degree`
2. `network.friend_out_degree`
3. `network.follow_in_degree`
4. `network.follow_out_degree`
5. `network.combined_in_degree` (distinct incoming neighbor IDs across both relations)
6. `network.combined_out_degree` (distinct outgoing neighbor IDs across both relations)
7. `network.combined_unique_neighbor_count` (distinct IDs in union of incoming/outgoing neighbors)
8. `network.isolate` (1 if feature 7 is zero, else 0)

Degree values are topology-only and include support-node neighbors. No label-derived or split-label composition fields are allowed. The graph coverage audit confirms every labeled and support account has a graph user node.

## Order, names, and combination

Each block follows the order above. For model matrices, use block order `profile`, then `tweet`, then `network`; concatenation retains each block's internal order. Dense numeric preprocessing outputs each source numeric value then its missing indicator in listed field order; Boolean flags follow; derived fields follow. The sparse TF-IDF names follow after tweet numeric summaries, in the saved vectorizer order. The implemented manifest must record the final exact matrix column names and order; duplicate names are an error.

The implementation is in `feature_schema.py`, `profile_features.py`, `text_features.py`, `network_features.py`, `preprocessing.py`, and `pipeline.py`. `extract_from_sources` reads the original archive plus the converted graph roster/edges; `save_feature_dataset` writes separate sparse group matrices, the fitted training-only preprocessor/vectorizer state, a metadata sidecar, and a manifest. Saving refuses to overwrite an existing destination. The manifest includes source paths and SHA-256 hashes, preprocessing configuration, package versions, vectorizer vocabulary/order, matrix shape/dtype, and exact feature-name order. IDs, split membership, labels, and overlap flags are sidecar metadata and never serialized into predictor matrices.

Run the automated checks from the repository root with `python -m unittest discover -s twis20\tests -v`. To create artifacts, use `python -m twis20.extract_features --output <new-directory>`; by default it reads the audited original archive and converted graph files in this folder. Extraction scans the full graph edge file and may take time. Do not select an existing output directory.

## Explicitly excluded inputs

- Target `label`, split as predictor, raw account identifiers, and label-derived graph statistics.
- Raw `profile.id`, `id_str`, `screen_name`, `name`, and identity-bearing URL strings.
- Raw location/profile-location strings, colors, image URLs, serialized `entities`, and high-cardinality identity text.
- Any presumed tweet like/reply/repost/retweet/impression/engagement attributes (not present in the inspected source).
- Graph `post` edges as network features and any inferred engagement value derived from edge counts.
