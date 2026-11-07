# TwiBot-20 Dataset Audit — Phase 1

Audit date: 2026-11-06<br>
Scope: local dataset verification only. No feature pipeline, EDA charts, or model training has been created.

## Executive summary

The local workspace contains the original JSON package, a converted graph-style dataset, a second archive containing both TwiBot-20 and TwiBot-22, and a processed-tensor archive. The primary labeled dataset is the original `Twibot-20.zip` package:

- Official files: `train.json`, `dev.json` (validation), `test.json`, and `support.json`.
- Labeled accounts: **11,826** — 8,278 train, 2,365 validation, and 1,183 test.
- Support accounts: **217,754**, unlabeled and not part of the labeled splits.
- The raw train/dev/test IDs match the corresponding labeled subsets in `split.csv`; raw labels also match `label.csv` by ID after interpreting the observed string values `"0"` and `"1"` as documented.
- `node.json` was streamed to EOF and contained **229,580 user nodes** and **33,488,192 tweet nodes**, with no duplicate node IDs observed. The exhaustive node-ID-to-roster comparison is complete; see [graph_id_coverage_audit.md](./graph_id_coverage_audit.md).
- `edge.csv` contains **33,716,171** well-formed rows, including one `post` edge per tweet node by count.
- All three local ZIP archives passed their ZIP CRC integrity checks. This verifies archive contents against their embedded CRCs, not the identity of the download source.

**Conclusion:** the labeled official splits are verified and should be preserved as provided. The converted graph has exact user-ID coverage of the full 229,580-account roster, including support accounts; see [graph_id_coverage_audit.md](./graph_id_coverage_audit.md) for the exhaustive comparison.

## Files and variants found

| Local file/package | Contents and audit disposition |
|---|---|
| `TwiBot-20-20261106T091806Z-1-002/TwiBot-20/Twibot-20.zip` | Primary source package. Contains `train.json`, `dev.json`, `test.json`, `support.json`, `readme.txt`, `LICENSE`, and seed-user text. The readme identifies the data as TwiBot-20, describes the fields, and maps label `1` to bot and `0` to human. |
| `TwiBot-20-Format22-003/Twibot-20/` | Converted graph-style files: `node.json`, `edge.csv`, `label.csv`, `split.csv`, `user_info.pt`, and `test.ipynb`. `node.json`, the CSV files, and notebook were inspected; `user_info.pt` was not deserialized. |
| `TwiBot-20-20261106T091806Z-1-004/TwiBot-20/data.zip` | Mixed archive with TwiBot-20 and TwiBot-22 subdirectories. Its TwiBot-20 `split_new.json` has train/dev/test lists but no support list; its `label_new.json` contains 11,826 labels plus an `"id": "label"` metadata entry. Do not mix its format or the TwiBot-22 files into the primary split without a separate reconciliation. |
| `TwiBot-20-20261106T091806Z-1-001/TwiBot-20/Twibot20_processed_data.zip` | Processed tensor package (`label.pt`, `edge_index.pt`, text/property tensors, and edge-type tensor). CRC-verified, but tensor semantics, feature ordering, and split provenance were not independently validated; it is not the Phase 1 source of truth. |

The original package readme states the record fields are user ID, profile, recent tweets, neighbors, domain, and label. It also says the support set supplies neighborhood information. The local readme includes an MIT license reference and the package includes a license file. No upstream cryptographic checksum or independent download provenance was present locally.

## Official split and label audit

The original archive calls the validation split `dev.json`; the converted metadata calls it `val`. These refer to the same validation split. Split membership was not recomputed or altered.

| Official split | Accounts | Bot (`"1"`) | Human (`"0"`) |
|---|---:|---:|---:|
| Train | 8,278 | 4,646 | 3,632 |
| Validation (`dev`) | 2,365 | 1,303 | 1,062 |
| Test | 1,183 | 640 | 543 |
| **Labeled total** | **11,826** | **6,589** | **5,237** |
| Support (unlabeled) | 217,754 | — | — |
| **All account IDs in `split.csv`** | **229,580** | — | — |

Checks performed:

- `label.csv`: 11,826 unique, nonblank IDs; values are `bot` and `human`.
- `split.csv`: 229,580 unique, nonblank IDs; split values are `train`, `val`, `test`, and `support`.
- Every labeled ID appears in `split.csv`; no support ID has a label in `label.csv`.
- The original train/dev/test IDs match the labeled rows for each split in `split.csv`.
- Original raw labels match `label.csv` for all 11,826 IDs. The raw JSON values are strings (`"0"`/`"1"`), despite the readme describing them numerically.
- `label_new.json` has the same 11,826 account labels as `label.csv`. Its `"id": "label"` entry is a metadata/header key, not another account.
- The newer `split_new.json` lists exactly 8,278 train, 2,365 dev, and 1,183 test IDs; it excludes the support roster represented in `split.csv`.

## Record schemas and available data

### Original labeled JSON records

Every labeled record has the top-level keys `ID`, `profile`, `tweet`, `neighbor`, `domain`, and `label`.

- **Profile:** 38 observed profile fields, including `followers_count`, `friends_count`, `statuses_count`, `favourites_count`, `listed_count`, `created_at`, `description`, `verified`, `geo_enabled`, and profile-style flags. Each of the 38 keys was present in every labeled profile, but many values are blank or sentinel strings.
- **Tweets:** `tweet` is a variable-length list of plain text strings, not structured tweet objects with like/reply/repost metrics. There are 1,999,788 tweet strings across labeled accounts. 80 accounts have no list-valued tweet field; observed list length is variable and reaches 201. This is not a uniform, guaranteed 200-tweet window.
- **Neighbors:** `neighbor`, when present, is a mapping with `following` and `follower` lists. These sampled lists contain 0–10 IDs each. The field is `None` for 1,087 labeled accounts.
- **Domain:** one or more category names from Politics, Business, Entertainment, and Sports.
- **Label:** string `"1"` for bot and `"0"` for human, consistent with the bundled readme and the converted labels.

### Converted graph-style files

- `node.json` is a JSON array with user nodes (`u...`) and tweet nodes (`t...`). A full streaming parse reached EOF: **229,580 users + 33,488,192 tweets**; no duplicate IDs or invalid JSON objects were observed.
- User-node top-level keys are `id`, `profile_image_url`, `name`, `url`, `username`, `created_at`, `pinned_tweet_id`, `protected`, `entities`, `location`, `withheld`, `verified`, `public_metrics`, and `description`.
- User `public_metrics` exposes `followers_count`, `following_count`, `tweet_count`, and `listed_count`. The graph-style tweet nodes expose only `id` and `text`; no per-tweet engagement counts were observed.
- Some converted values are serialized as strings with trailing whitespace; booleans and the `entities` structure are also represented as strings. Normalize deliberately and retain missingness rather than treating those strings as native booleans/objects or valid values.
- `edge.csv` fields are `source_id`, `relation`, and `target_id`. It has 33,716,171 nonblank, parseable rows: 33,488,192 `post`, 117,110 `friend`, and 110,869 `follow`. The `post` count equals the tweet-node count. No claim is made here that every edge endpoint or duplicate edge tuple was independently reconciled.

## Missing values and quality findings

| Area | Finding |
|---|---|
| Labeled account IDs and labels | No blank IDs/labels or duplicate IDs in `label.csv`; no duplicate IDs in `split.csv`; every labeled account has exactly one official split and label. |
| Profile values in raw labeled JSON | No explicit JSON `null` values were found in the 38 profile fields. Blank/`"None"`-style values occur and must be treated as missing: `utc_offset`, `time_zone`, and `lang` are missing for all 11,826 accounts; `profile_location` for 10,802; `url` for 6,038; `location` for 3,450; `description` for 1,523; and each of `profile_background_image_url` and `_https` for 3,082. |
| Tweet availability | 80 of 11,826 labeled records do not have a list-valued `tweet` field. Available text lists vary in length; do not assume all accounts have exactly 200 tweets. |
| Neighbor availability | 1,087 labeled accounts have `neighbor=None`; available neighbor lists are sampled and capped at 10 per direction. |
| Converted node profile | All user nodes have the same 14 top-level keys. `pinned_tweet_id` and `withheld` are explicit `null` for all users; 96 users have explicit `null` values in each of `profile_image_url`, `name`, `url`, `username`, `created_at`, `protected`, `entities`, `location`, `verified`, and `description`. Nested metric missingness has not been independently completed. Exact node-to-roster ID coverage is verified in [graph_id_coverage_audit.md](./graph_id_coverage_audit.md). |
| Edge table | All 33,716,171 rows had the expected three columns and nonblank values; no malformed rows were encountered. Duplicate edge tuples and full endpoint referential integrity were not audited. |
| Class distribution | The labeled data has more bots (6,589; 55.7%) than humans (5,237; 44.3%). Split-level counts are listed above; no class rebalancing was done. |

## Integrity and completeness checks

1. Read ZIP central directories and inspected member names and uncompressed sizes.
2. Ran ZIP CRC verification over every member of all three ZIP archives; each returned `OK`.
3. Parsed the original train/dev/test arrays to completion and checked IDs/labels against the converted split/label CSVs.
4. Read `label.csv` and `split.csv` completely; checked uniqueness, blanks, split categories, and label/split coverage.
5. Stream-parsed all of `node.json` and counted user/tweet nodes, malformed objects, and duplicate node IDs.
6. Stream-read all of `edge.csv` and counted rows, relation categories, and malformed/nonblank fields.

CRC success establishes that each archive is internally consistent with its stored CRCs. It does not establish an upstream publisher checksum or that different downloaded variants are interchangeable. The extracted converted files do not include a local manifest tying each file byte-for-byte to a particular archive member.

## Phase 1 decision

Use the original `Twibot-20.zip` train/dev/test labels and membership as the authoritative labeled split. Keep support IDs unlabeled. Converted graph user IDs match the full split roster; graph-derived features still require semantic and transformation review before use. Do not use the mixed TwiBot-20/TwiBot-22 archive or processed tensors as a replacement for the official splits.

This report intentionally stops before Phase 2 EDA and all downstream feature engineering/modeling.
