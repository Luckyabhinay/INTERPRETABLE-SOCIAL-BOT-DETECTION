# TwiBot-20 Phase 3 Network Feature Specification

## Scope and authority

This Stage A specification is a read-only reconstruction of the feature block
used by the locked Phase 4 run03 model. It is based on
`twis20/network_features.py`, `twis20/feature_schema.py`, the saved Phase 3
manifest, and `GRAPH_ID_COVERAGE_AUDIT.md`. No live-API relationship meaning is
inferred beyond what those artifacts establish.

Graph files are `TwiBot-20-Format22-003/Twibot-20/split.csv` and `edge.csv`.
The saved feature manifest records this block as CSR `float32` in the final
matrix; the extractor first creates a pandas `float64` frame and Phase 3 casts
it to CSR `float32` (`pipeline.py`, lines 164-180 and 245-250).

## Exact ordered features

The order is `NETWORK_COLUMNS` in `feature_schema.py` lines 88-97 and exactly
matches `network_features.py` lines 133-140:

| Position | Feature | Exact calculation for graph node `v` |
|---:|---|---|
| 1 | `network.friend_in_degree` | `|{u : (u, friend, v) in edge.csv}|` |
| 2 | `network.friend_out_degree` | `|{u : (v, friend, u) in edge.csv}|` |
| 3 | `network.follow_in_degree` | `|{u : (u, follow, v) in edge.csv}|` |
| 4 | `network.follow_out_degree` | `|{u : (v, follow, u) in edge.csv}|` |
| 5 | `network.combined_in_degree` | `|{u : (u, r, v), r in {friend, follow}}|` |
| 6 | `network.combined_out_degree` | `|{u : (v, r, u), r in {friend, follow}}|` |
| 7 | `network.combined_unique_neighbor_count` | `|combined_in(v) union combined_out(v)|` |
| 8 | `network.isolate` | `1.0` if feature 7 is zero, otherwise `0.0` |

Every set is a set of node IDs: duplicate relationship rows are de-duplicated
within each feature. The implementation uses `defaultdict(set)` for the six
directed sets (`network_features.py`, lines 90-95, 118-125). Self-loops are not
explicitly filtered: an input self-loop is retained once in its corresponding
in/out set and gives one unique neighbor ID. The audited graph contained only
roster endpoints, but this is not a claim that self-loops are absent.

## Graph interpretation and coverage

`edge.csv` rows are treated literally as directed `source_id -> target_id`
relationships (`FEATURE_SCHEMA.md`, Network feature block; implementation
lines 111-125). The code does **not** reverse `friend` or `follow`, symmetrize
them, infer an X-platform API meaning, or compute an undirected degree. `post`
edges are excluded (lines 103-106); any relation other than `friend`, `follow`,
or `post` is an error (lines 107-110).

The calculation is transductive: all `friend`/`follow` edges among `train`,
`val`, `test`, and `support` roster nodes contribute (lines 56-60 and 98-125).
It uses no labels, label compositions, propagation, or multi-hop statistics.
It is strictly one-hop degree/set aggregation. There is no normalization,
scaling, log transform, or learned graph transform.

`split.csv` contains 229,580 complete graph-user IDs: 8,278 train, 2,365 val,
1,183 test, and 217,754 unlabeled support nodes. Support nodes are retained as
ordinary topology endpoints; they never provide labels. This count and complete
roster coverage are established by `GRAPH_ID_COVERAGE_AUDIT.md`.

For an official account, Phase 3 maps numeric account ID `123` to graph ID
`u123` and validates its roster split (`network_features.py`, lines 21-26 and
69-88). An account present in the roster with no incoming/outgoing retained
relationships receives all degree values zero and `isolate=1.0`. An account
absent from the roster is an error, **not** an isolated zero-degree account.

## Reproducibility boundary

The original feature calculation can be reproduced exactly for a benchmark
account only by reading the complete roster and the complete `friend`/`follow`
edge set, retaining support endpoints, and applying the formulas above. A
partial live relationship sample is not semantically equivalent and must not be
used as this block. The existing X integrations fetch a profile and recent
tweets only; they provide no implemented relationship collector. Consequently,
Stage B may offer read-only benchmark lookups, but live accounts outside the
benchmark graph must return unavailable/incomplete network data.

## Evidence index

- Feature names/order: `twis20/feature_schema.py:88-97`; saved
  `phase3_features_phase4/feature_manifest.json` (`feature_names.network`).
- Directed set construction, duplicate handling, `post` exclusion and formulas:
  `twis20/network_features.py:90-145`.
- Account-ID/roster validation and transductive support-node use:
  `twis20/network_features.py:21-88`.
- Saved dtype/format: `twis20/pipeline.py:164-180,245-250`.
- Complete-roster/support counts and relationship-edge audit:
  `twis20/GRAPH_ID_COVERAGE_AUDIT.md`.
