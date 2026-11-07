# Stage B: benchmark graph provider

## Result

Implemented a read-only provider for the validated TwiBot-20 benchmark graph.
It is a standalone module and is not connected to arbitrary live X account
prediction. The provider accepts benchmark account IDs only; an ID absent from
the benchmark roster returns `account_not_in_benchmark_graph` with no feature
values. It never substitutes zeros for an absent account or a failed graph
load.

## Feature parity

The provider was loaded from the complete local graph and compared directly
with the saved Phase 3 `features_network.npz` values. Eight deterministic
benchmark accounts were selected: the first row in each official split and
the row attaining the maximum in each network feature column, with duplicate
row selections removed. The set covered train, dev, and test.

| Check | Result |
|---|---:|
| Phase 3 reference matrix | 11,826 x 8, float32 |
| Accounts directly compared | 8 |
| Individual feature values compared | 64 |
| Exact float32 array equality | **PASS** |
| Feature order | **PASS**; matches `NETWORK_COLUMNS` |
| Full graph load time | 84.787 seconds |

Compared metadata row indices and splits: `0/train`, `1/train`, `2/train`,
`1030/train`, `1225/train`, `8278/dev`, `10643/test`, and `10675/test`.
The original Phase 3 extractor was also used as the independent reference in
fixture tests.

## Graph semantics and integrity

- Keeps `friend` and `follow` as separate directed relations.
- Includes all train, validation, test, and unlabeled support roster nodes.
- Excludes `post` edges from features.
- Uses node-ID sets, matching Phase 3 duplicate-edge de-duplication.
- Retains self-loops in incoming and outgoing neighbor sets, matching Phase 3.
- Returns an all-zero degree vector with `network.isolate=1` only for an
  account present in the complete roster with no retained relationships.
- Validates exact source SHA-256 values from the Phase 3 feature manifest,
  roster split counts, edge relation counts, row structure, relation types,
  and friend/follow endpoint membership. Incomplete, altered, malformed, or
  unsupported graph data raises `GraphIntegrityError`.
- Reads no labels and writes no graph, feature, model, or split data.

Verified graph counts: 8,278 train, 2,365 validation, 1,183 test, 217,754
support nodes; 117,110 friend edges, 110,869 follow edges, and 33,488,192
excluded post edges.

| Input | SHA-256 |
|---|---|
| `split.csv` | `57bbab3c6c910caf2f5bd551b7fd6771cb60bfbef074d262f31dceb15986c73c` |
| `edge.csv` | `656c9751b9e0cd55244feb2229c026a904c24537013e43fbcb9cbc2e18ec44c9` |

## Tests

`python -m unittest twis20.live_network.test_benchmark_graph -v` passed all
8 tests, covering direct Phase 3 fixture parity, feature ordering, directed
relation semantics, duplicate rows, self-loops, post exclusion, support
nodes, isolates versus absent accounts, ordered/repeated batch lookups,
invalid IDs, malformed rows, invalid endpoints, and incomplete or unapproved
graph inputs.

## Preservation verification

The locked run03 model, Phase 3 preprocessing and manifest, and all snapshotted
Phase 1–6 artifact files were SHA-256 checked before and after Stage B. No
protected artifact hash or file inventory changed. No model was loaded for
prediction or training, and no Phase 1–6 output was written.
