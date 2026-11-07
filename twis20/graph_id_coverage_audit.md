# TwiBot-20 Graph User-ID Coverage Audit

Audit date: 2026-11-06<br>
Scope: exhaustive user-ID roster comparison and relationship-edge reference classification. No source dataset or archive was modified. No EDA, feature engineering, or model training was performed.

## Conclusion: PASS

Every expected account ID is represented exactly once as a user node in the converted graph. The graph contains all **11,826 labeled train/dev/test accounts** and all **217,754 unlabeled support accounts**. There are no duplicate user-node IDs, expected IDs missing from the graph, or unexpected graph user IDs.

## Sources and roster construction

The expected roster was constructed from:

- `TwiBot-20-Format22-003/Twibot-20/split.csv`: all train, val, test, and support IDs.
- `TwiBot-20-Format22-003/Twibot-20/label.csv`: the labeled account IDs.

The 11,826 IDs in `label.csv` exactly match the union of the train, val, and test IDs in `split.csv`. The remaining 217,754 IDs in `split.csv` have split `support` and no label entry. The three labeled split names correspond to the source package's train, dev (validation), and test splits.

| Roster group | Expected IDs |
|---|---:|
| Train | 8,278 |
| Validation (`val` / source `dev`) | 2,365 |
| Test | 1,183 |
| **Labeled total** | **11,826** |
| Unlabeled support | 217,754 |
| **Complete expected roster** | **229,580** |

## ID normalization and comparison

- IDs were read as strings; they were never converted to integers or floating-point values.
- Normalization removes surrounding whitespace only. The accepted observed account-ID form is a lowercase `u` followed by decimal digits.
- The normalized string is otherwise compared exactly. No case folding, numeric coercion, prefix rewriting, or removal of leading zeros is performed.
- Every graph user record had a valid ID in this form; **0 invalid user-ID records** were found.
- Duplicate detection and roster comparison were performed over the complete set of user-node IDs, not a sample.

## Graph coverage results

The full `node.json` array was streamed and parsed to EOF.

| Check | Result |
|---|---:|
| Total graph records (user + tweet nodes) | 33,717,772 |
| Total graph user-ID records | 229,580 |
| Unique graph user IDs | 229,580 |
| Duplicate user IDs | 0 |
| Expected IDs missing from graph | 0 |
| Unexpected graph user IDs | 0 |
| Labeled IDs covered | **11,826 / 11,826** |
| Unlabeled support IDs covered | **217,754 / 217,754** |

No missing or unexpected ID examples exist. Every graph user ID belongs to exactly one expected roster group.

## Support accounts versus missing labeled accounts

Support accounts are expected graph users, not unlabeled gaps in train/dev/test coverage. The graph has all 217,754 support user nodes as well as all 11,826 labeled user nodes.

The complete `edge.csv` was also scanned. It contains 117,110 `friend` and 110,869 `follow` edges. All 455,958 source/target endpoints on those user-relationship edges were classified against the expected roster; **none were unexpected**.

| Relationship | Endpoint side | Labeled endpoint occurrences | Support endpoint occurrences | Unexpected |
|---|---|---:|---:|---:|
| friend | source | 105,701 | 11,409 | 0 |
| friend | target | 14,238 | 102,872 | 0 |
| follow | source | 103,015 | 7,854 | 0 |
| follow | target | 5,510 | 105,359 | 0 |
| **Total** | **both sides** | **228,464** | **227,494** | **0** |

The source and target columns are directional edge endpoints; these occurrence totals are not distinct-account counts. Support endpoints are legitimate references to the support roster. They do not indicate missing labels for the official train/dev/test accounts.

## Checks executed

1. Read all rows from `split.csv` and `label.csv`; rejected duplicate or malformed roster IDs and verified labeled IDs equal the train/val/test union.
2. Stream-parsed every object in the 5.77 GB `node.json`; counted user records, validated IDs, tracked duplicate IDs, and compared the full user-ID set against the complete roster.
3. Stream-read all 33,716,171 rows in `edge.csv`; classified source and target IDs for every `friend` and `follow` edge as labeled, support, or unexpected.
4. Confirmed no invalid user IDs, missing roster IDs, duplicate user IDs, or unexpected graph IDs.

This audit verifies ID coverage and relationship endpoint membership. It does not assert that every tweet node or every edge is semantically correct, nor does it validate graph feature meaning.
