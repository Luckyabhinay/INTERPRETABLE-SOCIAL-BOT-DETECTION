from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from twis20.feature_schema import NETWORK_COLUMNS
from twis20.live_network.benchmark_graph import (
    GraphIntegrityError,
    GraphIntegrityPolicy,
    load_benchmark_graph,
)
from twis20.network_features import extract_network_features


def _write_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, str]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _policy(split_path: Path, edge_path: Path) -> GraphIntegrityPolicy:
    with split_path.open(encoding="utf-8", newline="") as stream:
        roster_counts = {key: 0 for key in ("train", "val", "test", "support")}
        for row in csv.DictReader(stream):
            roster_counts[row["split"]] += 1
    with edge_path.open(encoding="utf-8", newline="") as stream:
        edge_counts = {"friend": 0, "follow": 0, "post": 0}
        for row in csv.DictReader(stream):
            edge_counts[row["relation"]] += 1
    return GraphIntegrityPolicy(
        split_sha256=_sha256(split_path),
        edge_sha256=_sha256(edge_path),
        roster_counts=roster_counts,
        edge_counts=edge_counts,
    )


class BenchmarkGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.split_path = self.root / "split.csv"
        self.edge_path = self.root / "edge.csv"
        _write_csv(
            self.split_path,
            ["id", "split"],
            [
                {"id": "u1", "split": "train"},
                {"id": "u2", "split": "val"},
                {"id": "u3", "split": "test"},
                {"id": "u4", "split": "support"},
                {"id": "u5", "split": "support"},
            ],
        )
        _write_csv(
            self.edge_path,
            ["source_id", "relation", "target_id"],
            [
                {"source_id": "u1", "relation": "friend", "target_id": "u4"},
                {"source_id": "u1", "relation": "friend", "target_id": "u4"},
                {"source_id": "u4", "relation": "friend", "target_id": "u2"},
                {"source_id": "u1", "relation": "follow", "target_id": "u2"},
                {"source_id": "u3", "relation": "follow", "target_id": "u3"},
                {"source_id": "u1", "relation": "post", "target_id": "t0"},
            ],
        )
        self.policy = _policy(self.split_path, self.edge_path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _load(self):
        return load_benchmark_graph(
            self.split_path, self.edge_path, integrity_policy=self.policy
        )

    def test_feature_order_and_exact_phase3_reference_parity(self) -> None:
        graph = self._load()
        accounts = ["1", "2", "3"]
        splits = ["train", "dev", "test"]
        reference = extract_network_features(
            accounts, splits, self.split_path, self.edge_path
        )
        self.assertEqual(graph.get_network_features("1").feature_names, NETWORK_COLUMNS)
        for row, account_id in enumerate(accounts):
            result = graph.get_network_features(account_id)
            self.assertEqual(result.status, "available")
            self.assertEqual(result.values.dtype, np.float32)
            np.testing.assert_array_equal(
                result.values, reference.iloc[row].to_numpy(dtype=np.float32)
            )

    def test_directed_duplicates_self_loops_post_exclusion_and_support(self) -> None:
        graph = self._load()
        self.assertEqual(
            graph.get_network_features("1").values.tolist(),
            [0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 2.0, 0.0],
        )
        self.assertEqual(
            graph.get_network_features("2").values.tolist(),
            [1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0],
        )
        self.assertEqual(
            graph.get_network_features("3").values.tolist(),
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        )
        self.assertEqual(graph.get_network_features("4").provenance["roster_split"], "support")

    def test_genuine_isolate_is_distinct_from_absent_account(self) -> None:
        graph = self._load()
        isolate = graph.get_network_features("5")
        missing = graph.get_network_features("999")
        self.assertEqual(isolate.status, "available")
        self.assertEqual(isolate.values.tolist(), [0.0] * 7 + [1.0])
        self.assertEqual(missing.status, "account_not_in_benchmark_graph")
        self.assertIsNone(missing.values)
        self.assertEqual(missing.missing_feature_names, NETWORK_COLUMNS)

    def test_batch_preserves_order_duplicates_and_missing_status(self) -> None:
        graph = self._load()
        results = graph.get_network_features_batch(iter(["4", "999", "4"]))
        self.assertEqual(
            [result.status for result in results],
            ["available", "account_not_in_benchmark_graph", "available"],
        )
        np.testing.assert_array_equal(results[0].values, results[2].values)

    def test_rejects_malformed_rows_and_bad_relationship_endpoints(self) -> None:
        with self.edge_path.open("a", encoding="utf-8", newline="") as stream:
            stream.write("u1,friend,u999,extra\n")
        with self.assertRaisesRegex(GraphIntegrityError, "column count"):
            self._load()

        _write_csv(
            self.edge_path,
            ["source_id", "relation", "target_id"],
            [{"source_id": "u1", "relation": "friend", "target_id": "u999"}],
        )
        self.policy = _policy(self.split_path, self.edge_path)
        with self.assertRaisesRegex(GraphIntegrityError, "outside roster"):
            self._load()

    def test_rejects_incomplete_or_hash_mismatched_graph(self) -> None:
        self.edge_path.write_text(
            self.edge_path.read_text(encoding="utf-8").splitlines()[0] + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(GraphIntegrityError, "SHA-256|counts mismatch"):
            self._load()

    def test_nonofficial_graph_is_rejected_by_default(self) -> None:
        with self.assertRaisesRegex(GraphIntegrityError, "SHA-256"):
            load_benchmark_graph(self.split_path, self.edge_path)

    def test_rejects_invalid_account_id_format(self) -> None:
        with self.assertRaisesRegex(ValueError, "decimal ID strings"):
            self._load().get_network_features("u1")


if __name__ == "__main__":
    unittest.main()
