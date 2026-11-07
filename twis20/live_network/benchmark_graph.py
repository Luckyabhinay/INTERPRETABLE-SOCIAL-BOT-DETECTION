"""Read-only, integrity-checked lookup of the Phase 3 benchmark graph."""

from __future__ import annotations

import csv
import hashlib
import re
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from twis20.feature_schema import NETWORK_COLUMNS

_GRAPH_ID = re.compile(r"^u[0-9]+$")
_ACCOUNT_ID = re.compile(r"^[0-9]+$")
_VALID_SPLITS = {"train", "val", "test", "support"}
_DATA_ROOT = Path(__file__).resolve().parents[1] / "TwiBot-20-Format22-003" / "Twibot-20"


class GraphIntegrityError(RuntimeError):
    """Raised when the supplied graph is not the complete validated benchmark."""


@dataclass(frozen=True)
class GraphIntegrityPolicy:
    split_sha256: str
    edge_sha256: str
    roster_counts: dict[str, int]
    edge_counts: dict[str, int]


OFFICIAL_GRAPH_INTEGRITY = GraphIntegrityPolicy(
    split_sha256="57bbab3c6c910caf2f5bd551b7fd6771cb60bfbef074d262f31dceb15986c73c",
    edge_sha256="656c9751b9e0cd55244feb2229c026a904c24537013e43fbcb9cbc2e18ec44c9",
    roster_counts={"train": 8278, "val": 2365, "test": 1183, "support": 217754},
    edge_counts={"friend": 117110, "follow": 110869, "post": 33488192},
)


class _HashingLines(Iterator[str]):
    """Yield UTF-8 CSV lines while hashing their exact source bytes."""

    def __init__(self, path: Path):
        self.path = path
        self.stream = path.open("rb")
        self.digest = hashlib.sha256()
        self.first_line = True

    def __iter__(self) -> _HashingLines:
        return self

    def __next__(self) -> str:
        raw_line = self.stream.readline()
        if not raw_line:
            raise StopIteration
        self.digest.update(raw_line)
        encoding = "utf-8-sig" if self.first_line else "utf-8"
        self.first_line = False
        return raw_line.decode(encoding)

    def close(self) -> None:
        self.stream.close()


@dataclass(frozen=True)
class NetworkFeatureResult:
    status: str
    account_id: str
    feature_names: tuple[str, ...]
    values: np.ndarray | None
    provenance: dict
    missing_feature_names: tuple[str, ...] = ()


class BenchmarkGraph:
    """Complete directed friend/follow topology from the approved benchmark."""

    def __init__(
        self,
        roster: dict[str, str],
        friend_in: dict[str, set[str]],
        friend_out: dict[str, set[str]],
        follow_in: dict[str, set[str]],
        follow_out: dict[str, set[str]],
        combined_in: dict[str, set[str]],
        combined_out: dict[str, set[str]],
        provenance: dict,
    ) -> None:
        self._roster = roster
        self._friend_in = friend_in
        self._friend_out = friend_out
        self._follow_in = follow_in
        self._follow_out = follow_out
        self._combined_in = combined_in
        self._combined_out = combined_out
        self.provenance = provenance

    @staticmethod
    def _node(account_id: str) -> str:
        value = str(account_id)
        if not _ACCOUNT_ID.fullmatch(value):
            raise ValueError("Benchmark account IDs must be decimal ID strings")
        return f"u{value}"

    def get_network_features(self, account_id: str) -> NetworkFeatureResult:
        node = self._node(account_id)
        result_provenance = {**self.provenance, "graph_node_id": node}
        if node not in self._roster:
            return NetworkFeatureResult(
                status="account_not_in_benchmark_graph",
                account_id=account_id,
                feature_names=NETWORK_COLUMNS,
                values=None,
                provenance=result_provenance,
                missing_feature_names=NETWORK_COLUMNS,
            )

        neighbors = self._combined_in[node] | self._combined_out[node]
        values = np.asarray(
            [
                len(self._friend_in[node]),
                len(self._friend_out[node]),
                len(self._follow_in[node]),
                len(self._follow_out[node]),
                len(self._combined_in[node]),
                len(self._combined_out[node]),
                len(neighbors),
                float(not neighbors),
            ],
            dtype=np.float32,
        )
        result_provenance["roster_split"] = self._roster[node]
        return NetworkFeatureResult(
            status="available",
            account_id=account_id,
            feature_names=NETWORK_COLUMNS,
            values=values,
            provenance=result_provenance,
        )

    def get_network_features_batch(
        self, account_ids: Iterable[str]
    ) -> list[NetworkFeatureResult]:
        """Return one result per input, preserving order and repeated IDs."""
        return [self.get_network_features(account_id) for account_id in account_ids]


def _validate_row(row: dict[str | None, str | None], row_number: int, filename: str) -> None:
    if None in row or any(value is None for value in row.values()):
        raise GraphIntegrityError(
            f"Malformed {filename} row {row_number}: inconsistent CSV column count"
        )


def _read_roster(path: Path, policy: GraphIntegrityPolicy) -> tuple[dict[str, str], str]:
    roster: dict[str, str] = {}
    counts = {split: 0 for split in _VALID_SPLITS}
    lines = _HashingLines(path)
    try:
        reader = csv.DictReader(lines)
        if reader.fieldnames is None or not {"id", "split"} <= set(reader.fieldnames):
            raise GraphIntegrityError("split.csv must contain id and split columns")
        for row_number, row in enumerate(reader, start=2):
            _validate_row(row, row_number, "split.csv")
            node = (row["id"] or "").strip()
            split = (row["split"] or "").strip()
            if not _GRAPH_ID.fullmatch(node) or split not in _VALID_SPLITS:
                raise GraphIntegrityError(f"Invalid roster entry at split.csv row {row_number}")
            if node in roster:
                raise GraphIntegrityError(f"Duplicate graph roster ID at row {row_number}")
            roster[node] = split
            counts[split] += 1
    except (csv.Error, UnicodeError, OSError) as exc:
        raise GraphIntegrityError(f"Could not read complete split.csv: {exc}") from exc
    finally:
        lines.close()

    source_hash = lines.digest.hexdigest()
    if source_hash != policy.split_sha256:
        raise GraphIntegrityError("split.csv SHA-256 does not match the approved benchmark")
    if counts != policy.roster_counts:
        raise GraphIntegrityError(f"split.csv roster counts mismatch: {counts}")
    return roster, source_hash


def load_benchmark_graph(
    split_csv: str | Path | None = None,
    edge_csv: str | Path | None = None,
    *,
    integrity_policy: GraphIntegrityPolicy = OFFICIAL_GRAPH_INTEGRITY,
) -> BenchmarkGraph:
    """Load and verify the full graph once; no labels or external APIs are read."""
    if (split_csv is None) != (edge_csv is None):
        raise ValueError("Provide both split_csv and edge_csv, or neither")
    split_path = Path(split_csv) if split_csv is not None else _DATA_ROOT / "split.csv"
    edge_path = Path(edge_csv) if edge_csv is not None else _DATA_ROOT / "edge.csv"
    started = time.perf_counter()
    roster, split_hash = _read_roster(split_path, integrity_policy)

    friend_in: defaultdict[str, set[str]] = defaultdict(set)
    friend_out: defaultdict[str, set[str]] = defaultdict(set)
    follow_in: defaultdict[str, set[str]] = defaultdict(set)
    follow_out: defaultdict[str, set[str]] = defaultdict(set)
    combined_in: defaultdict[str, set[str]] = defaultdict(set)
    combined_out: defaultdict[str, set[str]] = defaultdict(set)
    counts = {"friend": 0, "follow": 0, "post": 0}
    lines = _HashingLines(edge_path)
    try:
        reader = csv.DictReader(lines)
        required = {"source_id", "relation", "target_id"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise GraphIntegrityError(
                "edge.csv must contain source_id, relation, and target_id columns"
            )
        for row_number, row in enumerate(reader, start=2):
            _validate_row(row, row_number, "edge.csv")
            source = (row["source_id"] or "").strip()
            relation = (row["relation"] or "").strip()
            target = (row["target_id"] or "").strip()
            if not source or not target or relation not in counts:
                raise GraphIntegrityError(f"Invalid edge.csv row {row_number}")
            counts[relation] += 1
            if relation == "post":
                continue
            if (
                not _GRAPH_ID.fullmatch(source)
                or not _GRAPH_ID.fullmatch(target)
                or source not in roster
                or target not in roster
            ):
                raise GraphIntegrityError(
                    f"Relationship endpoint outside roster at edge.csv row {row_number}"
                )
            if relation == "friend":
                friend_out[source].add(target)
                friend_in[target].add(source)
            else:
                follow_out[source].add(target)
                follow_in[target].add(source)
            combined_out[source].add(target)
            combined_in[target].add(source)
    except (csv.Error, UnicodeError, OSError) as exc:
        raise GraphIntegrityError(f"Could not read complete edge.csv: {exc}") from exc
    finally:
        lines.close()

    edge_hash = lines.digest.hexdigest()
    if edge_hash != integrity_policy.edge_sha256:
        raise GraphIntegrityError("edge.csv SHA-256 does not match the approved benchmark")
    if counts != integrity_policy.edge_counts:
        raise GraphIntegrityError(f"edge.csv relation counts mismatch: {counts}")

    return BenchmarkGraph(
        roster=roster,
        friend_in=friend_in,
        friend_out=friend_out,
        follow_in=follow_in,
        follow_out=follow_out,
        combined_in=combined_in,
        combined_out=combined_out,
        provenance={
            "graph_source": {
                "split_csv": str(split_path.resolve()),
                "edge_csv": str(edge_path.resolve()),
                "split_sha256": split_hash,
                "edge_sha256": edge_hash,
            },
            "feature_definition_version": "phase3-network-v1",
            "feature_names": NETWORK_COLUMNS,
            "directed": True,
            "duplicate_relationship_rows": "deduplicated per feature by node-ID sets",
            "self_loops": "retained in in-degree and out-degree sets",
            "post_edges_excluded": True,
            "support_nodes_included": True,
            "roster_counts": dict(integrity_policy.roster_counts),
            "relationship_rows": dict(counts),
            "labels_used": False,
            "graph_complete": True,
            "load_seconds": time.perf_counter() - started,
        },
    )


__all__ = [
    "BenchmarkGraph",
    "GraphIntegrityError",
    "GraphIntegrityPolicy",
    "NetworkFeatureResult",
    "OFFICIAL_GRAPH_INTEGRITY",
    "load_benchmark_graph",
]
