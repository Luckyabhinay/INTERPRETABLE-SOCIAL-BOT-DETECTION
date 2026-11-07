"""Topology-only features from the converted TwiBot-20 account graph."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_schema import NETWORK_COLUMNS

_GRAPH_ID = re.compile(r"^u([0-9]+)$")
_ACCOUNT_ID = re.compile(r"^[0-9]+$")
_EXPECTED_ROSTER_SPLITS = {"train", "val", "test", "support"}


def _graph_id(account_id: str) -> str:
    if not _ACCOUNT_ID.fullmatch(account_id):
        raise ValueError(
            f"Expected a numeric source account ID for graph join: {account_id!r}"
        )
    return f"u{account_id}"


def _read_roster(path: str | Path) -> dict[str, str]:
    roster: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"id", "split"} <= set(reader.fieldnames):
            raise ValueError("split.csv must contain id and split columns")
        for row_number, row in enumerate(reader, start=2):
            node_id = (row.get("id") or "").strip()
            split = (row.get("split") or "").strip()
            if not _GRAPH_ID.fullmatch(node_id):
                raise ValueError(f"Invalid graph ID in split.csv row {row_number}")
            if split not in _EXPECTED_ROSTER_SPLITS:
                raise ValueError(
                    f"Unexpected split in split.csv row {row_number}: {split!r}"
                )
            if node_id in roster:
                raise ValueError(f"Duplicate graph roster ID: {node_id}")
            roster[node_id] = split
    return roster


def extract_network_features(
    account_ids: Sequence[str],
    official_splits: Sequence[str],
    split_csv: str | Path,
    edge_csv: str | Path,
) -> pd.DataFrame:
    """Compute distinct-neighbor degrees; never reads labels or label.csv.

    This is intentionally transductive: all friend/follow edges across train,
    validation, test, and support accounts contribute to topology. Support
    nodes remain unlabeled and may only contribute as graph neighbors.
    """
    if len(account_ids) != len(official_splits):
        raise ValueError("Account ID and split lengths differ")
    if len(set(account_ids)) != len(account_ids):
        raise ValueError("Account IDs must be unique")
    if any(split not in {"train", "dev", "test"} for split in official_splits):
        raise ValueError("Official splits must be train, dev, or test")

    roster = _read_roster(split_csv)
    official_to_graph = {
        _graph_id(account_id): split
        for account_id, split in zip(account_ids, official_splits)
    }
    split_alias = {"train": "train", "dev": "val", "test": "test"}
    for account_id, split in zip(account_ids, official_splits):
        node_id = _graph_id(account_id)
        observed_split = roster.get(node_id)
        if observed_split != split_alias[split]:
            raise ValueError(
                f"Graph roster split mismatch for {account_id}: "
                f"official={split!r}, graph={observed_split!r}"
            )
    if not set(official_to_graph) <= set(roster):
        raise ValueError("One or more official accounts are absent from split.csv")
    if set(roster[node_id] for node_id in official_to_graph) != {
        split_alias[split] for split in official_splits
    }:
        raise AssertionError("Official split mapping validation failed")

    friend_in: defaultdict[str, set[str]] = defaultdict(set)
    friend_out: defaultdict[str, set[str]] = defaultdict(set)
    follow_in: defaultdict[str, set[str]] = defaultdict(set)
    follow_out: defaultdict[str, set[str]] = defaultdict(set)
    combined_in: defaultdict[str, set[str]] = defaultdict(set)
    combined_out: defaultdict[str, set[str]] = defaultdict(set)
    relation_counts = {"friend": 0, "follow": 0}

    with Path(edge_csv).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"source_id", "relation", "target_id"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError("edge.csv must contain source_id, relation, and target_id")
        for row_number, row in enumerate(reader, start=2):
            relation = (row.get("relation") or "").strip()
            if relation == "post":
                continue
            if relation not in {"friend", "follow"}:
                raise ValueError(
                    f"Unexpected graph relation at edge.csv row {row_number}: {relation!r}"
                )
            source = (row.get("source_id") or "").strip()
            target = (row.get("target_id") or "").strip()
            if source not in roster or target not in roster:
                raise ValueError(
                    f"Relationship endpoint outside split.csv roster at row {row_number}"
                )
            relation_counts[relation] += 1
            if relation == "friend":
                friend_out[source].add(target)
                friend_in[target].add(source)
            else:
                follow_out[source].add(target)
                follow_in[target].add(source)
            combined_out[source].add(target)
            combined_in[target].add(source)

    rows: list[dict[str, float]] = []
    for account_id in account_ids:
        node_id = _graph_id(account_id)
        unique_neighbors = combined_in[node_id] | combined_out[node_id]
        rows.append(
            {
                "network.friend_in_degree": float(len(friend_in[node_id])),
                "network.friend_out_degree": float(len(friend_out[node_id])),
                "network.follow_in_degree": float(len(follow_in[node_id])),
                "network.follow_out_degree": float(len(follow_out[node_id])),
                "network.combined_in_degree": float(len(combined_in[node_id])),
                "network.combined_out_degree": float(len(combined_out[node_id])),
                "network.combined_unique_neighbor_count": float(len(unique_neighbors)),
                "network.isolate": float(not unique_neighbors),
            }
        )
    frame = pd.DataFrame(rows, columns=NETWORK_COLUMNS, dtype=np.float64)
    if frame.columns.tolist() != list(NETWORK_COLUMNS):
        raise AssertionError("Network feature order does not match the schema")
    frame.attrs["transductive"] = True
    frame.attrs["relation_edge_counts"] = relation_counts
    frame.attrs["labels_used"] = False
    return frame


__all__ = ["NETWORK_COLUMNS", "extract_network_features"]
