"""Memory-bounded Phase 2 EDA for the local TwiBot-20 package."""

from __future__ import annotations

import array
import csv
import hashlib
import json
import math
import re
import sqlite3
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ROOT = Path(__file__).resolve().parent
SOURCE_ZIP = ROOT / "TwiBot-20-20261106T091806Z-1-002" / "TwiBot-20" / "Twibot-20.zip"
GRAPH_DIR = ROOT / "TwiBot-20-Format22-003" / "Twibot-20"
OUTPUT_DIR = ROOT / "eda_figures"
SPLITS = ("train", "dev", "test")
SPLIT_FILES = {"train": "Twibot-20/train.json", "dev": "Twibot-20/dev.json", "test": "Twibot-20/test.json"}
SPLIT_BITS = {"train": 1, "dev": 2, "test": 4}
PROFILE_NUMERIC = (
    "followers_count",
    "friends_count",
    "statuses_count",
    "favourites_count",
    "listed_count",
)
PROFILE_FLAGS = (
    "verified",
    "geo_enabled",
    "protected",
    "default_profile",
    "default_profile_image",
)
TWEET_HASH_BATCH = 10_000
TEXT_PATTERNS = {
    "hashtags": re.compile(r"(?<!\w)#[\w]+", re.UNICODE),
    "mentions": re.compile(r"(?<![\w@])@[\w]+", re.UNICODE),
    "urls": re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE),
}

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 120


def stream_json_array(stream: TextIO) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    eof = False
    while True:
        if position > 1024 * 1024:
            buffer = buffer[position:]
            position = 0
        if not eof and len(buffer) - position < 2 * 1024 * 1024:
            chunk = stream.read(4 * 1024 * 1024)
            if chunk:
                buffer += chunk
            else:
                eof = True
        while position < len(buffer):
            while position < len(buffer) and (buffer[position].isspace() or buffer[position] in ",["):
                position += 1
            if position >= len(buffer) or buffer[position] == "]":
                break
            try:
                obj, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                break
            position = end
            if not isinstance(obj, dict):
                raise ValueError("Expected each source array item to be a JSON object")
            yield obj
        if eof and position >= len(buffer):
            return
        if eof and position < len(buffer) and buffer[position] == "]":
            return


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().casefold() in {"", "none", "null", "nan"}


def as_number(value: Any) -> float | None:
    if is_missing(value) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def parse_date(value: Any) -> datetime | None:
    if is_missing(value):
        return None
    for fmt in ("%a %b %d %H:%M:%S %z %Y", "%a %b %d %H:%M:%S %Y"):
        try:
            return datetime.strptime(str(value).strip(), fmt)
        except ValueError:
            pass
    return None


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "min": None, "p01": None, "median": None, "mean": None, "p99": None, "max": None}
    array_values = np.asarray(values, dtype=np.float64)
    q01, median, q99 = np.percentile(array_values, [1, 50, 99])
    return {
        "n": int(array_values.size),
        "min": float(array_values.min()),
        "p01": float(q01),
        "median": float(median),
        "mean": float(array_values.mean()),
        "p99": float(q99),
        "max": float(array_values.max()),
    }


def profile_missing_record(profile: dict[str, Any], key: str) -> bool:
    return key not in profile or is_missing(profile.get(key))


def normalize_boolean(value: Any) -> str:
    if is_missing(value):
        return "missing"
    text = str(value).strip().casefold()
    if text in {"true", "1"}:
        return "true"
    if text in {"false", "0"}:
        return "false"
    return "other"


def initialize_split_stats() -> dict[str, Any]:
    return {
        "accounts": 0,
        "labels": Counter(),
        "top_keys": Counter(),
        "profile_keys": Counter(),
        "profile_missing": Counter(),
        "profile_numeric": {key: [] for key in PROFILE_NUMERIC},
        "profile_invalid_numeric": Counter(),
        "profile_negative_numeric": Counter(),
        "profile_flags": {key: Counter() for key in PROFILE_FLAGS},
        "created_year": Counter(),
        "created_invalid": 0,
        "tweet_state": Counter(),
        "tweets_per_account": array.array("I"),
        "tweet_items": 0,
        "empty_tweet_texts": 0,
        "duplicate_within_account": 0,
        "tweet_chars": array.array("I"),
        "tweet_words": array.array("I"),
        "hashtags": array.array("I"),
        "mentions": array.array("I"),
        "urls": array.array("I"),
        "id_profile_mismatch": Counter(),
        "profile_signature_counts": Counter(),
        "profile_signature_splits": defaultdict(set),
        "tweet_engagement_keys": Counter(),
    }


def add_tweet_hashes(connection: sqlite3.Connection, rows: list[tuple[bytes, int, int, int, int]]) -> None:
    if not rows:
        return
    connection.executemany(
        """
        INSERT INTO tweet_hashes(hash, split_mask, train_count, dev_count, test_count)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(hash) DO UPDATE SET
            split_mask = tweet_hashes.split_mask | excluded.split_mask,
            train_count = tweet_hashes.train_count + excluded.train_count,
            dev_count = tweet_hashes.dev_count + excluded.dev_count,
            test_count = tweet_hashes.test_count + excluded.test_count
        """,
        rows,
    )


def analyze_official_splits() -> tuple[dict[str, Any], dict[str, Any]]:
    result = {split: initialize_split_stats() for split in SPLITS}
    id_sets = {split: set() for split in SPLITS}
    field_union = set()
    with tempfile.TemporaryDirectory(prefix="twibot20_eda_") as temp_dir:
        connection = sqlite3.connect(Path(temp_dir) / "tweet_hashes.sqlite")
        connection.execute(
            "CREATE TABLE tweet_hashes (hash BLOB PRIMARY KEY, split_mask INTEGER, "
            "train_count INTEGER, dev_count INTEGER, test_count INTEGER)"
        )
        connection.execute("PRAGMA journal_mode = OFF")
        connection.execute("PRAGMA synchronous = OFF")
        try:
            with zipfile.ZipFile(SOURCE_ZIP) as archive:
                for split in SPLITS:
                    stats = result[split]
                    batch: list[tuple[bytes, int, int, int, int]] = []
                    with archive.open(SPLIT_FILES[split]) as binary_stream:
                        import io

                        with io.TextIOWrapper(binary_stream, encoding="utf-8") as text_stream:
                            for record in stream_json_array(text_stream):
                                stats["accounts"] += 1
                                stats["top_keys"].update(record.keys())
                                uid = str(record.get("ID", "")).strip()
                                if not uid or uid in id_sets[split]:
                                    raise ValueError(f"Blank or duplicate account ID in {split}: {uid!r}")
                                id_sets[split].add(uid)
                                label = str(record.get("label"))
                                if label not in {"0", "1"}:
                                    raise ValueError(f"Unexpected raw label in {split}: {label!r}")
                                stats["labels"][label] += 1
                                profile = record.get("profile")
                                if not isinstance(profile, dict):
                                    profile = {}
                                for identity_key in ("id", "id_str"):
                                    identity_value = profile.get(identity_key)
                                    if is_missing(identity_value):
                                        stats["id_profile_mismatch"][f"{identity_key}:missing"] += 1
                                    elif str(identity_value).strip() != uid:
                                        stats["id_profile_mismatch"][f"{identity_key}:mismatch"] += 1
                                stats["profile_keys"].update(profile.keys())
                                field_union.update(profile.keys())

                                for key in profile:
                                    if profile_missing_record(profile, key):
                                        stats["profile_missing"][key] += 1
                                for key in PROFILE_NUMERIC:
                                    raw_value = profile.get(key)
                                    number = as_number(raw_value)
                                    if number is None:
                                        if not is_missing(raw_value):
                                            stats["profile_invalid_numeric"][key] += 1
                                    else:
                                        stats["profile_numeric"][key].append(number)
                                        if number < 0:
                                            stats["profile_negative_numeric"][key] += 1
                                for key in PROFILE_FLAGS:
                                    stats["profile_flags"][key][normalize_boolean(profile.get(key))] += 1

                                created = parse_date(profile.get("created_at"))
                                if created is None:
                                    stats["created_invalid"] += 1
                                else:
                                    stats["created_year"][created.year] += 1

                                profile_without_identity = {
                                    key: value
                                    for key, value in profile.items()
                                    if key not in {"id", "id_str", "screen_name"}
                                }
                                signature = hashlib.sha256(
                                    json.dumps(
                                        profile_without_identity,
                                        sort_keys=True,
                                        ensure_ascii=False,
                                        separators=(",", ":"),
                                    ).encode("utf-8")
                                ).digest()
                                stats["profile_signature_counts"][signature] += 1
                                stats["profile_signature_splits"][signature].add(split)

                                tweet_value = record.get("tweet")
                                if not isinstance(tweet_value, list):
                                    stats["tweet_state"]["not_list_or_missing"] += 1
                                    tweets: list[Any] = []
                                else:
                                    tweets = tweet_value
                                    stats["tweet_state"]["list"] += 1
                                    stats["tweets_per_account"].append(len(tweets))
                                    if len(tweets) == 0:
                                        stats["tweet_state"]["empty_list"] += 1

                                account_texts = set()
                                for tweet in tweets:
                                    if not isinstance(tweet, str):
                                        stats["tweet_state"]["non_string_items"] += 1
                                        continue
                                    if not tweet.strip():
                                        stats["empty_tweet_texts"] += 1
                                    stats["tweet_items"] += 1
                                    stats["tweet_chars"].append(len(tweet))
                                    stats["tweet_words"].append(len(tweet.split()))
                                    stats["hashtags"].append(len(TEXT_PATTERNS["hashtags"].findall(tweet)))
                                    stats["mentions"].append(len(TEXT_PATTERNS["mentions"].findall(tweet)))
                                    stats["urls"].append(len(TEXT_PATTERNS["urls"].findall(tweet)))
                                    digest = hashlib.sha256(tweet.encode("utf-8")).digest()
                                    batch.append(
                                        (
                                            digest,
                                            SPLIT_BITS[split],
                                            int(split == "train"),
                                            int(split == "dev"),
                                            int(split == "test"),
                                        )
                                    )
                                    if tweet in account_texts:
                                        stats["duplicate_within_account"] += 1
                                    account_texts.add(tweet)
                                    if len(batch) >= TWEET_HASH_BATCH:
                                        add_tweet_hashes(connection, batch)
                                        batch.clear()

                                for key, value in record.items():
                                    if key.casefold() in {"like_count", "reply_count", "retweet_count", "repost_count", "favorite_count", "favourites_count"}:
                                        stats["tweet_engagement_keys"][key] += 1
                                for tweet in tweets:
                                    if isinstance(tweet, dict):
                                        stats["tweet_engagement_keys"].update(
                                            key
                                            for key in tweet
                                            if key.casefold()
                                            in {
                                                "like_count",
                                                "reply_count",
                                                "retweet_count",
                                                "repost_count",
                                                "favorite_count",
                                                "favourites_count",
                                            }
                                        )
                    add_tweet_hashes(connection, batch)
                    connection.commit()
                    print(f"Scanned {split}: {stats['accounts']:,} labeled records", flush=True)

            masks = {
                "unique_texts": connection.execute("SELECT COUNT(*) FROM tweet_hashes").fetchone()[0],
                "shared_train_dev": connection.execute(
                    "SELECT COUNT(*) FROM tweet_hashes WHERE (split_mask & 3) = 3"
                ).fetchone()[0],
                "shared_train_test": connection.execute(
                    "SELECT COUNT(*) FROM tweet_hashes WHERE (split_mask & 5) = 5"
                ).fetchone()[0],
                "shared_dev_test": connection.execute(
                    "SELECT COUNT(*) FROM tweet_hashes WHERE (split_mask & 6) = 6"
                ).fetchone()[0],
                "shared_all_splits": connection.execute(
                    "SELECT COUNT(*) FROM tweet_hashes WHERE split_mask = 7"
                ).fetchone()[0],
                "unique_texts_by_split": {
                    split: connection.execute(
                        f"SELECT COUNT(*) FROM tweet_hashes WHERE (split_mask & {SPLIT_BITS[split]}) != 0"
                    ).fetchone()[0]
                    for split in SPLITS
                },
                "occurrences_by_split": {
                    split: connection.execute(f"SELECT COALESCE(SUM({split}_count), 0) FROM tweet_hashes").fetchone()[0]
                    for split in SPLITS
                },
            }
        finally:
            connection.close()

    all_ids = set().union(*id_sets.values())
    overlap = {
        f"{left}_{right}": len(id_sets[left] & id_sets[right])
        for i, left in enumerate(SPLITS)
        for right in SPLITS[i + 1 :]
    }
    if sum(len(ids) for ids in id_sets.values()) != len(all_ids):
        raise ValueError("Account IDs overlap across official splits")

    profile_signatures: dict[bytes, set[str]] = defaultdict(set)
    profile_counts: Counter[bytes] = Counter()
    for split in SPLITS:
        for signature, count in result[split]["profile_signature_counts"].items():
            profile_counts[signature] += count
            profile_signatures[signature].add(split)
    duplicate_profile_groups = sum(count > 1 for count in profile_counts.values())
    duplicate_profile_records = sum(count - 1 for count in profile_counts.values() if count > 1)
    shared_profile_groups = sum(
        len(profile_signatures[signature]) > 1
        for signature, count in profile_counts.items()
        if count > 1
    )
    extras = {
        "account_ids_by_split": {split: len(id_sets[split]) for split in SPLITS},
        "id_overlap_by_split_pair": overlap,
        "account_ids_unique_total": len(all_ids),
        "profile_schema_union": sorted(field_union),
        "exact_profile_duplicate_groups_excluding_identity": duplicate_profile_groups,
        "exact_profile_duplicate_records_excluding_identity": duplicate_profile_records,
        "exact_profile_duplicate_groups_cross_split_excluding_identity": shared_profile_groups,
        "tweet_content": masks,
    }
    return result, extras


def analyze_graph() -> dict[str, Any]:
    with (GRAPH_DIR / "split.csv").open(newline="", encoding="utf-8-sig") as file:
        split_map = {row["id"].strip(): row["split"].strip() for row in csv.DictReader(file)}
    with (GRAPH_DIR / "label.csv").open(newline="", encoding="utf-8-sig") as file:
        label_map = {row["id"].strip(): row["label"].strip() for row in csv.DictReader(file)}
    user_ids = list(split_map)
    index = {uid: i for i, uid in enumerate(user_ids)}
    parent = list(range(len(user_ids)))
    size = [1] * len(user_ids)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            return
        if size[root_a] < size[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        size[root_a] += size[root_b]

    neighbors: dict[str, set[str]] = defaultdict(set)
    out_degree = Counter()
    in_degree = Counter()
    relation_counts = Counter()
    relation_source_degree = {relation: Counter() for relation in ("friend", "follow")}
    relation_target_degree = {relation: Counter() for relation in ("friend", "follow")}
    cohort_edges = Counter()
    unique_directed_pairs: set[tuple[str, str]] = set()
    unique_relation_edges: set[tuple[str, str, str]] = set()
    duplicate_edges = 0
    self_loops = 0
    malformed_edges = 0
    cross_cohort_edges = 0
    cohort_names = {"train": "train", "val": "dev", "test": "test", "support": "support"}

    edge_path = GRAPH_DIR / "edge.csv"
    with edge_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["source_id", "relation", "target_id"]:
            raise ValueError(f"Unexpected edge schema: {reader.fieldnames!r}")
        for row in reader:
            source = row["source_id"].strip()
            relation = row["relation"].strip()
            target = row["target_id"].strip()
            if not source or not target or not relation:
                malformed_edges += 1
                continue
            relation_counts[relation] += 1
            if relation == "post":
                continue
            if source not in index or target not in index:
                raise ValueError(f"Relationship endpoint absent from split roster: {source!r}, {target!r}")
            relation_edge = (source, relation, target)
            if relation_edge in unique_relation_edges:
                duplicate_edges += 1
                continue
            unique_relation_edges.add(relation_edge)
            if source == target:
                self_loops += 1
            src_cohort = cohort_names[split_map[source]]
            tgt_cohort = cohort_names[split_map[target]]
            cohort_edges[(relation, src_cohort, tgt_cohort)] += 1
            if src_cohort != tgt_cohort:
                cross_cohort_edges += 1
            relation_source_degree[relation][source] += 1
            relation_target_degree[relation][target] += 1
            directed_pair = (source, target)
            if directed_pair not in unique_directed_pairs:
                unique_directed_pairs.add(directed_pair)
                out_degree[source] += 1
                in_degree[target] += 1
                neighbors[source].add(target)
                neighbors[target].add(source)
                union(index[source], index[target])

    reciprocal_pairs = sum((target, source) in unique_directed_pairs for source, target in unique_directed_pairs) // 2
    degrees = {uid: len(neighbors.get(uid, ())) for uid in user_ids}
    component_sizes = Counter(find(i) for i in range(len(user_ids)))
    component_size_distribution = Counter(component_sizes.values())
    cohort_degree: dict[str, list[int]] = {cohort: [] for cohort in ("train", "dev", "test", "support")}
    for uid, degree in degrees.items():
        cohort_degree[cohort_names[split_map[uid]]].append(degree)

    def degree_summary(values: list[int]) -> dict[str, Any]:
        array_values = np.asarray(values, dtype=np.int64)
        return {
            "nodes": int(array_values.size),
            "isolates": int(np.count_nonzero(array_values == 0)),
            "mean": float(array_values.mean()) if array_values.size else 0.0,
            "median": float(np.median(array_values)) if array_values.size else 0.0,
            "p95": float(np.percentile(array_values, 95)) if array_values.size else 0.0,
            "p99": float(np.percentile(array_values, 99)) if array_values.size else 0.0,
            "max": int(array_values.max()) if array_values.size else 0,
        }

    return {
        "roster_counts": dict(Counter(split_map.values())),
        "labels": dict(Counter(label_map.values())),
        "edge_rows": sum(relation_counts.values()),
        "relation_counts": dict(relation_counts),
        "malformed_edges": malformed_edges,
        "unique_relation_edges": len(unique_relation_edges),
        "duplicate_relation_edges": duplicate_edges,
        "self_loop_edges": self_loops,
        "unique_directed_user_pairs": len(unique_directed_pairs),
        "reciprocal_directed_pair_count": reciprocal_pairs,
        "cross_cohort_user_edges": cross_cohort_edges,
        "cohort_edge_matrix": {
            f"{relation}:{source}->{target}": count
            for (relation, source, target), count in sorted(cohort_edges.items())
        },
        "node_degree_all": degree_summary(list(degrees.values())),
        "node_degree_by_cohort": {
            cohort: degree_summary(values) for cohort, values in cohort_degree.items()
        },
        "in_degree_all": degree_summary([in_degree[uid] for uid in user_ids]),
        "out_degree_all": degree_summary([out_degree[uid] for uid in user_ids]),
        "relation_source_degree": {
            relation: degree_summary([counts[uid] for uid in user_ids])
            for relation, counts in relation_source_degree.items()
        },
        "relation_target_degree": {
            relation: degree_summary([counts[uid] for uid in user_ids])
            for relation, counts in relation_target_degree.items()
        },
        "component_count_including_isolates": len(component_sizes),
        "isolated_node_count": sum(degree == 0 for degree in degrees.values()),
        "component_size_counts": dict(sorted(Counter(component_sizes.values()).items())),
        "largest_component_sizes": sorted(component_sizes.values(), reverse=True)[:20],
        "unique_degree_by_cohort": cohort_degree,
        "unique_degree_by_user": degrees,
    }


def save_figures(split_data: dict[str, Any], extras: dict[str, Any], graph: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"train": "#3B82F6", "dev": "#F59E0B", "test": "#10B981"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    splits = list(SPLITS)
    labels = ["Human", "Bot"]
    values = np.asarray(
        [[split_data[s]["labels"].get("0", 0), split_data[s]["labels"].get("1", 0)] for s in splits]
    )
    bottom = np.zeros(len(splits))
    for j, label in enumerate(labels):
        axes[0].bar(splits, values[:, j], bottom=bottom, label=label, color=["#64748B", "#EF4444"][j])
        bottom += values[:, j]
    axes[0].set_title("Labeled account counts")
    axes[0].set_ylabel("Accounts")
    axes[0].legend()
    rates = values[:, 1] / values.sum(axis=1)
    axes[1].bar(splits, rates * 100, color=[colors[s] for s in splits])
    axes[1].set_title("Bot share by official split")
    axes[1].set_ylabel("Bot accounts (%)")
    axes[1].set_ylim(0, 100)
    for i, value in enumerate(rates * 100):
        axes[1].text(i, value + 1, f"{value:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_distribution.png", bbox_inches="tight")
    plt.close(fig)

    profile_keys = sorted(set().union(*(set(split_data[s]["profile_missing"]) for s in SPLITS),
                                       *(set(split_data[s]["profile_keys"]) for s in SPLITS)))
    profile_rows = []
    for key in profile_keys:
        total = sum(split_data[s]["accounts"] for s in SPLITS)
        missing = sum(
            split_data[s]["profile_missing"].get(key, 0)
            + split_data[s]["accounts"]
            - split_data[s]["profile_keys"].get(key, 0)
            for s in SPLITS
        )
        profile_rows.append((key, 100 * missing / total))
    profile_rows.sort(key=lambda row: row[1], reverse=True)
    fig, ax = plt.subplots(figsize=(11, max(7, len(profile_rows) * 0.23)))
    y = np.arange(len(profile_rows))
    ax.barh(y, [row[1] for row in profile_rows], color="#7C3AED")
    ax.set_yticks(y, [row[0] for row in profile_rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Missing or blank values (%)")
    ax.set_title("Raw profile-field missingness (all labeled splits)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "profile_missingness.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feature in zip(axes.flat, PROFILE_NUMERIC + ("created_year",)):
        if feature == "created_year":
            for split in SPLITS:
                counter = split_data[split]["created_year"]
                years = sorted(counter)
                ax.plot(years, [counter[y] for y in years], label=split, color=colors[split])
            ax.set_title("Account creation year")
            ax.set_xlabel("Year")
            ax.set_ylabel("Accounts")
        else:
            for split in SPLITS:
                values = np.asarray(split_data[split]["profile_numeric"][feature], dtype=np.float64)
                if values.size:
                    ax.hist(np.log10(np.maximum(values, 0) + 1), bins=35, alpha=0.45, density=True,
                            label=split, color=colors[split])
            ax.set_title(feature)
            ax.set_xlabel("log10(value + 1)")
            ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    fig.suptitle("Profile distributions by official split", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "profile_distributions.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for split in SPLITS:
        counts = np.asarray(split_data[split]["tweets_per_account"], dtype=np.int64)
        axes[0].hist(counts, bins=np.arange(-0.5, max(10, int(counts.max()) + 1.5), 5), alpha=0.45,
                     density=True, label=split, color=colors[split])
        chars = np.frombuffer(split_data[split]["tweet_chars"], dtype=np.uint32)
        if chars.size:
            axes[1].hist(chars, bins=40, alpha=0.45, density=True, label=split, color=colors[split])
        tags = np.frombuffer(split_data[split]["hashtags"], dtype=np.uint32)
        if tags.size:
            axes[2].hist(tags, bins=np.arange(-0.5, min(30, int(tags.max()) + 1.5), 1),
                         alpha=0.45, density=True, label=split, color=colors[split])
    axes[0].set_title("Tweets available per account")
    axes[0].set_xlabel("Tweet texts")
    axes[0].set_ylabel("Density")
    axes[1].set_title("Text length")
    axes[1].set_xlabel("Characters per text")
    axes[1].set_ylabel("Density")
    axes[2].set_title("Hashtags per text")
    axes[2].set_xlabel("Regex-matched hashtags")
    axes[2].set_ylabel("Density")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tweet_distributions.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    degree_by_cohort = graph["unique_degree_by_cohort"]
    for cohort, values in degree_by_cohort.items():
        if values:
            positive = np.asarray([value for value in values if value > 0], dtype=np.int64)
            if positive.size:
                bins = np.unique(np.geomspace(1, max(2, int(positive.max())), 35).astype(int))
                axes[0].hist(positive, bins=bins, alpha=0.45, density=True, label=cohort)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("Unique undirected user degree")
    axes[0].set_xlabel("Distinct neighbors (log scale; isolates omitted)")
    axes[0].set_ylabel("Density (log scale)")
    axes[0].legend(fontsize=8)

    component_counts = graph["component_size_counts"]
    nontrivial = sorted(((int(size), int(count)) for size, count in component_counts.items() if int(size) > 1))
    if nontrivial:
        comp_sizes = np.asarray([item[0] for item in nontrivial])
        comp_counts = np.asarray([item[1] for item in nontrivial])
        axes[1].scatter(comp_sizes, comp_counts, s=22, alpha=0.8)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Weak component size frequency")
    axes[1].set_xlabel("Component size (log scale)")
    axes[1].set_ylabel("Number of components (log scale)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "network_distributions.png", bbox_inches="tight")
    plt.close(fig)


def format_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def pct(n: int, total: int) -> str:
    return f"{100 * n / total:.2f}%" if total else "n/a"


def build_report(split_data: dict[str, Any], extras: dict[str, Any], graph: dict[str, Any]) -> str:
    split_names = {"train": "Train", "dev": "Dev / validation", "test": "Test"}
    class_rows = []
    for split in SPLITS:
        stats = split_data[split]
        humans = stats["labels"].get("0", 0)
        bots = stats["labels"].get("1", 0)
        class_rows.append([split_names[split], stats["accounts"], humans, bots, pct(bots, stats["accounts"])])
    class_rows.append(
        [
            "Total",
            sum(split_data[s]["accounts"] for s in SPLITS),
            sum(split_data[s]["labels"].get("0", 0) for s in SPLITS),
            sum(split_data[s]["labels"].get("1", 0) for s in SPLITS),
            pct(
                sum(split_data[s]["labels"].get("1", 0) for s in SPLITS),
                sum(split_data[s]["accounts"] for s in SPLITS),
            ),
        ]
    )
    profile_rows = []
    outlier_rows = []
    for feature in PROFILE_NUMERIC:
        row = [feature]
        for split in SPLITS:
            values = split_data[split]["profile_numeric"][feature]
            summary = summarize(values)
            missing = (
                split_data[split]["accounts"]
                - split_data[split]["profile_keys"].get(feature, 0)
                + split_data[split]["profile_missing"].get(feature, 0)
            )
            total = split_data[split]["accounts"]
            row.append(
                f"n={summary['n']}, med={summary['median']:.1f}, p95={np.percentile(values, 95):.1f}, "
                f"max={summary['max']:.1f}, missing={missing} ({pct(missing, total)}), "
                f"negative={split_data[split]['profile_negative_numeric'].get(feature, 0)}, "
                f"invalid={split_data[split]['profile_invalid_numeric'].get(feature, 0)}"
                if values
                else f"n=0, missing={total}"
            )
        profile_rows.append(row)
        for split in SPLITS:
            values = np.asarray(split_data[split]["profile_numeric"][feature], dtype=np.float64)
            if values.size:
                q1, q3 = np.percentile(values, [25, 75])
                spread = q3 - q1
                lower_fence = q1 - 1.5 * spread
                upper_fence = q3 + 1.5 * spread
                below = int(np.count_nonzero(values < lower_fence))
                above = int(np.count_nonzero(values > upper_fence))
                p99 = float(np.percentile(values, 99))
                maximum = float(values.max())
            else:
                below = above = 0
                p99 = maximum = float("nan")
            outlier_rows.append(
                [
                    feature,
                    split_names[split],
                    f"{p99:.1f}" if math.isfinite(p99) else "n/a",
                    f"{maximum:.1f}" if math.isfinite(maximum) else "n/a",
                    below,
                    above,
                    split_data[split]["profile_negative_numeric"].get(feature, 0),
                    split_data[split]["profile_invalid_numeric"].get(feature, 0),
                ]
            )
    flag_rows = []
    for key in PROFILE_FLAGS:
        flag_rows.append(
            [
                key,
                *[
                    f"T={split_data[s]['profile_flags'][key].get('true', 0)}, "
                    f"F={split_data[s]['profile_flags'][key].get('false', 0)}, "
                    f"missing={split_data[s]['profile_flags'][key].get('missing', 0)}, "
                    f"other={split_data[s]['profile_flags'][key].get('other', 0)}"
                    for s in SPLITS
                ],
            ]
        )
    missing_rows = []
    all_profile_keys = sorted(set().union(*(set(split_data[s]["profile_keys"]) for s in SPLITS)))
    for key in all_profile_keys:
        values = []
        total_missing = 0
        total_accounts = 0
        for split in SPLITS:
            count = split_data[split]["accounts"]
            absent_key = count - split_data[split]["profile_keys"].get(key, 0)
            missing = absent_key + split_data[split]["profile_missing"].get(key, 0)
            total_missing += missing
            total_accounts += count
            values.append(str(missing))
        missing_rows.append((key, values, total_missing, total_accounts))
    missing_rows.sort(key=lambda row: row[2], reverse=True)
    missing_table = format_table(
        ["Profile field", "Train missing", "Dev missing", "Test missing", "All missing (%)"],
        [
            [key, *values, f"{count} ({pct(count, total)})"]
            for key, values, count, total in missing_rows
        ],
    )
    tweet_rows = []
    for split in SPLITS:
        stats = split_data[split]
        per_account = np.frombuffer(stats["tweets_per_account"], dtype=np.uint32)
        tweet_lengths = np.frombuffer(stats["tweet_chars"], dtype=np.uint32)
        tweet_words = np.frombuffer(stats["tweet_words"], dtype=np.uint32)
        hashtags = np.frombuffer(stats["hashtags"], dtype=np.uint32)
        mentions = np.frombuffer(stats["mentions"], dtype=np.uint32)
        urls = np.frombuffer(stats["urls"], dtype=np.uint32)
        tweet_rows.append(
            [
                split_names[split],
                stats["accounts"],
                stats["tweet_state"].get("not_list_or_missing", 0),
                stats["tweet_state"].get("empty_list", 0),
                stats["tweet_items"],
                int(np.median(per_account)) if per_account.size else "n/a",
                f"{np.percentile(per_account, 95):.0f}" if per_account.size else "n/a",
                stats["empty_tweet_texts"],
                f"{np.median(tweet_lengths):.0f}" if tweet_lengths.size else "n/a",
                f"{np.median(tweet_words):.0f}" if tweet_words.size else "n/a",
                f"{np.median(hashtags):.0f}" if hashtags.size else "n/a",
                f"{np.median(mentions):.0f}" if mentions.size else "n/a",
                f"{np.median(urls):.0f}" if urls.size else "n/a",
                stats["duplicate_within_account"],
                stats["tweet_state"].get("non_string_items", 0),
            ]
        )
    network_rows = []
    for cohort in ("train", "dev", "test", "support"):
        summary = graph["node_degree_by_cohort"][cohort]
        network_rows.append(
            [cohort, summary["nodes"], summary["isolates"], f"{summary['isolates'] / summary['nodes']:.2%}",
             f"{summary['median']:.1f}", f"{summary['p95']:.1f}", summary["max"]]
        )
    component_rows = [
        [int(size), int(count)]
        for size, count in sorted(
            graph["component_size_counts"].items(), key=lambda item: int(item[1]), reverse=True
        )[:15]
    ]
    relation_rows = [
        [relation, count] for relation, count in sorted(graph["relation_counts"].items())
    ]
    edge_matrix_rows = []
    for key, count in sorted(graph["cohort_edge_matrix"].items()):
        relation, cohorts = key.split(":", 1)
        edge_matrix_rows.append([relation, cohorts, count])
    direction_rows = [
        [
            "in-degree",
            graph["in_degree_all"]["median"],
            graph["in_degree_all"]["p95"],
            graph["in_degree_all"]["max"],
            graph["in_degree_all"]["isolates"],
        ],
        [
            "out-degree",
            graph["out_degree_all"]["median"],
            graph["out_degree_all"]["p95"],
            graph["out_degree_all"]["max"],
            graph["out_degree_all"]["isolates"],
        ],
        *[
            [
                f"{relation} source",
                graph["relation_source_degree"][relation]["median"],
                graph["relation_source_degree"][relation]["p95"],
                graph["relation_source_degree"][relation]["max"],
                graph["relation_source_degree"][relation]["isolates"],
            ]
            for relation in ("friend", "follow")
        ],
        *[
            [
                f"{relation} target",
                graph["relation_target_degree"][relation]["median"],
                graph["relation_target_degree"][relation]["p95"],
                graph["relation_target_degree"][relation]["max"],
                graph["relation_target_degree"][relation]["isolates"],
            ]
            for relation in ("friend", "follow")
        ],
    ]

    profile_dist_table = format_table(
        ["Raw profile count", "Train", "Dev", "Test"],
        profile_rows,
    )
    flag_table = format_table(["Raw profile flag", "Train", "Dev", "Test"], flag_rows)
    tweet_table = format_table(
        [
            "Split",
            "Accounts",
            "Tweet missing/non-list",
            "Empty list",
            "Text items",
            "Tweets/list median",
            "Tweets/list p95",
            "Blank text items",
            "Chars/text median",
            "Words/text median",
            "Hashtags/text median",
            "Mentions/text median",
            "URLs/text median",
            "Repeated text within same account",
            "Non-string tweet items",
        ],
        tweet_rows,
    )
    network_table = format_table(
        ["Cohort", "Accounts", "Isolates", "Isolate share", "Median degree", "p95 degree", "Max degree"],
        network_rows,
    )
    return f"""# TwiBot-20 Exploratory Data Analysis — Phase 2

Audit date: 2026-11-06<br>
Scope: descriptive analysis only. No source dataset or archive was modified. No feature engineering or model training was performed.

## Executive summary

- Official labeled splits contain {sum(split_data[s]['accounts'] for s in SPLITS):,} accounts: {sum(split_data[s]['labels'].get('1', 0) for s in SPLITS):,} bots and {sum(split_data[s]['labels'].get('0', 0) for s in SPLITS):,} humans.
- Split account IDs are disjoint: {extras['account_ids_unique_total']:,} unique IDs across train, dev, and test.
- Source tweets are plain text, not structured tweet objects. No per-tweet like, reply, repost, or retweet metrics were present in the records inspected; engagement distributions therefore cannot be computed.
- The converted user graph contains {sum(graph['roster_counts'].values()):,} roster users and {graph['unique_directed_user_pairs']:,} unique directed user pairs across `friend`/`follow`. It has {graph['component_count_including_isolates']:,} weak components including isolates; {graph['node_degree_all']['isolates']:,} users have no friend/follow neighbor.
- Exact text hashes overlap between official splits (train-dev {extras['tweet_content']['shared_train_dev']:,}, train-test {extras['tweet_content']['shared_train_test']:,}, dev-test {extras['tweet_content']['shared_dev_test']:,} unique texts). Retweets/common text may explain overlap; exact-text overlap is a contamination risk to review, not proof of account-level leakage.

## Scope, files, and method

Inputs were read-only:

- Original split records: `TwiBot-20-20261106T091806Z-1-002/TwiBot-20/Twibot-20.zip` members `train.json`, `dev.json`, `test.json`.
- Converted graph: `TwiBot-20-Format22-003/Twibot-20/split.csv`, `label.csv`, and `edge.csv`.
- Phase 1 reports: [dataset_audit.md](./dataset_audit.md) and [graph_id_coverage_audit.md](./graph_id_coverage_audit.md).

The archive arrays were parsed incrementally from ZIP streams. Tweet hashes for cross-split duplicate checks were accumulated in a temporary SQLite database that was removed after analysis. The graph user roster was initialized from `split.csv`; relationship edges were scanned as CSV. Counts refer to actual data read in this run.

Assumptions and transformations:

- `dev` is the official validation split, equivalent to `val` in the converted split file.
- Labels are raw string `"0"` = human and `"1"` = bot, matching the bundled TwiBot-20 readme and the Phase 1 label reconciliation.
- Profile missingness means a missing key, JSON `null`, or a string whose stripped, case-insensitive value is empty, `"none"`, `"null"`, or `"nan"`. No record was removed or imputed.
- Numeric summaries parse only finite numeric values. Invalid nonempty numeric representations and negative counts are reported separately; values are not clipped or winsorized.
- Creation-date analysis uses year only. The collection timestamp is not established by the available files, so account age is not calculated.
- Tweet hashtag, mention, and URL counts use the explicit regular expressions in `run_eda.py` on each unmodified text string. They are descriptive pattern counts, not canonical extraction or engineered model features.
- Tweet-text equality uses SHA-256 of exact UTF-8 strings without case, URL, whitespace, or retweet normalization. Matching hashes are counted as exact-text overlaps; no text is discarded.
- Network degree counts distinct directed user pairs across `friend` and `follow`; undirected degree and weak components collapse relation type and reciprocal direction. Isolates include all roster accounts with no such neighbor. `post` edges are excluded from user-network topology.

## Class distribution

{format_table(["Official split", "Accounts", "Human (0)", "Bot (1)", "Bot share"], class_rows)}

![Class counts and bot share](./eda_figures/class_distribution.png)

Class counts differ in absolute size because the official splits differ in size. The observed bot share is shown explicitly; no resampling or split adjustment was applied.

## Profile data

The raw profile object exposes {len(extras['profile_schema_union'])} distinct fields across labeled records. Raw counters are shown, not transformed features. The fields below are counts exposed by the source schema; values are not standardized.

{profile_dist_table}

Distribution plots use `log10(max(value, 0) + 1)` for readability only; summary tables remain in original units. Negative or invalid raw values, if any, are not plotted as negative magnitudes (they remain represented in the negative/invalid counts in the reproducibility script).

### Numeric outlier screening

The following are descriptive Tukey 1.5-IQR fence counts, not data-removal rules. All records remain included:

{format_table(["Count field", "Split", "p99", "Maximum", "Below lower fence", "Above upper fence", "Negative values", "Invalid nonmissing"], outlier_rows)}

![Profile distributions](./eda_figures/profile_distributions.png)

Boolean-like fields are parsed only for descriptive counts (`True`/`1`, `False`/`0`, missing, or other):

{flag_table}

Creation year is shown instead of account age because a defensible API capture/reference date is unavailable.

![Profile missingness](./eda_figures/profile_missingness.png)

### Missingness by raw profile field

{missing_table}

Blank sentinel values occur in addition to JSON null/missing keys. The missingness plot and table do not fill any values. `profile_missingness.png` includes the profile-key union and treats absent keys as missing.

## Tweet availability and text

{tweet_table}

The raw tweet field consists of text strings. The source schema does not expose structured engagement counters; exact tweet-level like/reply/repost metrics are unavailable here. Accordingly, no engagement values are fabricated or inferred from labels or graph edge counts.

![Tweet availability and text distributions](./eda_figures/tweet_distributions.png)

Tweet list lengths are variable, and missing/non-list fields are distinguished from empty lists. Blank text strings and repeated text within a single account are counted, not removed.

### Exact text overlap and split contamination

| Check | Unique text hashes |
|---|---:|
| Unique exact texts in train | {extras['tweet_content']['unique_texts_by_split']['train']:,} |
| Unique exact texts in dev | {extras['tweet_content']['unique_texts_by_split']['dev']:,} |
| Unique exact texts in test | {extras['tweet_content']['unique_texts_by_split']['test']:,} |
| Train and dev | {extras['tweet_content']['shared_train_dev']:,} |
| Train and test | {extras['tweet_content']['shared_train_test']:,} |
| Dev and test | {extras['tweet_content']['shared_dev_test']:,} |
| Present in all three splits | {extras['tweet_content']['shared_all_splits']:,} |

These are text-level overlaps, not duplicate accounts. Identical posts can arise from retweets, shared announcements, or copied content. The counts flag a possible text-level contamination pathway for any later text-based evaluation; they do not prove that labels, accounts, or train/test records were copied. Split IDs themselves are disjoint. No deduplication was performed.

## Network structure

### Relationship edge types

{format_table(["Relation", "Rows in edge.csv"], relation_rows)}

### Degree and isolates

Degree is the number of distinct neighboring user IDs across the union of `friend` and `follow` directed pairs, treated as undirected for this table. It is not a follower-count profile metric.

{network_table}

- Unique directed user pairs: {graph['unique_directed_user_pairs']:,}; reciprocal directed pair pairs: {graph['reciprocal_directed_pair_count']:,}.
- Unique `friend`/`follow` relationship rows after duplicate exclusion: {graph['unique_relation_edges']:,}; exact duplicate relation rows: {graph['duplicate_relation_edges']:,}; self-loop relation rows: {graph['self_loop_edges']:,}.
- Cross-cohort relation edges (different train/dev/test/support cohort endpoints): {graph['cross_cohort_user_edges']:,}. These are transductive graph links, not direct label leakage, but their use in any later evaluation must be documented and consistent with the chosen protocol.
- Weakly connected components (all roster nodes, including isolates): {graph['component_count_including_isolates']:,}; largest component sizes: {', '.join(map(str, graph['largest_component_sizes'][:10]))}.
- Overall distinct directed-pair in-degree: median {graph['in_degree_all']['median']:.1f}, p95 {graph['in_degree_all']['p95']:.1f}, max {graph['in_degree_all']['max']}; out-degree: median {graph['out_degree_all']['median']:.1f}, p95 {graph['out_degree_all']['p95']:.1f}, max {graph['out_degree_all']['max']}.

![Network degree and component distributions](./eda_figures/network_distributions.png)

### Directed degree and split-cohort edge counts

Directed degrees count unique source-target pairs after collapsing `friend` and `follow` for combined in/out degrees. Relation-specific degree rows count distinct edges within that relation.

{format_table(["Degree", "Median", "p95", "Maximum", "Zero-degree nodes"], direction_rows)}

Cross-cohort rows below count literal relation edges, not unique user pairs after combining relation types:

{format_table(["Relation", "Source cohort → target cohort", "Edges"], edge_matrix_rows)}

### Weak component size frequency

The table lists the 15 most frequent component sizes; the full size-frequency distribution is included in the chart.

{format_table(["Component size", "Component count"], component_rows)}

All counts were obtained on the complete account roster. The graph visualization excludes isolates from the log-scaled degree histogram and excludes size-1 components from the plotted nontrivial component-frequency scatter; tables retain them.

## Duplicates, outliers, and inconsistencies

- **Account IDs:** no within-split duplicates; official split IDs are pairwise disjoint. A cross-split ID overlap would have stopped this run.
- **Profile duplicates:** exact profile signatures excluding `id`, `id_str`, and `screen_name` yielded {extras['exact_profile_duplicate_groups_excluding_identity']:,} duplicate groups and {extras['exact_profile_duplicate_records_excluding_identity']:,} extra records; {extras['exact_profile_duplicate_groups_cross_split_excluding_identity']:,} such groups span more than one split. Shared default/empty profile attributes can make signatures coincide, so this is a screening signal, not proof of duplicate accounts.
- **Profile numeric outliers:** count variables are heavy-tailed and displayed on a log scale. No rows were removed; distribution summaries use raw values. See `run_eda.py` for the scanned numeric domain and generated descriptive plots.
- **Tweet records:** list lengths vary, blank texts and exact repeats are counted, and text overlap across splits is quantified above. No tweets/accounts were removed.
- **Graph data:** edge schema was checked during full scan. Duplicate relation rows and self-loops are counted above. Relationship meaning is kept as the literal `friend`/`follow` label; direction semantics are not inferred beyond source/target columns.
- **Profile-to-ID consistency:** after trimming surrounding whitespace only, source account `ID` was compared with profile `id` and `id_str`. Results: {sum(split_data[s]['id_profile_mismatch'].get('id:missing', 0) for s in SPLITS):,} missing `id`, {sum(split_data[s]['id_profile_mismatch'].get('id:mismatch', 0) for s in SPLITS):,} mismatched `id`; {sum(split_data[s]['id_profile_mismatch'].get('id_str:missing', 0) for s in SPLITS):,} missing `id_str`, {sum(split_data[s]['id_profile_mismatch'].get('id_str:mismatch', 0) for s in SPLITS):,} mismatched `id_str`. Identity fields should still be avoided or justified separately.

## Leakage and split-contamination investigation

1. **Account-level overlap:** train/dev/test IDs are disjoint, with {extras['account_ids_unique_total']:,} total unique labeled IDs.
2. **Target field:** raw labeled records contain a top-level `label` key. It is the target and must be excluded from all predictor inputs. This EDA does not construct predictors.
3. **Identity fields:** account IDs, profile IDs, screen names, and usernames are identifiers, not safe predictors by default. Label association or collection-order artifacts could make them leakage channels; they require exclusion or a documented, justified policy before modeling.
4. **Graph leakage:** the converted graph includes all roster users and cross-cohort links. It has no evidence in this EDA of label values embedded in graph user-node features; labels are maintained in separate label/split files. However, graph-derived statistics over the full graph are transductive and may incorporate dev/test topology. Any future evaluation must state whether transductive graph access is permitted and must not propagate target labels from validation/test/support.
5. **Exact tweet-text overlap:** see the table above. Shared texts are present across splits and may inflate text-based generalization estimates even though account IDs are disjoint. Consider reporting a separate exact-text-grouped sensitivity evaluation in a future experimental design, without altering the official split-based primary result.
6. **No test tuning:** this phase is descriptive only; no model, threshold, or feature choice was selected using test performance.

## Limitations

- The source bundle does not provide per-tweet engagement counts, so engagement distributions cannot be reported.
- No trustworthy data-collection timestamp was identified; account age in days is not computed.
- Text counts are regex-based; they do not use a Twitter tokenizer, sentiment model, language identification, or URL expansion.
- Exact matching of text does not detect near-duplicates or semantically equivalent tweets.
- Graph degree/component results depend on the converted `friend`/`follow` edges; they do not establish the semantics, recency, or completeness of the underlying API relationships.
- This report compares official labeled splits. Support-account profiles are not mixed into labeled profile distributions; support is included in complete-roster graph topology.
- The Phase 1 audit records that the `.pt` file was not deserialized. It is not used in this EDA.

## Recommendations for Phase 3

1. Preserve official train/dev/test account membership and labels exactly; keep support accounts unlabeled.
2. Explicitly exclude the top-level target and identity fields from ordinary predictive inputs unless a leakage analysis justifies otherwise.
3. Handle blank/sentinel values explicitly, retain missingness indicators, and fit any learned imputation/encoding only on training data.
4. Decide and document whether the graph experiment is transductive. Report a profile-only baseline separately before adding tweet or graph-derived inputs.
5. Account for exact tweet overlap in the interpretation of text-based results; retain the official split as primary and treat any grouped sensitivity analysis as supplementary.
6. Do not synthesize tweet engagement metrics from text or graph edges. Design live-compatibility work only after measuring fields actually available from the intended API.
7. Keep this EDA descriptive: no features, preprocessing pipeline, model, or test-set tuning was created.

## Reproducibility

Run from the repository root with the configured Python environment:

```powershell
python .\\twis20\\run_eda.py
```

The script writes this report and PNGs under `twis20/eda_figures/`; it does not write into the dataset package or archives. Analysis details and parsing rules are in [run_eda.py](./run_eda.py).
"""


def main() -> None:
    if not SOURCE_ZIP.is_file():
        raise FileNotFoundError(SOURCE_ZIP)
    if not (GRAPH_DIR / "edge.csv").is_file():
        raise FileNotFoundError(GRAPH_DIR / "edge.csv")
    split_data, extras = analyze_official_splits()
    graph = analyze_graph()
    save_figures(split_data, extras, graph)
    report = build_report(split_data, extras, graph)
    (ROOT / "eda_report.md").write_text(report, encoding="utf-8")
    print(f"Wrote {ROOT / 'eda_report.md'}", flush=True)
    print(f"Wrote figures to {OUTPUT_DIR}", flush=True)
    print(
        json.dumps(
            {
                "splits": {
                    split: {
                        "accounts": split_data[split]["accounts"],
                        "labels": dict(split_data[split]["labels"]),
                        "tweet_items": split_data[split]["tweet_items"],
                    }
                    for split in SPLITS
                },
                "tweet_content_overlap": extras["tweet_content"],
                "graph": {
                    key: value
                    for key, value in graph.items()
                    if key not in {"unique_degree_by_cohort", "unique_degree_by_user"}
                },
            },
            indent=2,
            default=str,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
