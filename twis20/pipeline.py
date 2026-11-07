"""End-to-end TwiBot-20 feature extraction and train-only preprocessing."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse

from .feature_schema import (
    FEATURE_GROUP_ORDER,
    SCHEMA_VERSION,
    feature_schema_manifest,
)
from .network_features import extract_network_features
from .preprocessing import FeaturePreprocessor
from .profile_features import extract_profile_features
from .records import AccountRecord, iter_official_records
from .text_features import extract_text_features, overlap_exposure

EXPERIMENTS = {
    "profile_only": ("profile",),
    "text_only": ("text_summary", "text_tfidf"),
    "profile_text": ("profile", "text_summary", "text_tfidf"),
    "profile_network": ("profile", "network"),
    "profile_text_network": (
        "profile",
        "text_summary",
        "text_tfidf",
        "network",
    ),
}


@dataclass(frozen=True)
class PreparedFeatureDataset:
    """Feature matrices and separate target/identity/evaluation sidecars."""

    account_ids: tuple[str, ...]
    splits: tuple[str, ...]
    labels: tuple[str, ...]
    features: dict[str, sparse.csr_matrix]
    feature_names: dict[str, tuple[str, ...]]
    overlap_exposure: pd.DataFrame
    manifest: dict[str, Any]
    preprocessor: FeaturePreprocessor

    def split_indices(self, split: str) -> np.ndarray:
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"Unknown official split: {split!r}")
        return np.fromiter(
            (index for index, value in enumerate(self.splits) if value == split),
            dtype=np.int64,
        )

    def experiment_matrix(self, experiment: str) -> sparse.csr_matrix:
        if experiment not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment: {experiment!r}")
        names = EXPERIMENTS[experiment]
        return sparse.hstack(
            [self.features[name] for name in names],
            format="csr",
            dtype=np.float32,
        )

    def experiment_feature_names(self, experiment: str) -> tuple[str, ...]:
        if experiment not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment: {experiment!r}")
        return tuple(
            name
            for group in EXPERIMENTS[experiment]
            for name in self.feature_names[group]
        )

    def metadata_frame(self) -> pd.DataFrame:
        """Return IDs, targets, split membership, and overlap flags separately."""
        metadata = pd.DataFrame(
            {
                "account_id": self.account_ids,
                "split": self.splits,
                "label": self.labels,
            }
        )
        return pd.concat(
            [metadata, self.overlap_exposure.reset_index(drop=True)],
            axis=1,
        )


def _validate_records(records: list[AccountRecord]) -> None:
    if not records:
        raise ValueError("No official account records were loaded")
    seen: set[str] = set()
    counts = {"train": 0, "dev": 0, "test": 0}
    for record in records:
        if record.split not in counts:
            raise ValueError(f"Unexpected official split: {record.split!r}")
        counts[record.split] += 1
        if record.account_id in seen:
            raise ValueError(f"Duplicate account ID: {record.account_id}")
        seen.add(record.account_id)
        if record.label not in {"0", "1"}:
            raise ValueError(
                f"Invalid label for account {record.account_id}: {record.label!r}"
            )
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"Every official split must be represented: {counts}")


def build_feature_dataset(
    records: list[AccountRecord],
    split_csv: str | Path,
    edge_csv: str | Path,
) -> PreparedFeatureDataset:
    """Extract independent feature blocks and fit learned transforms on train.

    The graph transformation is transductive and uses topology from every
    roster cohort. No graph labels are read or propagated.
    """
    _validate_records(records)
    account_ids = tuple(record.account_id for record in records)
    splits = tuple(record.split for record in records)
    labels = tuple(record.label for record in records)

    profile_raw = extract_profile_features([record.profile for record in records])
    text_summary_raw, documents = extract_text_features(
        [record.tweets for record in records]
    )
    network_frame = extract_network_features(
        account_ids,
        splits,
        split_csv=split_csv,
        edge_csv=edge_csv,
    )

    train_indices = [
        index for index, split in enumerate(splits) if split == "train"
    ]
    preprocessor = FeaturePreprocessor().fit(
        profile_raw,
        text_summary_raw,
        documents,
        train_indices,
    )
    transformed = preprocessor.transform(profile_raw, text_summary_raw, documents)

    feature_frames: dict[str, pd.DataFrame | sparse.csr_matrix] = {
        "profile": transformed["profile"],
        "text_summary": transformed["text_summary"],
        "text_tfidf": transformed["text_tfidf"],
        "network": network_frame,
    }
    feature_names: dict[str, tuple[str, ...]] = {}
    feature_matrices: dict[str, sparse.csr_matrix] = {}
    for group in FEATURE_GROUP_ORDER:
        frame_or_matrix = feature_frames[group]
        if isinstance(frame_or_matrix, pd.DataFrame):
            names = tuple(str(name) for name in frame_or_matrix.columns)
            matrix = sparse.csr_matrix(
                frame_or_matrix.to_numpy(dtype=np.float32, copy=True)
            )
        else:
            names = (
                tuple(
                    f"tweet.tfidf.{name}"
                    for name in preprocessor.feature_names_[group]
                )
                if group == "text_tfidf"
                else tuple(preprocessor.feature_names_[group])
            )
            matrix = frame_or_matrix.astype(np.float32).tocsr()
        if matrix.shape != (len(records), len(names)):
            raise AssertionError(
                f"{group} shape {matrix.shape} does not match rows/features"
            )
        if len(set(names)) != len(names):
            raise AssertionError(f"Duplicate feature names in {group}")
        feature_names[group] = names
        feature_matrices[group] = matrix

    forbidden_names = {"account_id", "id", "id_str", "screen_name", "label", "split"}
    for group, names in feature_names.items():
        if group == "text_tfidf":
            if not all(name.startswith("tweet.tfidf.") for name in names):
                raise AssertionError("TF-IDF feature names are not properly namespaced")
            continue
        if forbidden_names.intersection(names):
            raise AssertionError(f"Identity/target field entered {group} features")

    exposure = overlap_exposure(documents, splits)
    if len(exposure) != len(records):
        raise AssertionError("Overlap sidecar row count does not match source records")
    if tuple(
        f"tweet.tfidf.{name}"
        for name in preprocessor.feature_names_["text_tfidf"]
    ) != feature_names["text_tfidf"]:
        raise AssertionError("TF-IDF feature order changed after fitting")

    manifest = feature_schema_manifest(
        tuple(
            name.removeprefix("tweet.tfidf.")
            for name in feature_names["text_tfidf"]
        )
    )
    manifest.update(
        {
            "row_count": len(records),
            "split_counts": {
                split: splits.count(split) for split in ("train", "dev", "test")
            },
            "preprocessing_fit_split": "train",
            "imputation_medians": {
                "profile": preprocessor.profile_medians_,
                "text_summary": preprocessor.summary_medians_,
            },
            "source_paths": {
                "official_archive": None,
                "split_csv": str(Path(split_csv).resolve()),
                "edge_csv": str(Path(edge_csv).resolve()),
            },
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
                "joblib": joblib.__version__,
            },
            "graph_transductive": True,
            "graph_labels_read": False,
            "graph_relation_edge_counts": network_frame.attrs[
                "relation_edge_counts"
            ],
            "labels_in_feature_matrices": False,
            "account_ids_in_feature_matrices": False,
            "feature_matrices": {
                group: {
                    "shape": list(matrix.shape),
                    "dtype": str(matrix.dtype),
                    "format": "csr",
                }
                for group, matrix in feature_matrices.items()
            },
            "source_archive_members": [
                "Twibot-20/train.json",
                "Twibot-20/dev.json",
                "Twibot-20/test.json",
            ],
        }
    )
    manifest["schema_version"] = SCHEMA_VERSION
    return PreparedFeatureDataset(
        account_ids=account_ids,
        splits=splits,
        labels=labels,
        features=feature_matrices,
        feature_names=feature_names,
        overlap_exposure=exposure,
        manifest=manifest,
        preprocessor=preprocessor,
    )


def extract_from_sources(
    archive_path: str | Path,
    split_csv: str | Path,
    edge_csv: str | Path,
) -> PreparedFeatureDataset:
    """Read the three original split members and build the feature dataset."""
    records = list(iter_official_records(str(archive_path)))
    dataset = build_feature_dataset(records, split_csv, edge_csv)
    dataset.manifest["source_paths"]["official_archive"] = str(
        Path(archive_path).resolve()
    )
    return dataset


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_feature_dataset(
    dataset: PreparedFeatureDataset,
    output_directory: str | Path,
) -> Path:
    """Save sparse feature blocks and a separate metadata sidecar.

    The destination must not already exist. IDs, labels, and overlap flags are
    serialized only to metadata.json and never into any feature matrix.
    """
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for group, matrix in dataset.features.items():
            sparse.save_npz(destination / f"features_{group}.npz", matrix)
        dataset.metadata_frame().to_json(
            destination / "metadata.json",
            orient="records",
            force_ascii=False,
            indent=2,
        )
        joblib.dump(dataset.preprocessor, destination / "preprocessor.joblib", compress=3)
        manifest = dict(dataset.manifest)
        manifest["feature_files"] = {
            group: f"features_{group}.npz" for group in dataset.features
        }
        manifest["metadata_file"] = "metadata.json"
        manifest["preprocessor_file"] = "preprocessor.joblib"
        manifest["source_sha256"] = {}
        for key, raw_path in manifest["source_paths"].items():
            if raw_path is not None:
                manifest["source_sha256"][key] = _sha256_file(Path(raw_path))
        (destination / "feature_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        for path in destination.iterdir():
            if path.is_file():
                path.unlink()
        destination.rmdir()
        raise
    return destination


__all__ = [
    "EXPERIMENTS",
    "PreparedFeatureDataset",
    "build_feature_dataset",
    "extract_from_sources",
    "save_feature_dataset",
]
