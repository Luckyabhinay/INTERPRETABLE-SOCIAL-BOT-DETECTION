"""Phase 4: resource-bounded, interruption-safe TwiBot-20 model comparison.

The runner is intentionally not invoked by feature extraction or test suites.
It preserves train-only fitting, development-only selection, and a persisted
selection lock before any test labels are accessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, MutableSequence, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.tree import DecisionTreeClassifier

from twis20.pipeline import EXPERIMENTS

SEED = 20261106
EXPERIMENT_LABELS = {
    "profile_only": "A_profile_only",
    "text_only": "B_text_only",
    "profile_text": "C_profile_text",
    "profile_network": "D_profile_network",
    "profile_text_network": "E_profile_text_network",
}
BLOCK_NAMES = ("profile", "text_summary", "text_tfidf", "network")
ALGORITHMS = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "lightgbm",
    "xgboost",
)


def rss_mb() -> float | None:
    """Return process RSS if psutil is installed."""
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / 1024**2


def load_manifest(path: Path) -> dict[str, Any]:
    """Load the UTF-8 manifest independently of the Windows ANSI locale."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_metadata_without_labels(path: Path) -> pd.DataFrame:
    """Load routing metadata after removing labels during JSON decoding."""

    def remove_label(record: dict[str, Any]) -> dict[str, Any]:
        if "account_id" in record:
            record.pop("label", None)
        return record

    with path.open("r", encoding="utf-8") as stream:
        return pd.DataFrame(json.load(stream, object_hook=remove_label))


def load_labels_for_indices(path: Path, indices: np.ndarray) -> np.ndarray:
    """Return labels only for explicitly requested official-row indices."""
    requested = set(int(index) for index in indices)
    record_index = 0

    def retain_requested_label(record: dict[str, Any]) -> dict[str, Any]:
        nonlocal record_index
        if "account_id" in record:
            if record_index not in requested:
                record.pop("label", None)
            record_index += 1
        return record

    with path.open("r", encoding="utf-8") as stream:
        rows = json.load(stream, object_hook=retain_requested_label)
    labels = [
        int(row["label"])
        for index, row in enumerate(rows)
        if index in requested
    ]
    if len(labels) != len(indices):
        raise ValueError("Requested label indices were not all present in metadata")
    return np.asarray(labels, dtype=np.int8)


def _validate_selection_lock(lock_path: Path) -> dict[str, Any]:
    if not lock_path.is_file():
        raise RuntimeError("Test labels cannot be loaded before selection lock exists")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    selection = lock.get("selection")
    metadata = lock.get("selection_metadata")
    if not isinstance(selection, dict) or not selection:
        raise ValueError("Selection lock does not contain a selected candidate")
    if not isinstance(metadata, dict):
        raise ValueError("Selection lock is missing selection metadata")
    if metadata.get("selection_split") != "dev":
        raise ValueError("Test-label lock must record development-only selection")
    if metadata.get("test_metrics_used") is not False:
        raise ValueError("Test-label lock must affirm test metrics were not used")
    for field in ("experiment", "algorithm", "setting", "threshold"):
        if field not in selection:
            raise ValueError(f"Selection lock is missing {field!r}")
    return lock


def load_test_labels_after_lock(
    metadata_path: Path,
    test_indices: np.ndarray,
    lock_path: Path,
) -> np.ndarray:
    """Guard the sole test-label access point behind a valid persisted lock."""
    _validate_selection_lock(lock_path)
    return load_labels_for_indices(metadata_path, test_indices)


def load_blocks(root: Path) -> tuple[pd.DataFrame, dict[str, sparse.csr_matrix], dict[str, Any]]:
    """Load sparse feature blocks and label-free routing metadata."""
    root = Path(root)
    metadata = load_metadata_without_labels(root / "metadata.json")
    blocks = {
        name: sparse.load_npz(root / f"features_{name}.npz").tocsr()
        for name in BLOCK_NAMES
    }
    return metadata, blocks, load_manifest(root / "feature_manifest.json")


def matrix(
    blocks: dict[str, sparse.csr_matrix],
    experiment: str,
) -> sparse.csr_matrix:
    """Combine blocks without densifying sparse TF-IDF."""
    result = sparse.hstack(
        [blocks[name] for name in EXPERIMENTS[experiment]],
        format="csr",
        dtype=np.float32,
    )
    if not sparse.isspmatrix_csr(result):
        raise AssertionError("Experiment matrix must remain CSR")
    return result


def candidates(name: str, lightgbm_n_jobs: int = 1) -> list[tuple[str, Any]]:
    """Return fixed candidates with resource-bounded thread settings."""
    if lightgbm_n_jobs not in {1, 2}:
        raise ValueError("LightGBM n_jobs must be 1 or 2")
    if name == "logistic_regression":
        return [
            (
                "C_0.3",
                LogisticRegression(
                    C=0.3, max_iter=1000, solver="liblinear", random_state=SEED
                ),
            ),
            (
                "C_1",
                LogisticRegression(
                    C=1, max_iter=1000, solver="liblinear", random_state=SEED
                ),
            ),
        ]
    if name == "decision_tree":
        return [
            (
                "depth_12",
                DecisionTreeClassifier(
                    max_depth=12, min_samples_leaf=5, random_state=SEED
                ),
            ),
            (
                "depth_none",
                DecisionTreeClassifier(min_samples_leaf=5, random_state=SEED),
            ),
        ]
    if name == "random_forest":
        return [
            (
                "150_sqrt",
                RandomForestClassifier(
                    n_estimators=150,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    n_jobs=1,
                    random_state=SEED,
                ),
            ),
            (
                "300_sqrt",
                RandomForestClassifier(
                    n_estimators=300,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    n_jobs=1,
                    random_state=SEED,
                ),
            ),
        ]
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return [
            (
                "fast_100_15",
                LGBMClassifier(
                    n_estimators=100,
                    num_leaves=15,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    n_jobs=lightgbm_n_jobs,
                    random_state=SEED,
                    verbosity=-1,
                ),
            ),
            (
                "balanced_200_31",
                LGBMClassifier(
                    n_estimators=200,
                    num_leaves=31,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    n_jobs=lightgbm_n_jobs,
                    random_state=SEED,
                    verbosity=-1,
                ),
            ),
        ]
    if name == "xgboost":
        from xgboost import XGBClassifier

        return [
            (
                "depth_6",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    n_jobs=1,
                    random_state=SEED,
                    eval_metric="logloss",
                ),
            ),
            (
                "depth_8",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    n_jobs=1,
                    random_state=SEED,
                    eval_metric="logloss",
                ),
            ),
        ]
    raise KeyError(name)


def score(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, prediction),
        "precision_bot": precision_score(y, prediction, zero_division=0),
        "recall_bot": recall_score(y, prediction, zero_division=0),
        "f1_bot": f1_score(y, prediction, zero_division=0),
        "macro_f1": f1_score(y, prediction, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "roc_auc": roc_auc_score(y, probability),
        "pr_auc": average_precision_score(y, probability),
        "genuine_recall": recall_score(y, prediction, pos_label=0, zero_division=0),
        "false_positive_rate": fp / (fp + tn) if fp + tn else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    options = np.arange(0.10, 0.91, 0.01)
    best = max(
        options,
        key=lambda threshold: (
            f1_score(
                y, probability >= threshold, average="macro", zero_division=0
            ),
            -abs(threshold - 0.5),
        ),
    )
    return float(best)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return value.__name__
    return repr(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_event(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(event, ensure_ascii=False, default=_json_default) + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def _safe_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _candidate_configuration(model: Any) -> dict[str, Any]:
    get_params = getattr(model, "get_params", None)
    if get_params is None:
        return {"class": type(model).__qualname__}
    return {
        "class": type(model).__qualname__,
        "parameters": get_params(deep=True),
    }


def _candidate_paths(
    output: Path, experiment: str, algorithm: str, setting: str
) -> tuple[Path, Path, Path]:
    identifier = "__".join(
        _safe_identifier(part) for part in (experiment, algorithm, setting)
    )
    return (
        output / "candidate_records" / f"{identifier}.json",
        output / "candidate_models" / f"{identifier}.joblib",
        output / "candidate_models" / f".{identifier}.{os.getpid()}.tmp",
    )


def evaluate_candidate(
    *,
    output: Path,
    experiment: str,
    experiment_index: int,
    algorithm: str,
    algorithm_index: int,
    candidate_index: int,
    candidate_count: int,
    setting: str,
    model: Any,
    X_train: sparse.csr_matrix,
    X_dev: sparse.csr_matrix,
    y_train: np.ndarray,
    y_dev: np.ndarray,
    feature_input_fingerprint: str,
    memory: dict[str, Any],
) -> dict[str, Any]:
    """Fit, evaluate, and checkpoint exactly one candidate."""
    if not sparse.isspmatrix_csr(X_train) or not sparse.isspmatrix_csr(X_dev):
        raise TypeError("Candidate train/dev matrices must remain sparse CSR")

    record_path, model_path, temporary_model_path = _candidate_paths(
        output, experiment, algorithm, setting
    )
    start_wall = _utc_now()
    start_clock = time.perf_counter()
    record: dict[str, Any] = {
        "experiment": experiment,
        "experiment_index": experiment_index,
        "experiment_count": len(EXPERIMENTS),
        "configuration": EXPERIMENT_LABELS[experiment],
        "algorithm": algorithm,
        "algorithm_index": algorithm_index,
        "algorithm_count": len(ALGORITHMS),
        "setting": setting,
        "candidate_index": candidate_index,
        "candidate_count": candidate_count,
        "status": "running",
        "started_at": start_wall,
        "ended_at": None,
        "duration_seconds": None,
        "configuration_parameters": _candidate_configuration(model),
        "feature_input_fingerprint": feature_input_fingerprint,
        "train_shape": list(X_train.shape),
        "dev_shape": list(X_dev.shape),
        "sparse_format": "csr",
        **memory,
    }
    event_path = output / "candidate_events.jsonl"
    state_path = output / "run_state.json"
    _atomic_json(state_path, {"status": "running", "current_candidate": record})
    _atomic_json(record_path, record)
    _append_event(event_path, {**record, "event": "started"})
    print(
        f"[{experiment_index}/{len(EXPERIMENTS)}] {experiment} | "
        f"[{algorithm_index}/{len(ALGORITHMS)}] {algorithm} | "
        f"[{candidate_index}/{candidate_count}] {setting}: started",
        flush=True,
    )

    try:
        import tracemalloc

        rss_before = rss_mb()
        tracemalloc.start()
        model.fit(X_train, y_train)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        probability = model.predict_proba(X_dev)[:, 1]
        threshold = choose_threshold(y_dev, probability)
        metrics = score(y_dev, probability, threshold)

        joblib.dump(model, temporary_model_path, compress=3)
        os.replace(temporary_model_path, model_path)
        record.update(
            {
                "status": "complete",
                "ended_at": _utc_now(),
                "duration_seconds": time.perf_counter() - start_clock,
                "rss_before_mb": rss_before,
                "rss_after_mb": rss_mb(),
                "peak_tracemalloc_mb": peak / 1024**2,
                "threshold": threshold,
                "metrics": metrics,
                "model_file": str(model_path.relative_to(output)),
            }
        )
    except KeyboardInterrupt:
        try:
            import tracemalloc

            if tracemalloc.is_tracing():
                tracemalloc.stop()
        except ImportError:
            pass
        if temporary_model_path.exists():
            temporary_model_path.unlink()
        if model_path.exists():
            model_path.unlink()
        record.update(
            {
                "status": "interrupted",
                "ended_at": _utc_now(),
                "duration_seconds": time.perf_counter() - start_clock,
                "error": "KeyboardInterrupt",
                "traceback": traceback.format_exc(),
            }
        )
        _atomic_json(record_path, record)
        _append_event(event_path, {**record, "event": "interrupted"})
        _atomic_json(state_path, {"status": "interrupted", "current_candidate": record})
        print(
            f"[{experiment_index}/{len(EXPERIMENTS)}] {experiment} | "
            f"[{algorithm_index}/{len(ALGORITHMS)}] {algorithm} | "
            f"[{candidate_index}/{candidate_count}] {setting}: interrupted",
            flush=True,
        )
        return record
    except Exception as exc:
        try:
            import tracemalloc

            if tracemalloc.is_tracing():
                tracemalloc.stop()
        except ImportError:
            pass
        if temporary_model_path.exists():
            temporary_model_path.unlink()
        if model_path.exists():
            model_path.unlink()
        record.update(
            {
                "status": "failed",
                "ended_at": _utc_now(),
                "duration_seconds": time.perf_counter() - start_clock,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )

    _atomic_json(record_path, record)
    _append_event(
        event_path,
        {**record, "event": "completed" if record["status"] == "complete" else "failed"},
    )
    _atomic_json(
        state_path,
        {
            "status": "running",
            "last_candidate": record,
            "current_candidate": None,
        },
    )
    print(
        f"[{experiment_index}/{len(EXPERIMENTS)}] {experiment} | "
        f"[{algorithm_index}/{len(ALGORITHMS)}] {algorithm} | "
        f"[{candidate_index}/{candidate_count}] {setting}: {record['status']} "
        f"({record['duration_seconds']:.1f}s)",
        flush=True,
    )
    return record


def run_candidate_settings(
    *,
    output: Path,
    experiment: str,
    experiment_index: int,
    algorithm: str,
    algorithm_index: int,
    settings: MutableSequence[tuple[str, Any]],
    X_train: sparse.csr_matrix,
    X_dev: sparse.csr_matrix,
    y_train: np.ndarray,
    y_dev: np.ndarray,
    feature_input_fingerprint: str,
    memory: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run settings sequentially; completed candidates are never kept in RAM."""
    results: list[dict[str, Any]] = []
    candidate_count = len(settings)
    candidate_index = 0
    while settings:
        candidate_index += 1
        setting, model = settings.pop(0)
        record = evaluate_candidate(
            output=output,
            experiment=experiment,
            experiment_index=experiment_index,
            algorithm=algorithm,
            algorithm_index=algorithm_index,
            candidate_index=candidate_index,
            candidate_count=candidate_count,
            setting=setting,
            model=model,
            X_train=X_train,
            X_dev=X_dev,
            y_train=y_train,
            y_dev=y_dev,
            feature_input_fingerprint=feature_input_fingerprint,
            memory=memory,
        )
        results.append(record)
        del model
        if record["status"] == "interrupted":
            break
    return results


def save_plots(
    y: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    prefix: Path,
) -> np.ndarray:
    import matplotlib.pyplot as plt

    prediction = (probability >= threshold).astype(int)
    matrix_values = confusion_matrix(y, prediction, labels=[0, 1])
    fig, axis = plt.subplots()
    axis.imshow(matrix_values, cmap="Blues")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["genuine", "bot"],
        yticklabels=["genuine", "bot"],
        xlabel="predicted",
        ylabel="actual",
    )
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(matrix_values[row, column]),
                ha="center",
                va="center",
            )
    fig.tight_layout()
    fig.savefig(str(prefix) + "_confusion_matrix.png", dpi=160)
    plt.close(fig)

    false_positive_rate, true_positive_rate, _ = roc_curve(y, probability)
    fig, axis = plt.subplots()
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"AUC={roc_auc_score(y, probability):.3f}",
    )
    axis.plot([0, 1], [0, 1], "--", color="grey")
    axis.legend()
    axis.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC")
    fig.tight_layout()
    fig.savefig(str(prefix) + "_roc.png", dpi=160)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y, probability)
    fig, axis = plt.subplots()
    axis.plot(
        recall,
        precision,
        label=f"AP={average_precision_score(y, probability):.3f}",
    )
    axis.legend()
    axis.set(xlabel="recall", ylabel="precision", title="Precision-recall")
    fig.tight_layout()
    fig.savefig(str(prefix) + "_pr.png", dpi=160)
    plt.close(fig)
    return matrix_values


def importance(model: Any, output: Path, names: Sequence[str]) -> None:
    values = getattr(model, "feature_importances_", None)
    if values is None:
        values = np.abs(np.ravel(model.coef_))
    top = np.argsort(values)[-30:][::-1]
    pd.DataFrame(
        {
            "feature": np.asarray(names, dtype=object)[top],
            "importance": values[top],
        }
    ).to_csv(output, index=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_input_fingerprint(feature_root: Path) -> tuple[str, dict[str, Any]]:
    files = [
        feature_root / "feature_manifest.json",
        *(feature_root / f"features_{name}.npz" for name in BLOCK_NAMES),
    ]
    details = {
        path.name: {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in files
    }
    encoded = json.dumps(details, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), details


def _write_candidate_summary(
    output: Path, candidate_records: list[dict[str, Any]]
) -> None:
    path = output / "development_all_candidates.csv"
    frame = pd.DataFrame(candidate_records)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _checkpoint_active_interruption(output: Path) -> dict[str, Any] | None:
    """Mark a candidate interrupted if Ctrl-C escaped its normal checkpoint path."""
    state_path = output / "run_state.json"
    if not state_path.is_file():
        return None
    state = load_manifest(state_path)
    current = state.get("current_candidate")
    if not isinstance(current, dict) or current.get("status") != "running":
        return None

    record_path, model_path, _ = _candidate_paths(
        output,
        current["experiment"],
        current["algorithm"],
        current["setting"],
    )
    if record_path.is_file():
        persisted = load_manifest(record_path)
        if persisted.get("status") != "running":
            return None
        current = persisted
    if model_path.is_file():
        model_path.unlink()

    ended_at = datetime.now(timezone.utc)
    try:
        started_at = datetime.fromisoformat(current["started_at"])
        duration_seconds = max((ended_at - started_at).total_seconds(), 0.0)
    except (KeyError, TypeError, ValueError):
        duration_seconds = None
    current.update(
        {
            "status": "interrupted",
            "ended_at": ended_at.isoformat(),
            "duration_seconds": duration_seconds,
            "error": "KeyboardInterrupt",
            "traceback": "KeyboardInterrupt escaped candidate checkpoint handling.",
        }
    )
    _atomic_json(record_path, current)
    _append_event(
        output / "candidate_events.jsonl", {**current, "event": "interrupted"}
    )
    _atomic_json(
        state_path, {"status": "interrupted", "current_candidate": current}
    )
    return current


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lightgbm-n-jobs",
        type=int,
        choices=(1, 2),
        default=1,
        help="LightGBM thread count (default: 1; allowed values: 1 or 2)",
    )
    args = parser.parse_args(argv)

    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("candidate_records", "candidate_models", "plots", "importance"):
        (output / directory).mkdir()

    metadata, blocks, manifest = load_blocks(args.features)
    if len(metadata) != manifest.get("row_count"):
        raise ValueError("Metadata row count does not match the feature manifest")
    if any(not sparse.isspmatrix_csr(block) for block in blocks.values()):
        raise ValueError("All Phase 3 feature blocks must remain CSR matrices")
    if any(block.shape[0] != len(metadata) for block in blocks.values()):
        raise ValueError("Feature block row count does not match metadata")

    split_indices = {
        split: np.flatnonzero(metadata["split"].eq(split).to_numpy())
        for split in ("train", "dev", "test")
    }
    if any(len(split_indices[split]) == 0 for split in split_indices):
        raise ValueError("All official splits must be represented")
    metadata_path = args.features / "metadata.json"
    y_train = load_labels_for_indices(metadata_path, split_indices["train"])
    y_dev = load_labels_for_indices(metadata_path, split_indices["dev"])

    selection_protocol = {
        "seed": SEED,
        "selection_split": "dev",
        "test_labels_accessed": "only after valid locked selection is written",
        "threshold_objective": "max development macro F1",
        "test_used_for_tuning": False,
        "lightgbm_n_jobs": args.lightgbm_n_jobs,
        "resume_supported": False,
    }
    _atomic_json(output / "selection_protocol.json", selection_protocol)
    fingerprint, input_details = _feature_input_fingerprint(args.features)
    _atomic_json(
        output / "run_manifest.json",
        {
            "created_at": _utc_now(),
            "output_directory": str(output.resolve()),
            "feature_directory": str(args.features.resolve()),
            "feature_input_fingerprint": fingerprint,
            "feature_inputs": input_details,
            "manifest_schema_version": manifest.get("schema_version"),
            "official_split_counts": {
                split: int(len(indices)) for split, indices in split_indices.items()
            },
            "thread_policy": {
                "lightgbm_n_jobs": args.lightgbm_n_jobs,
                "random_forest_n_jobs": 1,
                "xgboost_n_jobs": 1,
            },
            "automatic_resume": False,
        },
    )

    candidate_records: list[dict[str, Any]] = []
    _atomic_json(output / "run_state.json", {"status": "running"})
    interrupted = False
    try:
        for experiment_index, experiment in enumerate(EXPERIMENTS, start=1):
            experiment_matrix = matrix(blocks, experiment)
            rss_before = rss_mb()
            dense_equivalent_mb = (
                experiment_matrix.shape[0] * experiment_matrix.shape[1] * 4 / 1024**2
            )
            train_matrix = experiment_matrix[split_indices["train"]].tocsr()
            dev_matrix = experiment_matrix[split_indices["dev"]].tocsr()
            memory = {
                "rss_before_mb": rss_before,
                "dense_equivalent_mb": dense_equivalent_mb,
                "full_matrix_nnz": int(experiment_matrix.nnz),
                "train_matrix_nnz": int(train_matrix.nnz),
                "dev_matrix_nnz": int(dev_matrix.nnz),
            }
            print(
                f"Experiment {experiment_index}/{len(EXPERIMENTS)}: {experiment} "
                f"matrix={experiment_matrix.shape}, format=CSR",
                flush=True,
            )
            for algorithm_index, algorithm in enumerate(ALGORITHMS, start=1):
                settings = candidates(algorithm, args.lightgbm_n_jobs)
                records = run_candidate_settings(
                    output=output,
                    experiment=experiment,
                    experiment_index=experiment_index,
                    algorithm=algorithm,
                    algorithm_index=algorithm_index,
                    settings=settings,
                    X_train=train_matrix,
                    X_dev=dev_matrix,
                    y_train=y_train,
                    y_dev=y_dev,
                    feature_input_fingerprint=fingerprint,
                    memory=memory,
                )
                candidate_records.extend(records)
                _write_candidate_summary(output, candidate_records)
                if records and records[-1]["status"] == "interrupted":
                    interrupted = True
                    break
                del settings
            del experiment_matrix, train_matrix, dev_matrix
            if interrupted:
                break
    except KeyboardInterrupt:
        interrupted = True

    if interrupted:
        active_interruption = _checkpoint_active_interruption(output)
        if active_interruption is not None and not any(
            record.get("experiment") == active_interruption.get("experiment")
            and record.get("algorithm") == active_interruption.get("algorithm")
            and record.get("setting") == active_interruption.get("setting")
            for record in candidate_records
        ):
            candidate_records.append(active_interruption)
        _atomic_json(
            output / "run_state.json",
            {
                "status": "interrupted",
                "ended_at": _utc_now(),
                "completed_candidates": sum(
                    record["status"] in {"complete", "failed"}
                    for record in candidate_records
                ),
                "interrupted_candidate_recorded": any(
                    record["status"] == "interrupted"
                    for record in candidate_records
                ),
                "automatic_resume": False,
            },
        )
        _write_candidate_summary(output, candidate_records)
        print("Training interrupted. Completed candidate checkpoints were preserved.", flush=True)
        return 130

    _write_candidate_summary(output, candidate_records)
    successful = [record for record in candidate_records if record["status"] == "complete"]
    if not successful:
        _atomic_json(
            output / "run_state.json",
            {"status": "failed", "ended_at": _utc_now(), "reason": "no successful candidates"},
        )
        raise RuntimeError("No successful development candidates; test labels remain locked")

    selected = max(
        successful,
        key=lambda record: (
            record["metrics"]["macro_f1"],
            record["metrics"]["pr_auc"],
            -record["duration_seconds"],
        ),
    )
    selection = {
        key: selected[key]
        for key in (
            "experiment",
            "configuration",
            "algorithm",
            "setting",
            "threshold",
            "metrics",
            "duration_seconds",
        )
    }
    selection["candidate_record"] = str(
        Path("candidate_records") / Path(
            _candidate_paths(
                output,
                selected["experiment"],
                selected["algorithm"],
                selected["setting"],
            )[0]
        ).name
    )
    lock_path = output / "locked_selection_before_test.json"
    _atomic_json(
        lock_path,
        {
            "selection": selection,
            "selection_metadata": {
                "seed": SEED,
                "selection_split": "dev",
                "ranking_metrics": ["macro_f1", "pr_auc", "duration_seconds"],
                "test_metrics_used": False,
                "feature_input_fingerprint": fingerprint,
            },
        },
    )
    _validate_selection_lock(lock_path)

    selected_model_path = output / selected["model_file"]
    model = joblib.load(selected_model_path)
    experiment = selected["experiment"]
    algorithm = selected["algorithm"]
    setting = selected["setting"]
    threshold = float(selected["threshold"])
    combined_matrix = matrix(blocks, experiment)

    # This is the only test-label access point and follows the validated lock.
    y_test = load_test_labels_after_lock(
        metadata_path, split_indices["test"], lock_path
    )
    probability = model.predict_proba(combined_matrix[split_indices["test"]])[:, 1]
    test_metrics = score(y_test, probability, threshold)
    confusion = save_plots(
        y_test,
        probability,
        threshold,
        output / "plots" / "official_test",
    )
    pd.DataFrame(
        [
            {
                "experiment": experiment,
                "configuration": EXPERIMENT_LABELS[experiment],
                "algorithm": algorithm,
                "setting": setting,
                "threshold": threshold,
                **test_metrics,
            }
        ]
    ).to_csv(output / "official_test_metrics.csv", index=False)
    pd.DataFrame(
        confusion,
        index=["actual_genuine", "actual_bot"],
        columns=["pred_genuine", "pred_bot"],
    ).to_csv(output / "official_test_confusion_matrix.csv")
    report = classification_report(
        y_test,
        probability >= threshold,
        target_names=["genuine", "bot"],
        output_dict=True,
        zero_division=0,
    )
    _atomic_json(output / "official_test_classification_report.json", report)

    sensitivity: list[dict[str, Any]] = []
    for flag in ("text_overlap_exact_train", "text_overlap_normalized_train"):
        for exposed in (False, True):
            test_positions = split_indices["test"]
            mask = metadata.loc[test_positions, flag].fillna(False).astype(bool).to_numpy()
            selected_positions = test_positions[mask == exposed]
            if len(selected_positions):
                cohort_positions = np.isin(test_positions, selected_positions)
                cohort_labels = y_test[cohort_positions]
                sensitivity.append(
                    {
                        "exposure_flag": flag,
                        "exposed": exposed,
                        "n_accounts": len(selected_positions),
                        "genuine_support": int((cohort_labels == 0).sum()),
                        "bot_support": int((cohort_labels == 1).sum()),
                        **score(cohort_labels, probability[cohort_positions], threshold),
                    }
                )
    pd.DataFrame(sensitivity).to_csv(
        output / "overlap_aware_test_sensitivity.csv", index=False
    )

    names = manifest.get("feature_names", {})
    feature_names = [
        name
        for group in EXPERIMENTS[experiment]
        for name in names.get(group, [])
    ]
    if feature_names:
        importance(
            model,
            output / "importance" / "selected_model_top30.csv",
            feature_names,
        )
    report_lines = [
        "# TwiBot-20 Phase 4 — Training Report",
        "",
        "## Protocol",
        "",
        f"- Fixed random seed: `{SEED}`.",
        "- Supervised fitting used official train labels only. Development labels selected the model setting and probability threshold; that locked selection was written before test labels were accessed.",
        "- Account identifiers, official split flags, labels, and overlap flags were retained only in metadata and never entered model matrices.",
        "- Network rows use friend/follow topology from train, dev, test, and unlabeled support accounts. This is transductive topology exposure; no graph labels or label-composition features were used.",
        "- Sparse matrices were retained throughout model fitting.",
        "- Candidate-level status, configuration, metrics, timing, and model files were checkpointed independently. Automatic resume is not implemented.",
        "",
        "## Locked selection",
        "",
        f"- Experiment: `{EXPERIMENT_LABELS[experiment]}`",
        f"- Algorithm / setting: `{algorithm}` / `{setting}`",
        f"- Development-selected threshold: `{threshold:.2f}`",
        "",
        "## Official test evaluation",
        "",
        "Primary all-account test metrics (bot is positive):",
        "",
        pd.DataFrame([test_metrics]).round(6).to_markdown(index=False),
        "",
        "## Overlap-aware sensitivity",
        "",
        "`overlap_aware_test_sensitivity.csv` partitions, but does not remove or reassign, official test accounts.",
    ]
    (output / "phase4_training_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    _atomic_json(
        output / "run_state.json",
        {
            "status": "complete",
            "ended_at": _utc_now(),
            "completed_candidates": len(candidate_records),
            "selected_experiment": experiment,
            "selected_algorithm": algorithm,
            "selected_setting": setting,
        },
    )
    print(
        json.dumps(
            {"selected": selection, "official_test": test_metrics, "output": str(output)},
            indent=2,
            default=_json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
