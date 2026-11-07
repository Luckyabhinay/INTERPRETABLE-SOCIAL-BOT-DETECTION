"""Package a Phase 4 profile+text candidate for live-compatible inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from twis20.feature_schema import FEATURE_GROUP_ORDER
from twis20.train_phase4 import (
    _feature_input_fingerprint,
    _validate_selection_lock,
    load_blocks,
    load_test_labels_after_lock,
    matrix,
    score,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURES = ROOT / "phase3_features_phase4"
DEFAULT_SOURCE_RUN = ROOT / "phase4_results_run03"
DEFAULT_OUTPUT = ROOT / "phase4_live_profile_text_run01"
LIVE_GROUPS = ("profile", "text_summary", "text_tfidf")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _candidate_records(source_run: Path, fingerprint: str) -> list[dict[str, Any]]:
    summary = source_run / "development_all_candidates.csv"
    if not summary.is_file():
        raise FileNotFoundError(f"Missing candidate summary: {summary}")

    import pandas as pd

    frame = pd.read_csv(summary)
    candidates: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        if row.get("experiment") != "profile_text" or row.get("status") != "complete":
            continue
        record_path = source_run / "candidate_records" / (
            f"profile_text__{row['algorithm']}__{row['setting']}.json"
        )
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("feature_input_fingerprint") != fingerprint:
            raise RuntimeError(f"Candidate input fingerprint mismatch: {record_path}")
        if record.get("train_shape") != [8278, 50072] or record.get("dev_shape") != [2365, 50072]:
            raise RuntimeError(f"Candidate has unexpected split or feature dimensions: {record_path}")
        if record.get("experiment") != "profile_text":
            raise RuntimeError(f"Candidate record has unexpected experiment: {record_path}")
        model_path = source_run / record["model_file"]
        if not model_path.is_file():
            raise FileNotFoundError(f"Candidate model missing: {model_path}")
        record["_record_path"] = record_path
        record["_model_path"] = model_path
        candidates.append(record)
    if not candidates:
        raise RuntimeError("No complete profile_text candidates match the current features")
    return candidates


def package_model(
    features: Path = DEFAULT_FEATURES,
    source_run: Path = DEFAULT_SOURCE_RUN,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")

    fingerprint, input_details = _feature_input_fingerprint(features)
    source_manifest = json.loads((source_run / "run_manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("feature_input_fingerprint") != fingerprint:
        raise RuntimeError("Source run was produced from different Phase 3 feature inputs")
    source_lock = json.loads(
        (source_run / "locked_selection_before_test.json").read_text(encoding="utf-8")
    )
    if source_lock.get("selection_metadata", {}).get("selection_split") != "dev":
        raise RuntimeError("Source run does not have a development-only selection lock")
    if source_lock.get("selection_metadata", {}).get("test_metrics_used") is not False:
        raise RuntimeError("Source run lock does not affirm the test-label boundary")

    candidates = _candidate_records(source_run, fingerprint)
    selected = max(
        candidates,
        key=lambda record: (
            float(record["metrics"]["macro_f1"]),
            float(record["metrics"]["pr_auc"]),
            -float(record["duration_seconds"]),
        ),
    )
    selected_model = joblib.load(selected["_model_path"])
    if selected_model.n_features_in_ != 50072:
        raise RuntimeError("Selected profile+text candidate has an unexpected input dimension")

    metadata, blocks, phase3_manifest = load_blocks(features)
    if phase3_manifest.get("feature_group_order") != list(FEATURE_GROUP_ORDER):
        raise RuntimeError("Phase 3 feature group order differs from the expected schema")
    feature_names = {
        group: phase3_manifest["feature_names"][group] for group in LIVE_GROUPS
    }
    ordered_names = [name for group in LIVE_GROUPS for name in feature_names[group]]
    if len(ordered_names) != 50072 or len(set(ordered_names)) != 50072:
        raise RuntimeError("Profile+text feature names are incomplete or duplicated")
    if selected_model.n_features_in_ != len(ordered_names):
        raise RuntimeError("Selected model dimension does not match the Phase 3 feature order")

    split_indices = {
        split: np.flatnonzero(metadata["split"].eq(split).to_numpy())
        for split in ("train", "dev", "test")
    }
    if {split: len(indices) for split, indices in split_indices.items()} != {
        "train": 8278,
        "dev": 2365,
        "test": 1183,
    }:
        raise RuntimeError("Official split counts do not match the Phase 3 protocol")

    output.mkdir(parents=True, exist_ok=False)
    model_path = output / "profile_text_model.joblib"
    preprocessor_path = output / "preprocessor.joblib"
    shutil.copyfile(selected["_model_path"], model_path)
    shutil.copyfile(features / "preprocessor.joblib", preprocessor_path)
    shutil.copyfile(selected["_record_path"], output / "selected_candidate_record.json")

    selection = {
        "experiment": selected["experiment"],
        "configuration": selected["configuration"],
        "algorithm": selected["algorithm"],
        "setting": selected["setting"],
        "threshold": float(selected["threshold"]),
        "metrics": selected["metrics"],
        "candidate_record": "selected_candidate_record.json",
        "candidate_source_run": str(source_run.resolve()),
    }
    lock = {
        "selection": selection,
        "selection_metadata": {
            "selection_split": "dev",
            "test_metrics_used": False,
            "ranking_metrics": ["macro_f1", "pr_auc", "duration_seconds"],
            "feature_input_fingerprint": fingerprint,
            "selection_candidates": len(candidates),
            "selection_source": "completed Phase 4 profile_text candidate records",
        },
    }
    lock_path = output / "locked_selection_before_test.json"
    _write_json(lock_path, lock)
    _validate_selection_lock(lock_path)

    deployment_manifest = {
        "model_name": "TwiBot-20 profile+text live-compatible model",
        "experiment": "profile_text",
        "feature_groups": list(LIVE_GROUPS),
        "feature_names": feature_names,
        "feature_count": len(ordered_names),
        "threshold": selection["threshold"],
        "selected_algorithm": selection["algorithm"],
        "selected_setting": selection["setting"],
        "development_metrics": selection["metrics"],
        "selection_protocol": lock["selection_metadata"],
        "feature_input_fingerprint": fingerprint,
        "feature_inputs": input_details,
        "model_sha256": _sha256_file(model_path),
        "preprocessor_sha256": _sha256_file(preprocessor_path),
        "model_source_sha256": _sha256_file(selected["_model_path"]),
        "phase3_preprocessor_source_sha256": _sha256_file(features / "preprocessor.joblib"),
        "official_split_counts": {
            split: len(indices) for split, indices in split_indices.items()
        },
        "network_features_used": False,
        "live_distribution_warning": (
            "Live X accounts may differ from the TwiBot-20 benchmark distribution."
        ),
    }
    _write_json(output / "deployment_manifest.json", deployment_manifest)

    y_test = load_test_labels_after_lock(
        features / "metadata.json", split_indices["test"], lock_path
    )
    X_test = matrix(blocks, "profile_text")[split_indices["test"]].tocsr()
    probability = selected_model.predict_proba(X_test)[:, 1]
    metrics = score(y_test, probability, selection["threshold"])
    confusion = confusion_matrix(
        y_test, probability >= selection["threshold"], labels=[0, 1]
    )
    test_report = classification_report(
        y_test,
        probability >= selection["threshold"],
        target_names=["genuine", "bot"],
        output_dict=True,
        zero_division=0,
    )
    _write_json(output / "official_test_metrics.json", metrics)
    _write_json(output / "official_test_classification_report.json", test_report)
    _write_json(
        output / "official_test_confusion_matrix.json",
        {"labels": ["genuine", "bot"], "matrix": confusion.tolist()},
    )

    report_lines = [
        "# TwiBot-20 live-compatible profile+text model",
        "",
        "This deployment package reuses a completed Phase 4 `profile_text` candidate,",
        "trained using the official train split and selected only by development",
        "macro F1, PR-AUC, and duration. It was copied into a separate directory;",
        "the locked run03 network-enabled model and its files were not changed.",
        "",
        "## Selected candidate",
        "",
        f"- Candidate: `{selection['algorithm']} / {selection['setting']}`",
        f"- Development threshold: `{selection['threshold']:.2f}`",
        f"- Development macro F1: `{selection['metrics']['macro_f1']:.6f}`",
        f"- Development PR-AUC: `{selection['metrics']['pr_auc']:.6f}`",
        f"- Input features: `{len(ordered_names)}` (profile + tweet summary + TF-IDF)",
        "- Network features: not used; no graph values are fabricated or required.",
        "",
        "## Official test result",
        "",
        f"- Accuracy: `{metrics['accuracy']:.6f}`",
        f"- Macro F1: `{metrics['macro_f1']:.6f}`",
        f"- ROC-AUC: `{metrics['roc_auc']:.6f}`",
        f"- PR-AUC: `{metrics['pr_auc']:.6f}`",
        f"- TN/FP/FN/TP: `{metrics['tn']}/{metrics['fp']}/{metrics['fn']}/{metrics['tp']}`",
        "",
        "The test labels were loaded only after `locked_selection_before_test.json`",
        "affirmed development-only selection and no test-metric use. These held-out",
        "benchmark results do not guarantee live X performance; current live fields,",
        "account availability, and domain shift may differ from the benchmark.",
    ]
    (output / "MODEL_PACKAGE_REPORT.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return {
        "output": str(output.resolve()),
        "selected": selection,
        "test_metrics": metrics,
        "model_sha256": deployment_manifest["model_sha256"],
        "feature_count": len(ordered_names),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(package_model(args.features, args.source_run, args.output), indent=2))


if __name__ == "__main__":
    main()
