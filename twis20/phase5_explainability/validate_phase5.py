"""Validate Phase 5 SHAP artifacts against the locked Phase 4 provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PHASE4 = ROOT / "phase4_results_run03"
PHASE3 = ROOT / "phase3_features_phase4"
FEATURE_FILES = (
    "feature_manifest.json",
    "features_profile.npz",
    "features_text_summary.npz",
    "features_text_tfidf.npz",
    "features_network.npz",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(output: Path) -> None:
    validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    shape = validation["shap_shape"]
    assert shape == [validation["sample_rows"], validation["feature_names"]], (
        f"SHAP dimensions do not match sample/features: {shape}"
    )
    assert validation["model_inputs"] == validation["feature_names"], (
        "Model input count differs from ordered feature-name count"
    )
    assert validation["shap_all_finite"] is True, "SHAP values are not all finite"
    assert validation["error_shap_all_finite"] is True, (
        "Error-analysis SHAP values are not all finite"
    )
    residual = validation["max_additivity_residual"]
    assert np.isfinite(residual) and residual <= 1e-4, (
        f"SHAP additivity residual exceeded tolerance: {residual}"
    )

    before = validation["input_fingerprints_before"]
    after = validation["input_fingerprints_after"]
    assert before == after, "Input artifact fingerprints changed during Phase 5"
    manifest = json.loads(
        (PHASE4 / "run_manifest.json").read_text(encoding="utf-8")
    )
    for filename in FEATURE_FILES:
        actual = sha256(PHASE3 / filename)
        expected = before[filename]
        assert actual == expected, f"Phase 3 input hash changed: {filename}"
        assert actual == manifest["feature_inputs"][filename]["sha256"], (
            f"Phase 3 provenance mismatch: {filename}"
        )

    selection_path = PHASE4 / "locked_selection_before_test.json"
    metrics_path = PHASE4 / "official_test_metrics.csv"
    assert before["selection_lock"] == sha256(selection_path), (
        "Locked selection hash differs from current Phase 4 artifact"
    )
    assert before["official_test_metrics"] == sha256(metrics_path), (
        "Official Phase 4 test-metrics hash differs from current artifact"
    )
    assert validation["selection_lock_sha256_before"] == validation[
        "selection_lock_sha256_after"
    ], "Selection lock changed during Phase 5"
    assert validation["phase4_test_metrics_sha256_before"] == validation[
        "phase4_test_metrics_sha256_after"
    ], "Phase 4 test metrics changed during Phase 5"
    lock = json.loads(selection_path.read_text(encoding="utf-8"))
    metadata = lock["selection_metadata"]
    assert metadata["selection_split"] == "dev"
    assert metadata["test_metrics_used"] is False
    assert validation["locked_threshold"] == lock["selection"]["threshold"]
    error_residual = validation["error_analysis_max_additivity_residual"]
    assert np.isfinite(error_residual) and error_residual <= 1e-4, (
        f"Error-analysis additivity residual exceeded tolerance: {error_residual}"
    )
    for error_shape in validation["error_shap_shapes"]:
        assert error_shape[1] == validation["feature_names"]

    importance = pd.read_csv(output / "global_shap_importance.csv")
    top20 = pd.read_csv(output / "top20_features.csv")
    groups = pd.read_csv(output / "feature_group_importance.csv")
    assert len(importance) == validation["feature_names"], (
        "Global SHAP output row count differs from feature count"
    )
    assert len(top20) == min(20, len(importance)), "Top-20 output has wrong dimensions"
    assert importance["feature"].is_unique, "Global feature names are not unique"
    assert np.isfinite(importance["mean_abs_shap"]).all(), (
        "Global SHAP importance contains non-finite values"
    )
    assert (importance["mean_abs_shap"] >= 0).all(), (
        "Global absolute SHAP importance contains negative values"
    )
    assert np.isfinite(groups["mean_abs_shap"]).all()
    assert (groups["mean_abs_shap"] >= 0).all()
    assert np.isclose(
        importance["mean_abs_shap"].sum(),
        groups["mean_abs_shap"].sum(),
        rtol=1e-10,
        atol=1e-10,
    ), "Feature-group totals do not reconcile with global importance"
    assert list(top20["feature"]) == list(importance.head(20)["feature"]), (
        "Top-20 list does not match ranked global importances"
    )
    errors = pd.read_csv(output / "error_analysis.csv")
    if not errors.empty:
        assert np.isfinite(errors[["mean_shap", "mean_abs_shap"]].to_numpy()).all()
        assert set(errors["error_type"]).issubset({"false_positive", "false_negative"})
        assert (errors["analysis_sample_count"] <= 30).all()

    report = (output / "PHASE5_EXPLAINABILITY_REPORT.md").read_text(encoding="utf-8")
    for section in (
        "## Official Phase 4 test results",
        "## Top 20 global features",
        "## Aggregate feature-group importance",
        "## False-positive and false-negative findings",
        "## Limitations",
    ):
        assert section in report, f"Report is missing section: {section}"
    official_metrics = pd.read_csv(metrics_path).iloc[0]
    assert f"{official_metrics['accuracy']:.6f}" in report, (
        "Report omits the actual locked test accuracy"
    )
    assert importance.iloc[0]["feature"] in report, (
        "Report omits the highest-ranked feature"
    )
    assert groups.iloc[0]["feature_group"] in report, (
        "Report omits the top feature group"
    )
    print(
        f"PASS: SHAP shape={shape}, finite=True, "
        f"max_additivity_residual={residual:.8g}, "
        f"features={len(importance)}, groups={len(groups)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    validate(args.output)


if __name__ == "__main__":
    main()
