"""Locale-independent manifest loading checks for Phase 4."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pandas as pd
import numpy as np
from scipy import sparse

from twis20.train_phase4 import (
    candidates,
    load_blocks,
    load_manifest,
    load_test_labels_after_lock,
    run_candidate_settings,
    _checkpoint_active_interruption,
)


class _FakeModel:
    def __init__(self, interrupt: bool = False, fail: bool = False) -> None:
        self.interrupt = interrupt
        self.fail = fail

    def get_params(self, deep: bool = True) -> dict[str, bool]:
        return {
            "interrupt": self.interrupt,
            "fail": self.fail,
            "deep_requested": deep,
        }

    def fit(self, X, y):
        assert sparse.isspmatrix_csr(X)
        assert y.shape[0] == X.shape[0]
        if self.interrupt:
            raise KeyboardInterrupt
        if self.fail:
            raise RuntimeError("intentional failure")
        return self

    def predict_proba(self, X):
        bot_probability = np.linspace(0.1, 0.9, X.shape[0])
        return np.column_stack((1 - bot_probability, bot_probability))


class Phase4ManifestEncodingTests(unittest.TestCase):
    def test_load_blocks_uses_utf8_when_windows_locale_is_cp1252(self) -> None:
        """A UTF-8 manifest must not be decoded through the ANSI code page."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"note": "café — 東京", "split_counts": {"train": 1}}
            (root / "feature_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
            )
            pd.DataFrame([{"account_id": "1", "split": "train", "label": 0}]).to_json(
                root / "metadata.json", orient="records", force_ascii=False
            )
            for name in ("profile", "text_summary", "text_tfidf", "network"):
                sparse.save_npz(root / f"features_{name}.npz", sparse.csr_matrix([[1.0]]))
            with patch("locale.getencoding", return_value="cp1252"):
                loaded_metadata, blocks, loaded_manifest = load_blocks(root)
                direct_manifest = load_manifest(root / "feature_manifest.json")
            self.assertEqual(loaded_manifest, manifest)
            self.assertEqual(direct_manifest, manifest)
            self.assertEqual(loaded_metadata.shape[0], 1)
            self.assertEqual({key: value.shape for key, value in blocks.items()}, {
                "profile": (1, 1), "text_summary": (1, 1),
                "text_tfidf": (1, 1), "network": (1, 1),
            })

    def test_test_labels_require_a_written_selection_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "metadata.json"
            metadata.write_text('[{"account_id":"1","split":"test","label":1}]', encoding="utf-8")
            lock = root / "locked_selection_before_test.json"
            with patch("twis20.train_phase4.load_labels_for_indices") as load_labels:
                with self.assertRaisesRegex(RuntimeError, "before selection lock"):
                    load_test_labels_after_lock(metadata, np.array([0]), lock)
                load_labels.assert_not_called()
                lock.write_text(
                    json.dumps(
                        {
                            "selection": {
                                "experiment": "profile_only",
                                "algorithm": "test",
                                "setting": "setting",
                                "threshold": 0.5,
                            },
                            "selection_metadata": {
                                "selection_split": "dev",
                                "test_metrics_used": False,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            self.assertEqual(load_test_labels_after_lock(metadata, np.array([0]), lock).tolist(), [1])

    def test_test_label_lock_rejects_non_dev_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "metadata.json"
            metadata.write_text('[{"account_id":"1","split":"test","label":1}]', encoding="utf-8")
            lock = root / "locked_selection_before_test.json"
            lock.write_text(
                json.dumps(
                    {
                        "selection": {
                            "experiment": "profile_only",
                            "algorithm": "test",
                            "setting": "setting",
                            "threshold": 0.5,
                        },
                        "selection_metadata": {
                            "selection_split": "test",
                            "test_metrics_used": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch("twis20.train_phase4.load_labels_for_indices") as load_labels:
                with self.assertRaisesRegex(ValueError, "development-only"):
                    load_test_labels_after_lock(metadata, np.array([0]), lock)
                load_labels.assert_not_called()

    def test_candidate_metrics_and_model_are_checkpointed_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ("candidate_records", "candidate_models"):
                (output / name).mkdir()
            X = sparse.csr_matrix([[0.0], [1.0]])
            settings = [
                ("broken", _FakeModel(fail=True)),
                ("working", _FakeModel()),
            ]
            records = run_candidate_settings(
                output=output,
                experiment="profile_only",
                experiment_index=1,
                algorithm="test",
                algorithm_index=1,
                settings=settings,
                X_train=X,
                X_dev=X,
                y_train=np.array([0, 1]),
                y_dev=np.array([0, 1]),
                feature_input_fingerprint="fixture-hash",
                memory={"dense_equivalent_mb": 0.001},
            )
            self.assertEqual([record["status"] for record in records], ["failed", "complete"])
            self.assertEqual(settings, [])
            record_path = output / "candidate_records" / "profile_only__test__working.json"
            model_path = output / "candidate_models" / "profile_only__test__working.joblib"
            persisted = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["status"], "complete")
            self.assertIn("metrics", persisted)
            self.assertIn("started_at", persisted)
            self.assertIn("ended_at", persisted)
            self.assertGreaterEqual(persisted["duration_seconds"], 0)
            self.assertTrue(model_path.is_file())
            events = [
                json.loads(line)
                for line in (output / "candidate_events.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual([event["event"] for event in events], [
                "started", "failed", "started", "completed"
            ])

    def test_keyboard_interrupt_is_checkpointed_and_stops_later_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ("candidate_records", "candidate_models"):
                (output / name).mkdir()
            X = sparse.csr_matrix([[0.0], [1.0]])
            settings = [
                ("interrupting", _FakeModel(interrupt=True)),
                ("not_started", _FakeModel()),
            ]
            records = run_candidate_settings(
                output=output,
                experiment="profile_only",
                experiment_index=1,
                algorithm="test",
                algorithm_index=1,
                settings=settings,
                X_train=X,
                X_dev=X,
                y_train=np.array([0, 1]),
                y_dev=np.array([0, 1]),
                feature_input_fingerprint="fixture-hash",
                memory={},
            )
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["status"], "interrupted")
            self.assertEqual(settings[0][0], "not_started")
            self.assertEqual(
                json.loads(
                    (output / "run_state.json").read_text(encoding="utf-8")
                )["status"],
                "interrupted",
            )
            events = [
                json.loads(line)
                for line in (output / "candidate_events.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual([event["event"] for event in events], [
                "started", "interrupted"
            ])

    def test_outer_interrupt_checkpoint_marks_running_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "candidate_records").mkdir()
            started = {
                "experiment": "profile_only",
                "algorithm": "test",
                "setting": "candidate",
                "status": "running",
                "started_at": "2026-01-01T00:00:00+00:00",
            }
            record_path = output / "candidate_records" / "profile_only__test__candidate.json"
            record_path.write_text(json.dumps(started), encoding="utf-8")
            (output / "run_state.json").write_text(
                json.dumps({"status": "running", "current_candidate": started}),
                encoding="utf-8",
            )
            interrupted = _checkpoint_active_interruption(output)
            self.assertIsNotNone(interrupted)
            self.assertEqual(interrupted["status"], "interrupted")
            self.assertGreaterEqual(interrupted["duration_seconds"], 0)
            self.assertEqual(
                json.loads(record_path.read_text(encoding="utf-8"))["status"],
                "interrupted",
            )

    def test_lightgbm_candidates_are_thread_bounded_and_smaller(self) -> None:
        lightgbm = ModuleType("lightgbm")

        class FakeLGBMClassifier:
            def __init__(self, **parameters):
                self.parameters = parameters

            def get_params(self, deep: bool = True):
                return {**self.parameters, "deep_requested": deep}

        lightgbm.LGBMClassifier = FakeLGBMClassifier
        with patch.dict(sys.modules, {"lightgbm": lightgbm}):
            settings = candidates("lightgbm", lightgbm_n_jobs=2)
            self.assertEqual(
                [
                    (model.parameters["n_estimators"], model.parameters["num_leaves"])
                    for _, model in settings
                ],
                [(100, 15), (200, 31)],
            )
            self.assertTrue(
                all(model.parameters["n_jobs"] == 2 for _, model in settings)
            )
            with self.assertRaises(ValueError):
                candidates("lightgbm", lightgbm_n_jobs=8)

    def test_feature_matrices_stay_sparse_when_combined(self) -> None:
        from twis20.train_phase4 import matrix

        blocks = {
            "profile": sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]]),
            "text_summary": sparse.csr_matrix([[1.0], [0.0]]),
            "text_tfidf": sparse.csr_matrix([[0.0, 1.0], [1.0, 0.0]]),
            "network": sparse.csr_matrix([[0.0], [1.0]]),
        }
        combined = matrix(blocks, "profile_text_network")
        self.assertTrue(sparse.isspmatrix_csr(combined))


if __name__ == "__main__":
    unittest.main()
