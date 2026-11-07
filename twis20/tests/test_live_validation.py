from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from fastapi.testclient import TestClient

import app as application
from twis20 import live_validation


def _registry_rows(count: int = 30) -> list[dict[str, str]]:
    rows = []
    classes = ("genuine_verified", "bot_verified", "suspicious_unverified")
    for index in range(count):
        rows.append(
            {
                "record_id": f"record-{index:02}",
                "username": f"account_{index:02}",
                "reference_class": classes[index % len(classes)],
                "reference_source": "independent review",
                "evidence_reference": f"evidence-{index:02}",
                "reviewer": "reviewer-1",
                "reviewed_at": "2026-01-15T12:00:00Z",
            }
        )
    return rows


class _FakeResponse:
    status = 200

    def __init__(self, value: dict):
        self.payload = json.dumps(value).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self):
        return self.payload


class LiveValidationTests(unittest.TestCase):
    def _write_registry(self, directory: str, rows: list[dict[str, str]]) -> Path:
        path = Path(directory) / "registry.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = live_validation.csv.DictWriter(stream, fieldnames=live_validation.REGISTRY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_load_registry_accepts_30_complete_unique_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_registry(directory, _registry_rows())
            loaded = live_validation.load_registry(path)
        self.assertEqual(len(loaded), 30)

    def test_load_registry_rejects_duplicate_handle_and_profile_url(self):
        rows = _registry_rows()
        rows[1]["username"] = "https://x.com/account_00"
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_registry(directory, rows)
            with self.assertRaisesRegex(ValueError, "duplicate X account"):
                live_validation.load_registry(path)

    def test_load_registry_rejects_incomplete_sample_and_non_iso_date(self):
        with tempfile.TemporaryDirectory() as directory:
            small = self._write_registry(directory, _registry_rows(29))
            with self.assertRaisesRegex(ValueError, "30–50"):
                live_validation.load_registry(small)
            rows = _registry_rows()
            rows[0]["reviewed_at"] = "yesterday"
            malformed = self._write_registry(directory, rows)
            with self.assertRaisesRegex(ValueError, "ISO-8601"):
                live_validation.load_registry(malformed)

    def test_analyze_account_keeps_uncalibrated_scores_and_no_raw_content(self):
        payload = {
            "status": "success",
            "model": "TwiBot-20 profile+text model",
            "prediction": "Likely Bot",
            "model_score_uncalibrated": 0.81,
            "class_scores_uncalibrated": {"Genuine": 0.19, "Bot": 0.81},
            "probability": None,
            "confidence": None,
            "threshold": 0.49,
            "feature_count_required": 50072,
            "feature_count_constructed": 50072,
            "feature_audit": {
                "feature_order_validated": True,
                "preprocessing_validated": True,
            },
            "ood_validation": {"audit_warnings": ["distribution differs"]},
            "profile_summary": {"description": "must not be saved"},
        }
        with patch.object(live_validation, "urlopen", return_value=_FakeResponse(payload)):
            result = live_validation.analyze_account(
                "http://127.0.0.1:8000", "account_01", 2
            )
        self.assertEqual(result["bot_score_uncalibrated"], 0.81)
        self.assertIsNone(result["probability"])
        self.assertNotIn("profile_summary", result)

    def test_analyze_account_rejects_invalid_score_and_records_http_422(self):
        payload = {
            "status": "success",
            "prediction": "Likely Genuine",
            "model_score_uncalibrated": 1.2,
            "class_scores_uncalibrated": {"Genuine": -0.2, "Bot": 1.2},
        }
        with patch.object(live_validation, "urlopen", return_value=_FakeResponse(payload)):
            invalid = live_validation.analyze_account(
                "http://127.0.0.1:8000", "account_01", 2
            )
        self.assertEqual(invalid["request_status"], "invalid_response")

        error = HTTPError(
            "http://127.0.0.1:8000/twibot20/analyze",
            422,
            "Unprocessable Entity",
            {},
            io.BytesIO(b'{"error_code":"feature_validation_failed"}'),
        )
        with patch.object(live_validation, "urlopen", side_effect=error):
            failed = live_validation.analyze_account(
                "http://127.0.0.1:8000", "account_01", 2
            )
        self.assertEqual(failed["request_status"], "http_error")
        self.assertEqual(failed["http_status"], 422)

        server_error = HTTPError(
            "http://127.0.0.1:8000/twibot20/analyze",
            500,
            "Internal Server Error",
            {},
            io.BytesIO(b'{"error_code":"analysis_failed"}'),
        )
        with patch.object(live_validation, "urlopen", side_effect=server_error):
            failed = live_validation.analyze_account(
                "http://127.0.0.1:8000", "account_01", 2
            )
        self.assertEqual(failed["http_status"], 500)
        self.assertEqual(failed["error_code"], "analysis_failed")

    def test_analyze_account_refuses_non_loopback_service_urls(self):
        with self.assertRaisesRegex(ValueError, "loopback"):
            live_validation.analyze_account(
                "https://example.org", "account_01", 2
            )

    def test_run_validation_excludes_unverified_and_failed_accounts_from_confusion(self):
        rows = _registry_rows()

        def fake_analyze(_base_url, username, _timeout):
            index = int(username[-2:])
            if index == 0:
                return {"request_status": "http_error", "http_status": 500}
            return {
                "request_status": "success",
                "prediction": "Likely Bot" if index % 2 else "Likely Genuine",
                "bot_score_uncalibrated": 0.8 if index % 2 else 0.2,
            }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            results, summary = live_validation.run_validation(
                rows,
                output,
                base_url="http://127.0.0.1:8000",
                delay_seconds=0,
                analyzer=fake_analyze,
            )
            summary_path = Path(directory) / "results_summary.json"
            self.assertTrue(output.exists())
            self.assertTrue(summary_path.exists())
            with self.assertRaises(FileExistsError):
                live_validation.run_validation(
                    rows,
                    output,
                    base_url="http://127.0.0.1:8000",
                    delay_seconds=0,
                    analyzer=fake_analyze,
                )
        self.assertEqual(len(results), 30)
        self.assertEqual(summary["failed_analyses"], 1)
        self.assertEqual(summary["suspicious_unverified_successes"], 10)
        self.assertEqual(summary["independently_labeled_successes"], 19)
        self.assertEqual(sum(summary["confusion_counts"].values()), 19)
        self.assertEqual(summary["confusion_counts"]["false_positive"], 5)
        self.assertEqual(summary["confusion_counts"]["false_negative"], 5)
        self.assertAlmostEqual(summary["false_positive_rate"], 5 / 9)
        self.assertAlmostEqual(summary["false_negative_rate"], 5 / 10)
        self.assertIsNotNone(summary["false_positive_rate"])
        self.assertIsNotNone(summary["false_negative_rate"])

    def test_benchmark_explainability_endpoint_identifies_separate_model(self):
        response = TestClient(application.app).get(
            "/twibot20/benchmark-explainability"
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["feature_count"], 50080)
        self.assertEqual(payload["explanation_split"], "development")
        self.assertEqual(payload["explanation_sample_size"], 200)
        self.assertAlmostEqual(payload["threshold"], 0.56)
        self.assertEqual(payload["shap_output"], "raw XGBoost margin")
        self.assertIn("not establish causal", payload["causality_note"])
        self.assertTrue(payload["global_features"])
        self.assertTrue(payload["individual_examples"])

    def test_dashboard_contains_model_separation_and_contribution_language(self):
        html = TestClient(application.app).get("/").text
        self.assertIn("Benchmark model explainability", html)
        self.assertIn("not the 50,072-feature live model", html)
        self.assertIn("not causal explanations", html)


if __name__ == "__main__":
    unittest.main()
