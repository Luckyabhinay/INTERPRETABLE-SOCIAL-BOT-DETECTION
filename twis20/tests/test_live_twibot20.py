from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
from fastapi.testclient import TestClient

import app as application
from twis20.live_twibot20 import (
    construct_profile_text,
    load_live_artifacts,
)


class _FakeModel:
    n_features_in_ = 50072

    def __init__(self, bot_score=0.8, classes=(0, 1), outputs=None):
        self.bot_score = bot_score
        self.classes_ = None if classes is None else np.asarray(classes)
        self.outputs = outputs

    def predict_proba(self, matrix):
        self.observed_shape = matrix.shape
        if self.outputs is not None:
            return np.asarray(self.outputs)
        scores = {0: 1 - self.bot_score, 1: self.bot_score}
        return np.asarray([[scores[label] for label in self.classes_]])


class LiveTwiBot20Tests(unittest.TestCase):
    @staticmethod
    def _twibot_analysis_function():
        return TestClient(application.app).get("/").text

    def _call_endpoint(
        self,
        *,
        feature_audit=None,
        artifact_error=None,
        provider="xapi",
        model_score=0.8,
        model_classes=(0, 1),
        model_outputs=None,
    ):
        manifest = {
            "model_name": "TwiBot-20 profile+text live-compatible model",
            "threshold": 0.49,
            "live_distribution_warning": "Benchmark/live distribution may differ.",
        }
        feature_audit = feature_audit or {
            "expected_feature_count": 50072,
            "constructed_feature_count": 50072,
            "profile_feature_count": 56,
            "tweet_summary_feature_count": 16,
            "tfidf_feature_count": 50000,
            "missing_feature_count": 0,
            "duplicate_or_invalid_feature_count": 0,
            "feature_order_validated": True,
            "preprocessing_validated": True,
            "preprocessing_details": {},
        }
        with (
            patch.object(application.XFetchClient, "has_key", return_value=provider == "xfetch"),
            patch.object(
                application.XFetchClient,
                "fetch_profile",
                return_value=(
                    {
                        "user": {
                            "id": "1",
                            "name": "Example",
                            "username": "example_user",
                            "follower_count": 12484212,
                            "following_count": 3,
                            "tweet_count": 31153,
                            "created_at": "2006-03-21T18:50:00.000Z",
                            "description": "bio",
                        },
                        "recent_tweets": [{"text": "a recent tweet"}],
                    },
                    None,
                    None,
                    200,
                ),
            ),
            patch.object(application.XApiClient, "has_token", return_value=provider == "xapi"),
            patch.object(
                application.XApiClient,
                "fetch_recent_tweets",
                return_value=([{"text": "a recent tweet"}], None),
            ),
            patch.object(
                application.XApiClient,
                "fetch_user_by_username",
                return_value=(
                    {
                        "id": "1",
                        "name": "Example",
                        "username": "example_user",
                        "public_metrics": {
                            "followers_count": 1,
                            "following_count": 2,
                            "tweet_count": 3,
                            "listed_count": 0,
                        },
                        "verified": False,
                        "description": "bio",
                        "created_at": "2020-01-01T00:00:00Z",
                    },
                    None,
                    200,
                ),
            ),
            patch.object(
                application,
                "load_live_artifacts",
                side_effect=artifact_error,
                return_value=(
                    manifest,
                    None,
                    _FakeModel(model_score, model_classes, model_outputs),
                    list(range(50072)),
                ),
            ),
            patch.object(
                application,
                "construct_profile_text",
                return_value=(
                    sparse.csr_matrix((1, 50072), dtype=np.float32),
                    {"profile": 56, "text_summary": 16, "text_tfidf": 50000},
                    [],
                    feature_audit,
                ),
            ),
        ):
            response = TestClient(application.app).post(
                "/twibot20/analyze", json={"username": "example_user"}
            )
        return response

    def test_profile_text_model_and_preprocessing_have_matching_dimensions(self):
        manifest, preprocessor, model, names = load_live_artifacts()
        matrix, available, missing, audit = construct_profile_text(
            {
                "public_metrics": {
                    "followers_count": 1,
                    "following_count": 2,
                    "tweet_count": 3,
                    "listed_count": 0,
                },
                "verified": False,
                "description": "hello",
                "created_at": "2020-01-01T00:00:00Z",
            },
            [{"text": "hello world"}],
            preprocessor,
        )
        self.assertEqual(manifest["experiment"], "profile_text")
        self.assertEqual(matrix.shape, (1, 50072))
        self.assertEqual(len(names), 50072)
        self.assertEqual(model.n_features_in_, 50072)
        self.assertEqual(model.classes_.tolist(), [0, 1])
        self.assertEqual(
            available, {"profile": 56, "text_summary": 16, "text_tfidf": 50000}
        )
        self.assertEqual(missing, [])
        self.assertEqual(audit["expected_feature_count"], 50072)
        self.assertEqual(audit["constructed_feature_count"], 50072)
        self.assertEqual(audit["profile_feature_count"], 56)
        self.assertEqual(audit["tweet_summary_feature_count"], 16)
        self.assertEqual(audit["tfidf_feature_count"], 50000)
        self.assertEqual(audit["missing_feature_count"], 0)
        self.assertEqual(audit["duplicate_or_invalid_feature_count"], 0)
        self.assertTrue(audit["feature_order_validated"])
        self.assertTrue(audit["preprocessing_validated"])

    @patch.object(application.XFetchClient, "has_key", return_value=False)
    @patch.object(application.XApiClient, "has_token", return_value=True)
    @patch.object(
        application.XApiClient,
        "fetch_recent_tweets",
        return_value=([{"text": "a recent tweet"}], None),
    )
    @patch.object(
        application.XApiClient,
        "fetch_user_by_username",
        return_value=(
            {
                "id": "1",
                "name": "Example",
                "username": "example_user",
                "public_metrics": {
                    "followers_count": 1,
                    "following_count": 2,
                    "tweet_count": 3,
                    "listed_count": 0,
                },
                "verified": False,
                "description": "bio",
                "created_at": "2020-01-01T00:00:00Z",
            },
            None,
            200,
        ),
    )
    @patch.object(
        application,
        "construct_profile_text",
        return_value=(
            sparse.csr_matrix((1, 50072), dtype=np.float32),
            {"profile": 56, "text_summary": 16, "text_tfidf": 50000},
            [],
            {
                "expected_feature_count": 50072,
                "constructed_feature_count": 50072,
                "profile_feature_count": 56,
                "tweet_summary_feature_count": 16,
                "tfidf_feature_count": 50000,
                "missing_feature_count": 0,
                "duplicate_or_invalid_feature_count": 0,
                "feature_order_validated": True,
                "preprocessing_validated": True,
                "preprocessing_details": {},
            },
        ),
    )
    @patch.object(
        application,
        "load_live_artifacts",
        return_value=(
            {
                "model_name": "TwiBot-20 profile+text live-compatible model",
                "threshold": 0.49,
                "live_distribution_warning": "Benchmark/live distribution may differ.",
            },
            None,
            _FakeModel(),
            list(range(50072)),
        ),
    )
    def test_endpoint_scores_without_network_features(self, *_):
        coroutine = application.analyze_twibot20(
            application.XAnalyzeRequest(username="example_user")
        )
        with self.assertRaises(StopIteration) as complete:
            coroutine.send(None)
        response = complete.exception.value
        payload = json.loads(response.body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["prediction"], "Likely Bot")
        self.assertIsNone(payload["probability"])
        self.assertIsNone(payload["confidence"])
        self.assertEqual(payload["model_score_uncalibrated"], 0.8)
        self.assertAlmostEqual(payload["class_scores_uncalibrated"]["Genuine"], 0.2)
        self.assertAlmostEqual(payload["class_scores_uncalibrated"]["Bot"], 0.8)
        self.assertFalse(payload["score_calibrated"])
        self.assertIn("Genuine=0 and Bot=1", payload["score_interpretation"])
        self.assertIn("sum to 1", payload["score_interpretation"])
        self.assertEqual(payload["threshold"], 0.49)
        self.assertEqual(payload["feature_count_constructed"], 50072)
        self.assertEqual(payload["feature_count_required"], 50072)
        self.assertEqual(payload["network_features_used"], False)
        self.assertEqual(payload["missing_feature_groups"], [])
        self.assertEqual(payload["feature_audit"]["constructed_feature_count"], 50072)
        self.assertEqual(payload["available_features"], [])
        self.assertEqual(payload["missing_features"], [])
        self.assertEqual(payload["explanation"], [])
        self.assertEqual(payload["lime_explanations"], [])
        self.assertEqual(payload["profile_summary"]["followers_count"], 1)
        self.assertEqual(payload["profile_summary"]["friends_count"], 2)
        self.assertEqual(payload["profile_summary"]["status_count"], 3)
        self.assertIsNotNone(payload["profile_summary"]["account_age_days"])

    def test_xfetch_profile_stats_and_age_are_serialized(self):
        response = self._call_endpoint(provider="xfetch")
        self.assertEqual(response.status_code, 200)
        profile = response.json()["profile_summary"]
        self.assertEqual(profile["followers_count"], 12484212)
        self.assertEqual(profile["friends_count"], 3)
        self.assertEqual(profile["status_count"], 31153)
        self.assertGreater(profile["account_age_days"], 6000)

    def test_genuine_classification_with_point_one_percent_bot_raw_score(self):
        response = self._call_endpoint(model_score=0.001)
        payload = response.json()
        self.assertEqual(payload["prediction"], "Likely Genuine")
        self.assertEqual(payload["model_score_uncalibrated"], 0.001)
        self.assertAlmostEqual(payload["class_scores_uncalibrated"]["Genuine"], 0.999)
        self.assertAlmostEqual(payload["class_scores_uncalibrated"]["Bot"], 0.001)
        self.assertIsNone(payload["probability"])
        self.assertIsNone(payload["confidence"])

    def test_bot_classification_with_point_seven_four_five_bot_score(self):
        response = self._call_endpoint(model_score=0.745)
        payload = response.json()
        self.assertEqual(payload["prediction"], "Likely Bot")
        self.assertEqual(payload["model_score_uncalibrated"], 0.745)
        self.assertAlmostEqual(payload["class_scores_uncalibrated"]["Genuine"], 0.255)
        self.assertAlmostEqual(payload["class_scores_uncalibrated"]["Bot"], 0.745)
        self.assertIsNone(payload["probability"])
        self.assertIsNone(payload["confidence"])

    def test_high_bot_score_class_scores_are_returned_independent_of_class_order(self):
        response = self._call_endpoint(model_score=0.869, model_classes=(1, 0))
        payload = response.json()
        self.assertEqual(payload["prediction"], "Likely Bot")
        self.assertEqual(payload["model_score_uncalibrated"], 0.869)
        self.assertAlmostEqual(payload["class_scores_uncalibrated"]["Genuine"], 0.131)
        self.assertAlmostEqual(payload["class_scores_uncalibrated"]["Bot"], 0.869)
        self.assertIsNone(payload["probability"])
        self.assertIsNone(payload["confidence"])

    def test_missing_model_classes_returns_structured_error_without_scores(self):
        payload = self._call_endpoint(model_classes=None).json()
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error_code"], "invalid_model_scores")
        self.assertIsNone(payload["model_score_uncalibrated"])
        self.assertEqual(payload["class_scores_uncalibrated"], {"Genuine": None, "Bot": None})

    def test_unrecognized_model_classes_fail_closed(self):
        payload = self._call_endpoint(model_classes=(0, 2)).json()
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error_code"], "invalid_model_scores")

    def test_invalid_model_score_outputs_fail_closed(self):
        invalid_outputs = (
            [[0.2, 0.7]],
            [[float("nan"), float("nan")]],
            [[0.2, 0.2]],
            [[1.1, -0.1]],
        )
        for outputs in invalid_outputs:
            with self.subTest(outputs=outputs):
                payload = self._call_endpoint(model_outputs=outputs).json()
                self.assertEqual(payload["status"], "error")
                self.assertEqual(payload["error_code"], "invalid_model_scores")

    def test_missing_profile_stats_are_not_reported_as_zero(self):
        self.assertEqual(
            application._profile_summary_metrics({}),
            {
                "account_age_days": None,
                "followers_count": None,
                "friends_count": None,
                "status_count": None,
            },
        )

    def test_feature_validation_failure_returns_structured_http_422(self):
        audit = {
            "expected_feature_count": 50072,
            "constructed_feature_count": 50072,
            "profile_feature_count": 56,
            "tweet_summary_feature_count": 16,
            "tfidf_feature_count": 50000,
            "missing_feature_count": 0,
            "duplicate_or_invalid_feature_count": 0,
            "feature_order_validated": False,
            "preprocessing_validated": True,
            "preprocessing_details": {},
        }
        response = self._call_endpoint(feature_audit=audit)
        self.assertEqual(response.status_code, 422)
        payload = response.json()
        self.assertEqual(payload["status"], "insufficient_features")
        self.assertEqual(payload["error_code"], "feature_validation_failed")
        self.assertEqual(payload["feature_audit"], audit)
        self.assertIsNone(payload["prediction"])
        self.assertEqual(payload["available_features"], [])
        self.assertEqual(payload["missing_features"], [])

    def test_artifact_failure_returns_structured_http_500(self):
        response = self._call_endpoint(artifact_error=RuntimeError("test failure"))
        self.assertEqual(response.status_code, 500)
        payload = response.json()
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error_code"], "artifact_validation_failed")
        self.assertIsNone(payload["prediction"])
        self.assertEqual(payload["available_features"], [])
        self.assertEqual(payload["missing_features"], [])
        self.assertEqual(payload["explanation"], [])
        self.assertEqual(payload["lime_explanations"], [])

    @patch.object(application.XFetchClient, "has_key", return_value=False)
    @patch.object(application.XApiClient, "has_token", return_value=True)
    @patch.object(
        application.XApiClient,
        "fetch_recent_tweets",
        return_value=([{"text": "a recent tweet"}], None),
    )
    @patch.object(
        application.XApiClient,
        "fetch_user_by_username",
        return_value=(
            {
                "id": "1",
                "name": "Example",
                "username": "example_user",
                "public_metrics": {
                    "followers_count": 1,
                    "following_count": 2,
                    "tweet_count": 3,
                    "listed_count": 0,
                },
                "verified": False,
                "description": "bio",
                "created_at": "2020-01-01T00:00:00Z",
            },
            None,
            200,
        ),
    )
    def test_endpoint_scores_with_packaged_model(self, *_):
        coroutine = application.analyze_twibot20(
            application.XAnalyzeRequest(username="example_user")
        )
        with self.assertRaises(StopIteration) as complete:
            coroutine.send(None)
        response = complete.exception.value
        payload = json.loads(response.body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["feature_count_constructed"], 50072)
        self.assertEqual(payload["feature_count_required"], 50072)
        self.assertIsNone(payload["probability"])
        self.assertTrue(np.isfinite(payload["model_score_uncalibrated"]))
        self.assertFalse(payload["score_calibrated"])
        self.assertIn(payload["prediction"], {"Likely Bot", "Likely Genuine"})
        self.assertFalse(payload["network_features_used"])

    def test_route_is_registered(self):
        self.assertIn(
            "/twibot20/analyze",
            {route.path for route in application.app.routes},
        )

    def test_twibot20_and_cresci_dashboards_are_separate(self):
        root_html = TestClient(application.app).get("/").text
        legacy = application.web_dashboard()
        with self.assertRaises(StopIteration) as legacy_done:
            legacy.send(None)
        legacy_html = legacy_done.exception.value.body.decode()
        self.assertIn("<title>TwiBot-20 | Research Dashboard</title>", root_html)
        self.assertIn("Social bot analysis", root_html)
        self.assertIn("/twibot20/analyze", root_html)
        self.assertIn("50,072 features", root_html)
        self.assertIn("Model &amp; Feature Validation", root_html)
        self.assertNotIn("canonical 34 features", root_html)
        self.assertIn('href="#model-information"', root_html)
        self.assertIn('href="#evaluation-results"', root_html)
        self.assertIn('href="/cresci15"', root_html)
        self.assertIn("Calibrated probability unavailable.", root_html)
        self.assertIn("Uncalibrated model scores", root_html)
        self.assertIn("Likely Genuine: —", root_html)
        self.assertIn("Likely Bot: —", root_html)
        self.assertIn("class_scores_uncalibrated", root_html)
        self.assertIn("@media (max-width: 680px)", root_html)
        self.assertIn("Interpretable AI Social Bot Detection", legacy_html)
        self.assertIn("/analyze_x", legacy_html)
        self.assertIn("canonical 34 features", legacy_html)
        self.assertIn("data.available_features.length", legacy_html)

    def test_minimal_optional_response_fields_do_not_break_dashboard_rendering(self):
        html = self._twibot_analysis_function()
        self.assertIn("data.profile_summary && typeof data.profile_summary === 'object'", html)
        self.assertIn("data.ood_validation && Array.isArray(data.ood_validation.audit_warnings)", html)
        self.assertIn("const values = audit && typeof audit === 'object' ? audit : {}", html)
        self.assertIn("Analysis service returned an invalid response", html)
        self.assertNotIn("data.available_features.length", html)
        self.assertNotIn("data.probabilities.Bot", html)

    def test_both_class_scores_render_only_when_valid_and_normalized(self):
        html = self._twibot_analysis_function()
        self.assertIn("const genuineScore = classScores.Genuine", html)
        self.assertIn("const botScore = classScores.Bot", html)
        self.assertIn("Math.abs(genuineScore + botScore - 1) <= 1e-6", html)
        self.assertIn("Math.abs(botScore - score) <= 1e-6", html)
        self.assertIn("const genuinePercent = genuineScore * 100", html)
        self.assertIn("const botPercent = botScore * 100", html)
        self.assertIn("Likely Genuine: ${genuinePercent.toFixed(1)}%", html)
        self.assertIn("Likely Bot: ${botPercent.toFixed(1)}%", html)
        self.assertIn("Uncalibrated model scores from predict_proba", html)
        self.assertNotIn("data.probabilities.Human", html)
        self.assertNotIn("data.probabilities.Bot", html)

    def test_missing_class_scores_hide_bar_and_report_unavailable(self):
        html = self._twibot_analysis_function()
        self.assertIn("rawBarWrap').style.display = validScore ? 'block' : 'none'", html)
        self.assertIn("Calibrated probability unavailable.", html)

    def test_invalid_class_scores_hide_bar(self):
        html = self._twibot_analysis_function()
        self.assertIn("Number.isFinite(genuineScore)", html)
        self.assertIn("Number.isFinite(botScore)", html)

    def test_out_of_range_or_mismatched_class_scores_hide_bar(self):
        html = self._twibot_analysis_function()
        self.assertIn("genuineScore >= 0 && genuineScore <= 1", html)
        self.assertIn("botScore >= 0 && botScore <= 1", html)
        self.assertIn("Math.abs(botScore - score) <= 1e-6", html)


if __name__ == "__main__":
    unittest.main()
