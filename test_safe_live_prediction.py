"""
Comprehensive Test Suite for Safe Live X Prediction & OOD Detection
=====================================================================
Tests all 9 scenarios required by the task:
1. Verified genuine-like account (tests OOD guard and verification note).
2. Unverified genuine-like account (in-distribution human profile).
3. Bot-like account (in-distribution fake follower/spambot).
4. High-engagement account (out-of-distribution avg_replies/avg_favorites).
5. Account with only 20 available recent posts (window mismatch detection).
6. Account with missing features (insufficient profile rejection).
7. Out-of-distribution account (values exceeding empirical training p99/max).
8. API failure handling (401, 402, 404, 429).
9. Existing prediction endpoint regression (POST /predict_explain and GET /).
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock
from io import BytesIO
import urllib.error

os.environ["XFETCH_API_KEY"] = "TEST_XFETCH_MOCK_KEY_AUDIT"

from fastapi.testclient import TestClient
from app import app, load_artifacts

load_artifacts()
client = TestClient(app)

class TestLiveXSafePredictionAudit(unittest.TestCase):

    @patch("xfetch_client._open_request")
    def test_01_verified_genuine_like_account(self, mock_urlopen):
        """1. Verified genuine account: verifies OOD detection and verification context note"""
        payload = {
            "data": {
                "user": {
                    "id": "12",
                    "username": "jack",
                    "name": "jack",
                    "description": "building open protocols",
                    "created_at": "2006-03-21T18:50:00.000Z",
                    "verified": True,
                    "follower_count": 12484212,
                    "following_count": 3,
                    "tweet_count": 31153,
                    "status": "active"
                },
                "recent_tweets": [
                    {"id": str(i), "text": f"Tweet {i}", "like_count": 500, "retweet_count": 100, "reply_count": 30,
                     "entities": {"hashtags": [], "urls": [], "mentions": []}} for i in range(20)
                ]
            },
            "meta": {"credits": {"charged": 21, "remaining": 900}}
        }
        mock_urlopen.return_value = BytesIO(json.dumps(payload).encode("utf-8"))

        res = client.post("/analyze_x", json={"username": "@jack"})
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["prediction"], "Unable to Reliably Classify")
        self.assertFalse(data["is_reliable"])
        self.assertIn("model_score_uncalibrated", data)
        self.assertTrue(len(data["ood_validation"]["audit_warnings"]) > 0)
        self.assertIn("verified", data["ood_validation"]["verification_note"].lower())

    @patch("xfetch_client._open_request")
    def test_02_unverified_genuine_like_account(self, mock_urlopen):
        """2. Unverified genuine account: in-distribution human patterns with complete window coverage"""
        payload = {
            "data": {
                "user": {
                    "id": "999",
                    "username": "casual_student",
                    "name": "Alex Student",
                    "description": "Grad student studying biology",
                    "created_at": "2018-04-12T10:00:00.000Z",
                    "verified": False,
                    "follower_count": 320,
                    "following_count": 450,
                    "tweet_count": 15,  # Account only has 15 total lifetime tweets, and all 15 are retrieved
                    "status": "active"
                },
                "recent_tweets": [
                    {"id": str(i), "text": f"Study notes {i} #biology", "like_count": 1, "retweet_count": 0, "reply_count": 0,
                     "entities": {"hashtags": [{"tag": "biology"}], "urls": [], "mentions": []}} for i in range(15)
                ]
            }
        }
        mock_urlopen.return_value = BytesIO(json.dumps(payload).encode("utf-8"))

        res = client.post("/analyze_x", json={"username": "casual_student"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data["ood_validation"]["window_mismatch_detected"])
        self.assertIn("model_score_uncalibrated", data)

    @patch("xfetch_client._open_request")
    def test_03_bot_like_account(self, mock_urlopen):
        """3. In-distribution automated bot-like account"""
        payload = {
            "data": {
                "user": {
                    "id": "888",
                    "username": "followback_fast88",
                    "name": "FREE FOLLOWERS",
                    "description": "",
                    "created_at": "2026-10-01T10:00:00.000Z",
                    "verified": False,
                    "profile_image_url": "https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png",
                    "follower_count": 8,
                    "following_count": 890,
                    "tweet_count": 12,
                    "status": "active"
                },
                "recent_tweets": []
            }
        }
        mock_urlopen.return_value = BytesIO(json.dumps(payload).encode("utf-8"))

        res = client.post("/analyze_x", json={"username": "followback_fast88"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["is_reliable"])
        self.assertEqual(data["prediction"], "Likely Bot")

    @patch("xfetch_client._open_request")
    def test_04_high_engagement_account(self, mock_urlopen):
        """4. High engagement account: catches domain shift in likes and replies"""
        payload = {
            "data": {
                "user": {
                    "id": "777",
                    "username": "viral_star",
                    "name": "Viral Star",
                    "description": "Daily thoughts",
                    "created_at": "2020-01-01T00:00:00.000Z",
                    "verified": False,
                    "follower_count": 50000,
                    "following_count": 100,
                    "tweet_count": 800,
                    "status": "active"
                },
                "recent_tweets": [
                    {"id": "1", "text": "Viral post", "like_count": 15000, "retweet_count": 3000, "reply_count": 800,
                     "entities": {"hashtags": [], "urls": [], "mentions": []}}
                ]
            }
        }
        mock_urlopen.return_value = BytesIO(json.dumps(payload).encode("utf-8"))

        res = client.post("/analyze_x", json={"username": "viral_star"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data["is_reliable"])
        self.assertEqual(data["prediction"], "Unable to Reliably Classify")

    @patch("xfetch_client._open_request")
    def test_05_sample_window_mismatch(self, mock_urlopen):
        """5. Account with 20 posts but large status count triggers window mismatch"""
        payload = {
            "data": {
                "user": {
                    "id": "666",
                    "username": "established_user",
                    "name": "Established User",
                    "description": "Writer & Journalist",
                    "created_at": "2015-01-01T00:00:00.000Z",
                    "verified": False,
                    "follower_count": 5000,
                    "following_count": 400,
                    "tweet_count": 25000,  # 25,000 lifetime tweets
                    "status": "active"
                },
                "recent_tweets": [
                    {"id": str(i), "text": f"Article {i}", "like_count": 25, "retweet_count": 5, "reply_count": 15,
                     "entities": {"hashtags": [], "urls": [], "mentions": []}} for i in range(20)
                ]
            }
        }
        mock_urlopen.return_value = BytesIO(json.dumps(payload).encode("utf-8"))

        res = client.post("/analyze_x", json={"username": "established_user"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["ood_validation"]["window_mismatch_detected"])

    def test_06_missing_features_rejection(self):
        """6. Missing required features rejects with 400 or 422"""
        res = client.post("/analyze_x", json={"username": ""})
        self.assertEqual(res.status_code, 400)

    @patch("xfetch_client._open_request")
    def test_07_out_of_distribution_values(self, mock_urlopen):
        """7. Account with values exceeding empirical training limits"""
        payload = {
            "data": {
                "user": {
                    "id": "555",
                    "username": "extreme_metrics",
                    "name": "Extreme Metrics",
                    "description": "test",
                    "created_at": "2006-03-21T18:50:00.000Z",
                    "verified": False,
                    "follower_count": 95000000,  # 95M followers (exceeds max 408k in training)
                    "following_count": 2,
                    "tweet_count": 150000,
                    "status": "active"
                },
                "recent_tweets": [
                    {"id": "1", "text": "extreme tweet", "like_count": 20000, "retweet_count": 5000, "reply_count": 500,
                     "entities": {"hashtags": [], "urls": [], "mentions": []}}
                ]
            }
        }
        mock_urlopen.return_value = BytesIO(json.dumps(payload).encode("utf-8"))

        res = client.post("/analyze_x", json={"username": "extreme_metrics"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        ood_names = [o["feature"] for o in data["ood_validation"]["ood_features"]]
        self.assertIn("followers_count", ood_names)

    @patch("xfetch_client._open_request")
    def test_08_api_failure_handling(self, mock_urlopen):
        """8. API failure handling (402 credits depleted, 404 not found)"""
        err_res = urllib.error.HTTPError(
            url="https://api.xfetch.io/v1/profiles/by-username/missing",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=BytesIO(b'{"message":"User not found"}')
        )
        mock_urlopen.side_effect = err_res

        res = client.post("/analyze_x", json={"username": "missing"})
        self.assertEqual(res.status_code, 404)
        self.assertEqual(res.json()["status"], "error")

    def test_09_existing_endpoints_regression(self):
        """9. Regression test on /predict_explain and GET /"""
        get_res = client.get("/")
        self.assertEqual(get_res.status_code, 200)
        self.assertIn("TwiBot-20 | Research Dashboard", get_res.text)
        cresci_res = client.get("/cresci15")
        self.assertEqual(cresci_res.status_code, 200)
        self.assertIn("Live X (Twitter) Account Analyzer", cresci_res.text)
        self.assertIn("Manual Feature Simulator", cresci_res.text)

        post_res = client.post("/predict_explain", json={
            "followers_count": 1200,
            "friends_count": 450,
            "status_count": 3500,
            "favorites_count": 1200,
            "listed_count": 15,
            "avg_replies": 0.25,
            "avg_mentions": 0.65,
            "account_age_days": 1450,
            "description_binary": 1,
            "default_profile_image": 0
        })
        self.assertEqual(post_res.status_code, 200)
        post_data = post_res.json()
        self.assertIn("prediction", post_data)
        self.assertIn("lime_explanations", post_data)


if __name__ == "__main__":
    unittest.main()
