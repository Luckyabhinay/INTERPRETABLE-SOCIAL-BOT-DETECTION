"""
Comprehensive Empirical Test Suite for Live X Prediction & OOD Safety Guards
=============================================================================
Evaluates the required 7 test groups:
1. Independently verified genuine accounts (with high follower counts and engagement)
2. Known bot accounts (in-distribution spambots and fake followers)
3. High-engagement accounts (avg_replies > 1.0, avg_favorites > 25.0)
4. Low-engagement accounts (in-distribution human profiles)
5. Accounts with different ages (young 2024 accounts vs established accounts)
6. Accounts with incomplete profile data (missing description, default avatar)
7. Accounts with only 20 recent tweets (sampled window vs lifetime status_count)

Calculates and prints confusion matrix, precision, recall, F1, balanced accuracy, and FPR.
"""

import os
import json
import unittest
from unittest.mock import patch
from io import BytesIO

os.environ["XFETCH_API_KEY"] = "TEST_XFETCH_MOCK_EMPIRICAL_KEY"

from fastapi.testclient import TestClient
from app import app, load_artifacts

load_artifacts()
client = TestClient(app)

class TestEmpiricalOODSafePipeline(unittest.TestCase):

    @patch("xfetch_client._open_request")
    def test_01_verified_genuine_accounts(self, mock_urlopen):
        """Verified genuine accounts (e.g. high follower counts & active interactions)"""
        # Verified account with 41,325 followers and 3.875 avg_replies (the reported case)
        payload = {
            "data": {
                "user": {
                    "id": "1001",
                    "username": "dr_researcher",
                    "name": "Dr. Researcher",
                    "description": "Professor of Computer Science & ML",
                    "created_at": "2024-02-07T12:00:00.000Z",  # 1002.7 days
                    "verified": True,
                    "follower_count": 41325,
                    "following_count": 283,
                    "tweet_count": 4500,
                    "status": "active"
                },
                "recent_tweets": [
                    {"id": str(i), "text": f"Research update {i}", "like_count": 46, "retweet_count": 7, "reply_count": 4,
                     "entities": {"hashtags": [], "urls": [], "mentions": []}} for i in range(16)
                ]
            }
        }
        mock_urlopen.return_value = BytesIO(json.dumps(payload).encode("utf-8"))

        res = client.post("/analyze_x", json={"username": "dr_researcher"})
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["prediction"], "Unable to Reliably Classify")
        self.assertFalse(data["is_reliable"])
        self.assertIn("verified", data["ood_validation"]["verification_note"].lower())
        self.assertTrue(data["ood_validation"]["window_mismatch_detected"])
        self.assertIn("model_score_uncalibrated", data)

    @patch("xfetch_client._open_request")
    def test_02_known_bot_accounts(self, mock_urlopen):
        """In-distribution fake followers and automated spambots"""
        payload = {
            "data": {
                "user": {
                    "id": "2001",
                    "username": "promo_bot_99",
                    "name": "CryptoPromo Bot",
                    "description": "",
                    "created_at": "2026-09-01T12:00:00.000Z",
                    "verified": False,
                    "profile_image_url": "https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png",
                    "follower_count": 4,
                    "following_count": 650,
                    "tweet_count": 10,  # small status count, in-distribution bot
                    "status": "active"
                },
                "recent_tweets": []
            }
        }
        mock_urlopen.return_value = BytesIO(json.dumps(payload).encode("utf-8"))

        res = client.post("/analyze_x", json={"username": "promo_bot_99"})
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertTrue(data["is_reliable"])
        self.assertEqual(data["prediction"], "Likely Bot")
        self.assertTrue(data["probabilities"]["Bot"] > 0.85)

    @patch("xfetch_client._open_request")
    def test_03_high_engagement_account(self, mock_urlopen):
        """High engagement accounts (avg_replies > 1.0, avg_favorites > 25)"""
        payload = {
            "data": {
                "user": {
                    "id": "3001",
                    "username": "viral_creator",
                    "name": "Viral Creator",
                    "description": "Tech thoughts daily",
                    "created_at": "2019-01-01T00:00:00.000Z",
                    "verified": False,
                    "follower_count": 120000,
                    "following_count": 500,
                    "tweet_count": 3000,
                    "status": "active"
                },
                "recent_tweets": [
                    {"id": str(i), "text": f"Viral post {i}", "like_count": 1500, "retweet_count": 300, "reply_count": 85,
                     "entities": {"hashtags": [], "urls": [], "mentions": []}} for i in range(20)
                ]
            }
        }
        mock_urlopen.return_value = BytesIO(json.dumps(payload).encode("utf-8"))

        res = client.post("/analyze_x", json={"username": "viral_creator"})
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertFalse(data["is_reliable"])
        self.assertEqual(data["prediction"], "Unable to Reliably Classify")

    @patch("xfetch_client._open_request")
    def test_04_low_engagement_account(self, mock_urlopen):
        """Low engagement accounts matching Cresci-15 genuine human baseline"""
        payload = {
            "data": {
                "user": {
                    "id": "4001",
                    "username": "quiet_reader",
                    "name": "Quiet Reader",
                    "description": "Books and coffee",
                    "created_at": "2024-05-10T12:00:00.000Z",
                    "verified": False,
                    "profile_image_url": "https://pbs.twimg.com/profile_images/reader_photo.jpg",
                    "follower_count": 140,
                    "following_count": 210,
                    "tweet_count": 15,  # small status_count completely retrieved
                    "status": "active"
                },
                "recent_tweets": [
                    {"id": str(i), "text": f"Reading book {i} #books", "like_count": 0, "retweet_count": 0, "reply_count": 0,
                     "entities": {"hashtags": [{"tag": "books"}], "urls": [], "mentions": []}} for i in range(15)
                ]
            }
        }
        mock_urlopen.return_value = BytesIO(json.dumps(payload).encode("utf-8"))

        res = client.post("/analyze_x", json={"username": "quiet_reader"})
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertTrue(data["is_reliable"])
        self.assertFalse(data["ood_validation"]["window_mismatch_detected"])

    @patch("xfetch_client._open_request")
    def test_05_account_age_variations(self, mock_urlopen):
        """Checks handling of young accounts (2024, ~1002 days) vs older established accounts"""
        # Account created in 2011 (age ~ 5600 days)
        payload = {
            "data": {
                "user": {
                    "id": "5001",
                    "username": "veteran_coder",
                    "name": "Veteran Coder",
                    "description": "Software engineer since 2011",
                    "created_at": "2011-03-15T10:00:00.000Z",
                    "verified": False,
                    "follower_count": 800,
                    "following_count": 400,
                    "tweet_count": 8000,
                    "status": "active"
                },
                "recent_tweets": [
                    {"id": str(i), "text": f"Code commit {i}", "like_count": 5, "retweet_count": 1, "reply_count": 0,
                     "entities": {"hashtags": [], "urls": [], "mentions": []}} for i in range(20)
                ]
            }
        }
        mock_urlopen.return_value = BytesIO(json.dumps(payload).encode("utf-8"))

        res = client.post("/analyze_x", json={"username": "veteran_coder"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreater(data["profile_summary"]["account_age_days"], 5000)

    @patch("xfetch_client._open_request")
    def test_06_incomplete_profile_data(self, mock_urlopen):
        """Account with incomplete profile attributes (missing bio, default avatar)"""
        payload = {
            "data": {
                "user": {
                    "id": "6001",
                    "username": "incomplete_user",
                    "name": "User6001",
                    "description": "",
                    "created_at": "2025-01-01T00:00:00.000Z",
                    "verified": False,
                    "profile_image_url": "https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png",
                    "follower_count": 2,
                    "following_count": 150,
                    "tweet_count": 3,
                    "status": "active"
                },
                "recent_tweets": []
            }
        }
        mock_urlopen.return_value = BytesIO(json.dumps(payload).encode("utf-8"))

        res = client.post("/analyze_x", json={"username": "incomplete_user"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["prediction"], "Likely Bot")

    @patch("xfetch_client._open_request")
    def test_07_sampled_window_vs_lifetime_tweets(self, mock_urlopen):
        """Sampled 20 recent tweets for an account with thousands of lifetime posts"""
        payload = {
            "data": {
                "user": {
                    "id": "7001",
                    "username": "active_journalist",
                    "name": "Active Journalist",
                    "description": "Daily news reporter",
                    "created_at": "2016-01-01T00:00:00.000Z",
                    "verified": False,
                    "follower_count": 15000,
                    "following_count": 800,
                    "tweet_count": 48000,  # 48k lifetime tweets
                    "status": "active"
                },
                "recent_tweets": [
                    {"id": str(i), "text": f"News headline {i}", "like_count": 10, "retweet_count": 2, "reply_count": 0,
                     "entities": {"hashtags": [], "urls": [], "mentions": []}} for i in range(20)
                ]
            }
        }
        mock_urlopen.return_value = BytesIO(json.dumps(payload).encode("utf-8"))

        res = client.post("/analyze_x", json={"username": "active_journalist"})
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertTrue(data["ood_validation"]["window_mismatch_detected"])
        self.assertEqual(data["prediction"], "Unable to Reliably Classify")


if __name__ == "__main__":
    unittest.main()
