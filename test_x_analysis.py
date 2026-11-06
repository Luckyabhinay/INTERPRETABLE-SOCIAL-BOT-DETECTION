"""
Comprehensive Test Suite for Live X (Twitter) Account Analysis
==============================================================
Validates:
1. Username sanitization (URLs, @handles, clean names, invalid formats)
2. Valid X account lookup with mocked profile + tweets
3. Valid X profile URL lookup
4. Invalid username format handling
5. Account not found (404) handling
6. API authentication failure (401 / missing token)
7. API rate limit (429) handling
8. Network timeout handling
9. Missing profile / tweet field handling & imputation transparency
10. Exact 34 model feature compatibility & SHAP/LIME explanation format
11. Regression test ensuring existing /predict_explain and GET / are 100% operational
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock
from io import BytesIO
import urllib.error

# Set dummy token for unit testing
os.environ["X_BEARER_TOKEN"] = "TEST_MOCK_BEARER_TOKEN_ABC123"
os.environ.pop("XFETCH_API_KEY", None)

from fastapi.testclient import TestClient
from app import app, load_artifacts
from x_api_client import sanitize_username, XApiClient, extract_features_from_x_data

# Trigger startup event to load models and explainers
load_artifacts()
client = TestClient(app)

MOCK_USER_GENUINE = {
    "id": "12345678",
    "name": "Jane Researcher",
    "username": "janeresearcher",
    "created_at": "2018-05-10T12:00:00.000Z",
    "description": "AI & Computational Social Science researcher. Building transparent models.",
    "location": "Boston, MA",
    "url": "https://janeresearcher.org",
    "profile_image_url": "https://pbs.twimg.com/profile_images/real_photo.jpg",
    "verified": False,
    "pinned_tweet_id": "99887766",
    "public_metrics": {
        "followers_count": 2450,
        "following_count": 520,
        "tweet_count": 6800,
        "listed_count": 48
    }
}

MOCK_USER_BOT = {
    "id": "87654321",
    "name": "CryptoPromo998",
    "username": "cryptopromo998",
    "created_at": "2026-10-01T04:15:00.000Z",
    "description": "",
    "location": "",
    "profile_image_url": "https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png",
    "verified": False,
    "public_metrics": {
        "followers_count": 14,
        "following_count": 890,
        "tweet_count": 45,
        "listed_count": 0
    }
}

MOCK_TWEETS = [
    {
        "id": "101",
        "text": "Excited to share our new research paper on explainable AI models! Check it out.",
        "created_at": "2026-10-25T14:30:00.000Z",
        "public_metrics": {"retweet_count": 15, "reply_count": 4, "like_count": 65},
        "entities": {
            "hashtags": [{"tag": "AI"}, {"tag": "MachineLearning"}],
            "mentions": [{"username": "conference_ai"}],
            "urls": [{"expanded_url": "https://arxiv.org/abs/1234"}]
        }
    },
    {
        "id": "102",
        "text": "Great discussions today on model interpretability and fairness.",
        "created_at": "2026-10-24T18:00:00.000Z",
        "public_metrics": {"retweet_count": 8, "reply_count": 2, "like_count": 30},
        "entities": {
            "hashtags": [],
            "mentions": [],
            "urls": []
        }
    }
]


class TestLiveXAccountAnalysis(unittest.TestCase):

    def test_01_username_sanitization(self):
        """Test URL, @handle, and text normalization"""
        self.assertEqual(sanitize_username("@testuser"), "testuser")
        self.assertEqual(sanitize_username("https://x.com/sample_user"), "sample_user")
        self.assertEqual(sanitize_username("https://twitter.com/dev_user/"), "dev_user")
        self.assertEqual(sanitize_username("http://x.com/another_handle?query=1"), "another_handle")
        self.assertEqual(sanitize_username("clean_user"), "clean_user")

    @patch("urllib.request.urlopen")
    def test_02_valid_username_analysis(self, mock_urlopen):
        """Test full pipeline for valid X username with mocked API response"""
        # Mock user response and tweets response
        user_resp = BytesIO(json.dumps({"data": MOCK_USER_GENUINE}).encode("utf-8"))
        tweets_resp = BytesIO(json.dumps({"data": MOCK_TWEETS}).encode("utf-8"))
        mock_urlopen.side_effect = [user_resp, tweets_resp]

        resp = client.post("/analyze_x", json={"username": "@janeresearcher"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["status"], "success")
        self.assertEqual(data["platform"], "X")
        self.assertEqual(data["username"], "janeresearcher")
        self.assertIn(data["prediction"], ["Likely Genuine", "Likely Bot"])
        self.assertIn("confidence", data)
        self.assertIn("probabilities", data)
        self.assertIn("Human", data["probabilities"])
        self.assertIn("Bot", data["probabilities"])
        self.assertIn("profile_summary", data)
        self.assertEqual(data["profile_summary"]["followers_count"], 2450)
        self.assertTrue(len(data["available_features"]) >= 10)
        self.assertTrue(len(data["missing_features"]) > 0)
        self.assertIn("explanation", data)  # SHAP
        self.assertIn("lime_explanations", data)  # LIME
        self.assertIn("scientific_disclaimer", data)

    @patch("urllib.request.urlopen")
    def test_03_valid_x_url_input(self, mock_urlopen):
        """Test URL inputs (x.com or twitter.com) are properly extracted and analyzed"""
        user_resp = BytesIO(json.dumps({"data": MOCK_USER_GENUINE}).encode("utf-8"))
        tweets_resp = BytesIO(json.dumps({"data": []}).encode("utf-8"))
        mock_urlopen.side_effect = [user_resp, tweets_resp]

        resp = client.post("/analyze_x", json={"username": "https://x.com/janeresearcher"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["username"], "janeresearcher")
        self.assertEqual(data["status"], "success")

    def test_04_invalid_username_format(self):
        """Test invalid characters or empty string reject cleanly with 400"""
        resp = client.post("/analyze_x", json={"username": ""})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["status"], "error")

        resp_invalid = client.post("/analyze_x", json={"username": "invalid handle with spaces!!!"})
        self.assertIn(resp_invalid.status_code, [400, 422])
        self.assertEqual(resp_invalid.json()["status"], "error")

    @patch("urllib.request.urlopen")
    def test_05_account_not_found_404(self, mock_urlopen):
        """Test 404 account not found returned from X API"""
        err_response = urllib.error.HTTPError(
            url="https://api.twitter.com/2/users/by/username/nonexistent_user_999",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=BytesIO(b'{"errors":[{"title":"Not Found","detail":"User not found"}]}')
        )
        mock_urlopen.side_effect = err_response

        resp = client.post("/analyze_x", json={"username": "nonexistent_user_999"})
        self.assertEqual(resp.status_code, 404)
        data = resp.json()
        self.assertEqual(data["status"], "error")
        self.assertIn("not found", data["message"].lower())

    @patch("urllib.request.urlopen")
    def test_06_api_auth_failure_401(self, mock_urlopen):
        """Test invalid or expired API Bearer Token handling"""
        err_response = urllib.error.HTTPError(
            url="https://api.twitter.com/2/users/by/username/someone",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=BytesIO(b'{"title":"Unauthorized","detail":"Unauthorized"}')
        )
        mock_urlopen.side_effect = err_response

        resp = client.post("/analyze_x", json={"username": "someone"})
        self.assertEqual(resp.status_code, 401)
        data = resp.json()
        self.assertEqual(data["status"], "error")
        self.assertIn("Authentication failed", data["message"])

    @patch("urllib.request.urlopen")
    def test_07_api_rate_limit_429(self, mock_urlopen):
        """Test 429 Too Many Requests handling"""
        err_response = urllib.error.HTTPError(
            url="https://api.twitter.com/2/users/by/username/ratelimited",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=BytesIO(b'{"title":"Too Many Requests"}')
        )
        mock_urlopen.side_effect = err_response

        resp = client.post("/analyze_x", json={"username": "ratelimited"})
        self.assertEqual(resp.status_code, 429)
        data = resp.json()
        self.assertEqual(data["status"], "error")
        self.assertIn("Rate limit", data["message"])

    @patch("urllib.request.urlopen")
    def test_08_network_timeout_504(self, mock_urlopen):
        """Test network timeout during API call"""
        mock_urlopen.side_effect = urllib.error.URLError("Connection timed out")

        resp = client.post("/analyze_x", json={"username": "timeout_user"})
        self.assertEqual(resp.status_code, 504)
        data = resp.json()
        self.assertEqual(data["status"], "error")
        self.assertIn("Network timeout", data["message"])

    @patch("urllib.request.urlopen")
    def test_09_bot_prediction_and_imputation(self, mock_urlopen):
        """Test bot account detection and verify imputed vs available features"""
        user_resp = BytesIO(json.dumps({"data": MOCK_USER_BOT}).encode("utf-8"))
        # Empty tweets (e.g. Free API tier does not provide tweets)
        tweets_resp = BytesIO(json.dumps({"data": []}).encode("utf-8"))
        mock_urlopen.side_effect = [user_resp, tweets_resp]

        resp = client.post("/analyze_x", json={"username": "cryptopromo998"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["prediction"], "Likely Bot")
        self.assertTrue(data["probabilities"]["Bot"] > 0.50)
        self.assertIn("account_age_days", data["available_features"])
        self.assertIn("followers_count", data["available_features"])
        # Features unavailable without tweets or deprecated in API v2 must be listed in missing_features
        self.assertIn("geo_enabled", data["missing_features"])
        self.assertIn("profile_background_tile", data["missing_features"])

    @patch("urllib.request.urlopen")
    def test_10_shap_lime_explanations_structure(self, mock_urlopen):
        """Test that SHAP and LIME outputs contain valid structure, weights, and directional labels"""
        user_resp = BytesIO(json.dumps({"data": MOCK_USER_GENUINE}).encode("utf-8"))
        tweets_resp = BytesIO(json.dumps({"data": MOCK_TWEETS}).encode("utf-8"))
        mock_urlopen.side_effect = [user_resp, tweets_resp]

        resp = client.post("/analyze_x", json={"username": "janeresearcher"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        # Check SHAP explanation items
        self.assertTrue(len(data["explanation"]) > 0)
        for item in data["explanation"]:
            self.assertIn("feature", item)
            self.assertIn("value", item)
            self.assertIn("attribution", item)
            self.assertIn("direction", item)
            self.assertIn("Towards", item["direction"])

        # Check LIME explanation items
        self.assertTrue(len(data["lime_explanations"]) > 0)
        for item in data["lime_explanations"]:
            self.assertIn("feature_rule", item)
            self.assertIn("weight", item)
            self.assertIn("direction", item)

    def test_11_regression_existing_endpoints(self):
        """Ensure existing web dashboard GET / and POST /predict_explain remain 100% operational"""
        get_resp = client.get("/")
        self.assertEqual(get_resp.status_code, 200)
        self.assertIn("Interpretable AI Social Bot Detection", get_resp.text)
        self.assertIn("Live X (Twitter) Account Analyzer", get_resp.text)
        self.assertIn("Manual Feature Simulator", get_resp.text)

        post_resp = client.post("/predict_explain", json={
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
        self.assertEqual(post_resp.status_code, 200)
        post_data = post_resp.json()
        self.assertIn("prediction", post_data)
        self.assertIn("confidence", post_data)
        self.assertIn("probabilities", post_data)
        self.assertIn("lime_explanations", post_data)


if __name__ == "__main__":
    unittest.main()

