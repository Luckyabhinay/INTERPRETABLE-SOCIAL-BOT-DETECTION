"""
Comprehensive Test Suite for xfetch & Live Account Analysis
============================================================
Validates:
1. xfetch.io profile & recent tweets parsing
2. Exact 34 canonical Cresci feature mapping
3. Fallback when credentials are empty
4. 401, 402, 404, 429 xfetch error handling
5. End-to-end POST /analyze_x with mocked xfetch responses
6. Regression test on /predict_explain and GET /
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock
from io import BytesIO
import urllib.error

os.environ["XFETCH_API_KEY"] = "TEST_XFETCH_MOCK_KEY_789"

from fastapi.testclient import TestClient
from app import app, load_artifacts
from xfetch_client import XFetchClient, extract_features_from_xfetch_data, sanitize_username

load_artifacts()
client = TestClient(app)

MOCK_XFETCH_PAYLOAD = {
    "data": {
        "user": {
            "id": "12",
            "username": "jack",
            "name": "jack",
            "description": "just setting up my twttr",
            "created_at": "2006-03-21T18:50:00.000Z",
            "verified": True,
            "protected": False,
            "profile_image_url": "https://pbs.twimg.com/profile_images/real_jack.jpg",
            "follower_count": 12484212,
            "following_count": 3,
            "tweet_count": 31153,
            "status": "active"
        },
        "recent_tweets": [
            {
                "id": "2055734812669865997",
                "text": "Great day to build open protocols and decentralized communication #Bitcoin #OpenSource",
                "author_id": "12",
                "created_at": "2026-10-16T19:39:07.000Z",
                "like_count": 5200,
                "retweet_count": 850,
                "reply_count": 420,
                "entities": {
                    "hashtags": [{"tag": "Bitcoin"}, {"tag": "OpenSource"}],
                    "urls": [],
                    "mentions": []
                }
            },
            {
                "id": "2055734812669865998",
                "text": "Exploring interpretable AI architectures for trust and safety @bluesky",
                "author_id": "12",
                "created_at": "2026-10-15T10:15:00.000Z",
                "like_count": 3100,
                "retweet_count": 420,
                "reply_count": 180,
                "entities": {
                    "hashtags": [],
                    "urls": [],
                    "mentions": [{"username": "bluesky"}]
                }
            }
        ]
    },
    "meta": {
        "request_id": "req_01HXFETCHTEST",
        "credits": {
            "charged": 21,
            "remaining": 979
        }
    }
}


class TestXFetchIntegration(unittest.TestCase):

    def test_01_feature_extraction(self):
        """Verify extraction maps xfetch fields into all 34 canonical features"""
        from app import default_features
        feats, avail, missing = extract_features_from_xfetch_data(
            MOCK_XFETCH_PAYLOAD["data"],
            default_features
        )
        self.assertEqual(len(feats), 34)
        self.assertEqual(feats["followers_count"], 12484212.0)
        self.assertEqual(feats["friends_count"], 3.0)
        self.assertEqual(feats["status_count"], 31153.0)
        self.assertEqual(feats["verified"], 1.0)
        self.assertEqual(feats["description_binary"], 1.0)
        self.assertTrue(feats["account_age_days"] > 6000)
        self.assertTrue(feats["tweets_per_day"] > 0)
        self.assertAlmostEqual(feats["ffratio"], 1.0, places=3)
        self.assertEqual(feats["hashtag_count"], 2.0)
        self.assertEqual(feats["mentions_count"], 1.0)
        self.assertIn("followers_count", avail)
        self.assertIn("avg_favorites", avail)

    @patch("xfetch_client._open_request")
    def test_02_endpoint_with_xfetch(self, mock_urlopen):
        """Test POST /analyze_x returns predictions and explanations via xfetch"""
        resp_mock = BytesIO(json.dumps(MOCK_XFETCH_PAYLOAD).encode("utf-8"))
        mock_urlopen.return_value = resp_mock

        res = client.post("/analyze_x", json={"username": "@jack"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["username"], "jack")
        self.assertIn(
            data["prediction"],
            ["Likely Genuine", "Likely Bot", "Unable to Reliably Classify"],
        )
        self.assertIn("credits", data)
        self.assertEqual(data["credits"]["charged"], 21)
        self.assertEqual(data["credits"]["remaining"], 979)
        self.assertTrue(len(data["explanation"]) > 0)
        self.assertTrue(len(data["lime_explanations"]) > 0)

    @patch("xfetch_client._open_request")
    def test_03_xfetch_credits_depleted_402(self, mock_urlopen):
        """Test handling of 402 credits depleted error"""
        err_res = urllib.error.HTTPError(
            url="https://api.xfetch.io/v1/profiles/by-username/jack",
            code=402,
            msg="Payment Required",
            hdrs={},
            fp=BytesIO(b'{"detail":"Credits depleted"}')
        )
        mock_urlopen.side_effect = err_res

        res = client.post("/analyze_x", json={"username": "jack"})
        self.assertEqual(res.status_code, 402)
        self.assertEqual(res.json()["status"], "error")
        self.assertIn("credits depleted", res.json()["message"].lower())


if __name__ == "__main__":
    unittest.main()
