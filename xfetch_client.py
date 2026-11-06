"""
xfetch.io API Client & Feature Extractor
========================================
Integrates with xfetch.io's aggregated workflow endpoint:
GET https://api.xfetch.io/v1/profiles/by-username/{username}

Retrieves profile metrics, recent tweets, and credit usage in a single request.
Maps payload to the exact 34 canonical Cresci features with transparent imputation.
"""

import os
import re
import json
import string
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional

# Sentiment lexicon for linguistic features
AFINN_SAMPLE = {
    "love": 3, "great": 3, "good": 2, "best": 3, "awesome": 4, "happy": 3, "excellent": 3,
    "nice": 2, "wonderful": 4, "glad": 2, "super": 3, "perfect": 3, "thank": 2, "thanks": 2,
    "hate": -3, "bad": -2, "worst": -3, "terrible": -3, "awful": -3, "fake": -2, "scam": -3,
    "horrible": -3, "sad": -2, "angry": -2, "sucks": -3, "poor": -2, "boring": -2, "spam": -3
}

def analyze_sentiment(text: str) -> Tuple[float, float]:
    if not text or not isinstance(text, str):
        return 0.0, 0.0
    words = text.lower().translate(str.maketrans("", "", string.punctuation)).split()
    if not words:
        return 0.0, 0.0
    score = 0
    sentiment_count = 0
    for w in words:
        if w in AFINN_SAMPLE:
            score += AFINN_SAMPLE[w]
            sentiment_count += 1
    polarity = max(-1.0, min(1.0, score / (sentiment_count if sentiment_count > 0 else 1))) if sentiment_count > 0 else 0.0
    subjectivity = min(1.0, sentiment_count / (len(words) * 0.5 + 1.0))
    return polarity, subjectivity

def compute_linguistic_features(text_list: List[str]) -> Dict[str, float]:
    if not text_list:
        return {
            "unique_word_count": 0.0,
            "unique_word_use": 0.0,
            "punctuation_count": 0.0,
            "avg_sentence_length": 0.0,
            "punctuation_density": 0.0
        }
    total_text = " ".join(text_list)
    total_chars = len(total_text)
    punct_count = sum(1 for c in total_text if c in string.punctuation)
    words = total_text.lower().translate(str.maketrans("", "", string.punctuation)).split()
    total_words = len(words)
    unique_words = set(words)
    unique_count = len(unique_words)
    unique_word_use = (unique_count / total_words) if total_words > 0 else 0.0
    avg_sentence_len = (total_chars / len(text_list)) if text_list else 0.0
    punct_density = (punct_count / total_chars) if total_chars > 0 else 0.0
    return {
        "unique_word_count": float(unique_count),
        "unique_word_use": round(unique_word_use, 4),
        "punctuation_count": float(punct_count),
        "avg_sentence_length": round(avg_sentence_len, 2),
        "punctuation_density": round(punct_density, 4)
    }

def sanitize_username(input_str: str) -> str:
    """
    Normalizes username from URL, @handle, or raw text.
    Handles:
      - @username
      - username
      - https://x.com/username
      - https://twitter.com/username
      - http://x.com/username/
    """
    if not input_str:
        return ""
    s = input_str.strip()
    url_match = re.search(r'(?:https?://)?(?:www\.)?(?:twitter\.com|x\.com)/([A-Za-z0-9_]{1,25})', s, re.IGNORECASE)
    if url_match:
        return url_match.group(1)
    s = s.lstrip('@').strip().strip('/')
    clean_match = re.match(r'^[A-Za-z0-9_]{1,25}$', s)
    if clean_match:
        return clean_match.group(0)
    return s


class XFetchClient:
    """
    Client for xfetch.io REST API (Profile + Recent Tweets in 1 call)
    """
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("XFETCH_API_KEY", "").strip()
        if not key:
            # Check local .env file
            env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
            if os.path.exists(env_file):
                try:
                    with open(env_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("XFETCH_API_KEY="):
                                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                                if key:
                                    break
                except Exception:
                    pass
        self.api_key = key
        self.base_url = "https://api.xfetch.io"

    def has_key(self) -> bool:
        return bool(self.api_key)

    def fetch_profile(self, username: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str], int]:
        """
        Fetches profile and recent tweets in one request from xfetch.io.
        Returns: (user_dict, recent_tweets_list, error_message, status_code, meta_dict)
        """
        if not self.has_key():
            return None, None, "XFETCH_API_KEY is not configured. Add your API key to .env or set environment variable.", 401

        clean_user = sanitize_username(username)
        if not clean_user or not re.match(r'^[A-Za-z0-9_]{1,25}$', clean_user):
            return None, None, f"Invalid username format: '{username}'", 400

        url = f"{self.base_url}/v1/profiles/by-username/{clean_user}"
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("User-Agent", "SocialBotDetectionAuditor/2.0")

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                user_data = data.get("data", {}).get("user", {})
                recent_tweets = data.get("data", {}).get("recent_tweets", [])
                meta = data.get("meta", {})

                if not user_data:
                    return None, None, f"Account '{clean_user}' not found on X.", 404

                return {
                    "user": user_data,
                    "recent_tweets": recent_tweets,
                    "meta": meta
                }, None, None, 200

        except urllib.error.HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode("utf-8")
            except Exception:
                pass

            if e.code == 401:
                return None, None, "xfetch Authentication failed: Invalid or expired API Key.", 401
            elif e.code == 402:
                return None, None, "xfetch API credits depleted (HTTP 402). Please top up credits at xfetch.io.", 402
            elif e.code == 404:
                return None, None, f"Account '{clean_user}' not found on X.", 404
            elif e.code == 429:
                return None, None, "xfetch rate limit reached. Please wait a moment before trying again.", 429
            elif e.code == 403:
                return None, None, f"xfetch Access forbidden: {e.reason}", 403
            else:
                detail = f"xfetch HTTP error: {e.code} - {e.reason}"
                if body_text:
                    try:
                        err_json = json.loads(body_text)
                        if "detail" in err_json:
                            detail = f"xfetch error: {err_json['detail']}"
                        elif "message" in err_json:
                            detail = f"xfetch error: {err_json['message']}"
                    except Exception:
                        pass
                return None, None, detail, e.code

        except urllib.error.URLError as e:
            return None, None, f"Network timeout connecting to xfetch API: {e.reason}", 504
        except Exception as e:
            return None, None, f"Unexpected error during xfetch lookup: {str(e)}", 500


def extract_features_from_xfetch_data(
    payload: Dict[str, Any],
    default_medians: Dict[str, float]
) -> Tuple[Dict[str, float], List[str], List[str]]:
    """
    Maps xfetch profile + recent_tweets into the canonical 34 features.
    """
    user_data = payload.get("user", {})
    recent_tweets = payload.get("recent_tweets", []) or []

    # xfetch field names:
    # follower_count or followers_count
    followers_count = float(user_data.get("follower_count", user_data.get("followers_count", 0)))
    friends_count = float(user_data.get("following_count", user_data.get("friends_count", 0)))
    status_count = float(user_data.get("tweet_count", user_data.get("status_count", 0)))
    listed_count = float(user_data.get("listed_count", 0))

    created_at_str = user_data.get("created_at")
    desc = user_data.get("description", "") or ""
    verified = 1.0 if user_data.get("verified", False) else 0.0

    # Calculate account age
    if created_at_str:
        try:
            # Handles ISO strings like "2006-03-21T18:50:00.000Z"
            dt = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            age_days = max(1.0, (datetime.now(dt.tzinfo) - dt).total_seconds() / 86400.0)
        except Exception:
            age_days = float(default_medians.get("account_age_days", 1200.0))
    else:
        age_days = float(default_medians.get("account_age_days", 1200.0))

    tweets_per_day = status_count / age_days
    ffratio = followers_count / (followers_count + friends_count + 1e-5)
    description_binary = 1.0 if len(desc.strip()) > 0 else 0.0

    profile_img = user_data.get("profile_image_url", "")
    default_profile_image = 1.0 if ("default_profile" in profile_img or not profile_img) else 0.0

    comp_fields = [
        bool(user_data.get("location")),
        description_binary == 1.0,
        bool(user_data.get("url")),
        default_profile_image == 0.0,
        bool(user_data.get("pinned_tweet_id"))
    ]
    profile_completeness = sum(1 for c in comp_fields if c) / len(comp_fields)

    extracted: Dict[str, float] = {
        "account_age_days": round(age_days, 1),
        "followers_count": followers_count,
        "friends_count": friends_count,
        "status_count": status_count,
        "listed_count": listed_count,
        "ffratio": round(ffratio, 6),
        "tweets_per_day": round(tweets_per_day, 4),
        "description_binary": description_binary,
        "verified": verified,
        "default_profile_image": default_profile_image,
        "profile_completeness": round(profile_completeness, 4)
    }

    available_features = list(extracted.keys())
    missing_features = []

    # Extract tweet-level metrics if returned
    if recent_tweets:
        tweet_texts = [t.get("text", "") for t in recent_tweets if t.get("text")]
        ling = compute_linguistic_features(tweet_texts)

        pols, subjs = [], []
        for txt in tweet_texts:
            p, s = analyze_sentiment(txt)
            pols.append(p)
            subjs.append(s)

        extracted["unique_word_count"] = ling["unique_word_count"]
        extracted["unique_word_use"] = ling["unique_word_use"]
        extracted["punctuation_count"] = ling["punctuation_count"]
        extracted["avg_sentence_length"] = ling["avg_sentence_length"]
        extracted["punctuation_density"] = ling["punctuation_density"]
        extracted["avg_polarity"] = round(sum(pols) / len(pols), 4) if pols else 0.0
        extracted["avg_subjectivity"] = round(sum(subjs) / len(subjs), 4) if subjs else 0.0

        n_tw = len(recent_tweets)
        # xfetch tweet structure provides: like_count, retweet_count, reply_count
        retweets = sum(t.get("retweet_count", t.get("public_metrics", {}).get("retweet_count", 0)) for t in recent_tweets)
        replies = sum(t.get("reply_count", t.get("public_metrics", {}).get("reply_count", 0)) for t in recent_tweets)
        likes = sum(t.get("like_count", t.get("public_metrics", {}).get("like_count", 0)) for t in recent_tweets)

        # Count hashtags, mentions, urls in entities
        hashtags = sum(len(t.get("entities", {}).get("hashtags", [])) for t in recent_tweets)
        mentions = sum(len(t.get("entities", {}).get("mentions", [])) for t in recent_tweets)
        urls = sum(len(t.get("entities", {}).get("urls", [])) for t in recent_tweets)

        extracted["hashtag_count"] = float(hashtags)
        extracted["mentions_count"] = float(mentions)
        extracted["url_count"] = float(urls)
        extracted["retweet_count"] = float(retweets)
        extracted["reply_count"] = float(replies)

        extracted["avg_hashtag"] = round(hashtags / n_tw, 4)
        extracted["avg_mentions"] = round(mentions / n_tw, 4)
        extracted["avg_url"] = round(urls / n_tw, 4)
        extracted["avg_retweets"] = round(retweets / n_tw, 4)
        extracted["avg_replies"] = round(replies / n_tw, 4)
        extracted["avg_favorites"] = round(likes / n_tw, 4)
        extracted["avg_user_engagement"] = round((retweets + replies + likes) / n_tw, 4)

        for k in ["unique_word_count", "unique_word_use", "punctuation_count", "avg_sentence_length",
                  "punctuation_density", "avg_polarity", "avg_subjectivity", "hashtag_count", "mentions_count",
                  "url_count", "retweet_count", "reply_count", "avg_hashtag", "avg_mentions", "avg_url",
                  "avg_retweets", "avg_replies", "avg_favorites", "avg_user_engagement"]:
            available_features.append(k)

    canonical_34 = [
        "account_age_days", "avg_favorites", "avg_hashtag", "avg_mentions", "avg_polarity",
        "avg_replies", "avg_retweets", "avg_sentence_length", "avg_subjectivity", "avg_url",
        "avg_user_engagement", "default_profile", "default_profile_image", "description_binary",
        "favorites_count", "ffratio", "followers_count", "friends_count", "geo_enabled",
        "hashtag_count", "listed_count", "mentions_count", "profile_background_tile",
        "profile_completeness", "punctuation_count", "punctuation_density", "reply_count",
        "retweet_count", "status_count", "tweets_per_day", "unique_word_count",
        "unique_word_use", "url_count", "verified"
    ]

    for feat in canonical_34:
        if feat not in extracted:
            extracted[feat] = float(default_medians.get(feat, 0.0))
            missing_features.append(feat)

    return extracted, available_features, missing_features

