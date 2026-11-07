import os
import sys
import csv
import string
import math
from datetime import datetime
from collections import defaultdict
from text_provenance import build_text_provenance_features

# Set stdout encoding for Windows console compatibility
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRESCI15_DIR = os.path.join(BASE_DIR, "15")
OUTPUT_FILE = os.path.join(BASE_DIR, "cresci15_provenance_features.csv")

SUBDATASETS = [
    ("E13.csv", 0),  # Genuine
    ("FSF.csv", 1),  # Bot
    ("INT.csv", 1),  # Bot
    ("TFP.csv", 1),  # Bot
    ("TWT.csv", 1),  # Bot
]

def safe_int(val, default=0):
    try:
        if val is None or val == "" or str(val).upper() == "NULL":
            return default
        return int(float(val))
    except (ValueError, TypeError):
        return default

def safe_float(val, default=0.0):
    try:
        if val is None or val == "" or str(val).upper() == "NULL":
            return default
        return float(val)
    except (ValueError, TypeError):
        return default

def safe_bool(val):
    if val is None or val == "" or str(val).upper() == "NULL":
        return 0
    s = str(val).strip().lower()
    return 1 if s in ("1", "true", "yes") else 0

def parse_date(date_str):
    if not date_str or str(date_str).upper() == "NULL":
        return None
    for fmt in ("%a %b %d %H:%M:%S %z %Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            pass
    return None

# Fast sentiment dictionary based on VADER/AFINN sentiment words for fast, self-contained execution
AFINN_SAMPLE = {
    "love": 3, "great": 3, "good": 2, "best": 3, "awesome": 4, "happy": 3, "excellent": 3,
    "nice": 2, "wonderful": 4, "glad": 2, "super": 3, "perfect": 3, "thank": 2, "thanks": 2,
    "hate": -3, "bad": -2, "worst": -3, "terrible": -3, "awful": -3, "fake": -2, "scam": -3,
    "horrible": -3, "sad": -2, "angry": -2, "sucks": -3, "poor": -2, "boring": -2, "spam": -3
}

def analyze_sentiment(text):
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

def compute_linguistic_features(text_list):
    if not text_list:
        return {
            "unique_word_count": 0,
            "unique_word_use": 0.0,
            "punctuation_count": 0,
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
        "unique_word_count": unique_count,
        "unique_word_use": round(unique_word_use, 4),
        "punctuation_count": punct_count,
        "avg_sentence_length": round(avg_sentence_len, 2),
        "punctuation_density": round(punct_density, 4)
    }

def main():
    print("=" * 70)
    print("PHASE 1: EXTRACTING CANONICAL 31-FEATURE DATASET FOR CRESCI-15")
    print("=" * 70)
    
    # Dataset reference timestamp (Snapshot of Cresci-15 collection ~ Feb 2015)
    # Using snapshot date avoids temporal data leakage from system runtime clock
    SNAPSHOT_DATE = datetime(2015, 5, 1)

    all_users = []

    for folder_name, label in SUBDATASETS:
        folder_path = os.path.join(CRESCI15_DIR, folder_name)
        users_file = os.path.join(folder_path, "users.csv")
        tweets_file = os.path.join(folder_path, "tweets.csv")
        dataset_code = folder_name.replace(".csv", "")

        print(f"\n[+] Processing {dataset_code} (Target Label: {label})...")
        
        users = {}
        with open(users_file, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row.get("id")
                if not uid:
                    continue
                users[uid] = row

        print(f"    Loaded {len(users)} user profiles.")

        tweet_stats = defaultdict(lambda: {
            "tweet_count": 0,
            "retweet_count": 0,
            "reply_count": 0,
            "favorite_count": 0,
            "hashtag_count": 0,
            "url_count": 0,
            "mentions_count": 0,
            "sample_texts": []
        })

        if os.path.exists(tweets_file):
            print(f"    Streaming and aggregating tweets from {tweets_file}...")
            with open(tweets_file, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for count, row in enumerate(reader):
                    uid = row.get("user_id")
                    if not uid or uid not in users:
                        continue
                    stats = tweet_stats[uid]
                    stats["tweet_count"] += 1
                    stats["retweet_count"] += safe_int(row.get("retweet_count"))
                    stats["reply_count"] += safe_int(row.get("reply_count"))
                    stats["favorite_count"] += safe_int(row.get("favorite_count"))
                    stats["hashtag_count"] += safe_int(row.get("num_hashtags"))
                    stats["url_count"] += safe_int(row.get("num_urls"))
                    stats["mentions_count"] += safe_int(row.get("num_mentions"))
                    
                    # Keep sample of texts for linguistic and sentiment extraction
                    if len(stats["sample_texts"]) < 50:
                        txt = row.get("text", "")
                        if txt:
                            stats["sample_texts"].append(txt)
                            
                    if (count + 1) % 500000 == 0:
                        print(f"    ... processed {count + 1:,} tweets")

        for uid, urow in users.items():
            tstats = tweet_stats[uid]
            tw_count = tstats["tweet_count"]

            # User Profile Features (5)
            verified = safe_bool(urow.get("verified"))
            friends_count = safe_int(urow.get("friends_count"))
            followers_count = safe_int(urow.get("followers_count"))
            listed_count = safe_int(urow.get("listed_count"))
            favorites_count = safe_int(urow.get("favourites_count"))

            # Content Features (6)
            hashtag_count = tstats["hashtag_count"]
            mentions_count = tstats["mentions_count"]
            retweet_count = tstats["retweet_count"]
            reply_count = tstats["reply_count"]
            url_count = tstats["url_count"]
            status_count = safe_int(urow.get("statuses_count"))

            # Engagement Metrics (7)
            ffratio = followers_count / (followers_count + friends_count + 1e-5)
            n_tw = max(1, tw_count)
            avg_hashtag = hashtag_count / n_tw if tw_count > 0 else 0.0
            avg_retweets = retweet_count / n_tw if tw_count > 0 else 0.0
            avg_replies = reply_count / n_tw if tw_count > 0 else 0.0
            avg_mentions = mentions_count / n_tw if tw_count > 0 else 0.0
            avg_url = url_count / n_tw if tw_count > 0 else 0.0
            avg_favorites = tstats["favorite_count"] / n_tw if tw_count > 0 else 0.0
            avg_user_engagement = (retweet_count + reply_count + tstats["favorite_count"]) / (tw_count + 1e-5) if tw_count > 0 else 0.0

            # Profile Attributes (6)
            desc = urow.get("description", "") or ""
            desc_binary = 1 if len(desc.strip()) > 0 and desc.strip().upper() != "NULL" else 0
            default_profile = safe_bool(urow.get("default_profile"))
            default_profile_image = safe_bool(urow.get("default_profile_image"))
            geo_enabled = safe_bool(urow.get("geo_enabled"))
            profile_background_tile = safe_bool(urow.get("profile_background_tile"))
            
            # Profile completeness score (fraction of metadata attributes present)
            comp_fields = [
                urow.get("location"),
                desc,
                urow.get("url"),
                not default_profile_image,
                urow.get("profile_banner_url"),
                not default_profile
            ]
            profile_completeness = sum(1 for f in comp_fields if f and str(f).strip().upper() not in ("", "NULL", "0", "FALSE")) / len(comp_fields)

            tweet_texts = tstats["sample_texts"]
            description_texts = [desc] if desc_binary else []
            text_features = build_text_provenance_features(
                tweet_texts,
                description_texts,
                compute_linguistic_features,
                analyze_sentiment,
            )

            # Additional temporal features (useful auxiliary metrics)
            c_date = parse_date(urow.get("created_at"))
            if c_date:
                age_days = max(1.0, (SNAPSHOT_DATE - c_date.replace(tzinfo=None)).total_seconds() / 86400.0)
            else:
                age_days = 365.0
            tweets_per_day = status_count / age_days

            row_data = {
                "user_id": uid,
                "dataset": dataset_code,
                "has_tweet_records": int(tw_count > 0),
                "tweet_record_count": tw_count,
                # 1. User Profile (5)
                "verified": verified,
                "friends_count": friends_count,
                "followers_count": followers_count,
                "listed_count": listed_count,
                "favorites_count": favorites_count,
                # 2. Content Features (6)
                "hashtag_count": hashtag_count,
                "mentions_count": mentions_count,
                "retweet_count": retweet_count,
                "reply_count": reply_count,
                "url_count": url_count,
                "status_count": status_count,
                # 3. Engagement Metrics (7)
                "ffratio": round(ffratio, 6),
                "avg_hashtag": round(avg_hashtag, 4),
                "avg_retweets": round(avg_retweets, 4),
                "avg_replies": round(avg_replies, 4),
                "avg_mentions": round(avg_mentions, 4),
                "avg_url": round(avg_url, 4),
                "avg_favorites": round(avg_favorites, 4),
                "avg_user_engagement": round(avg_user_engagement, 4),
                # 5. Profile Attributes (6)
                "profile_completeness": round(profile_completeness, 4),
                "description_binary": desc_binary,
                "default_profile": default_profile,
                "default_profile_image": default_profile_image,
                "geo_enabled": geo_enabled,
                "profile_background_tile": profile_background_tile,
                # Auxiliary
                "account_age_days": round(age_days, 1),
                "tweets_per_day": round(tweets_per_day, 4),
                # Target Label (0 = Genuine, 1 = Bot)
                "label": label
            }
            row_data.update(text_features)
            all_users.append(row_data)

    print(f"\n[+] Total user feature vectors generated: {len(all_users)}")

    fieldnames = list(all_users[0].keys())
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_users)

    print(f"[✓] Provenance-aware feature extraction completed successfully: {OUTPUT_FILE}")
    print(f"    Total Columns: {len(fieldnames)}")
    genuine_c = sum(1 for u in all_users if u["label"] == 0)
    bot_c = sum(1 for u in all_users if u["label"] == 1)
    print(f"    Genuine Accounts (label=0): {genuine_c}")
    print(f"    Bot Accounts     (label=1): {bot_c}")

if __name__ == "__main__":
    main()
