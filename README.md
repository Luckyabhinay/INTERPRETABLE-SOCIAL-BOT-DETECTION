# Interpretable Social Bot Detection Using Machine Learning & Live X (Twitter) Analysis

An explainable machine learning architecture for detecting automated spambots and fake followers on social platforms. Features dual-tier Explainable AI (SHAP TreeExplainer & LIME Tabular Explainer), canonical 34-feature extraction, a FastAPI interactive dashboard, and live X (Twitter) API v2 account analysis.

---

## 🚀 Key Highlights & Capabilities

- **Champion ML Classifier:** Tuned LightGBM and XGBoost trained on rigorous 80/20 train/held-out test split of the Cresci benchmark datasets (zero data leakage).
- **Dual-Tier Explainability:**
  - **Global & Local SHAP (TreeExplainer):** Fast exact Tree SHAP attributions quantifying exact feature impact on prediction log-odds.
  - **Local LIME (LimeTabularExplainer):** Linear surrogate rules providing interpretable boundaries (e.g. `avg_replies <= 0.00`, `ffratio > 0.25`).
- **Live X (Twitter) API v2 Integration:**
  - Secure profile lookup via official API (`https://api.twitter.com/2/users/by/username/:username`).
  - Accepts `@username`, `username`, or full profile URLs (`https://x.com/user`, `https://twitter.com/user`).
  - Maps live metadata into the 34 canonical features.
  - Handles missing attributes transparently using training median defaults (never hallucinates data).
  - Robust error handling for HTTP 400, 401 (invalid/missing token), 404 (user not found), 429 (rate limits), and 504 (timeouts).
- **Scientific Transparency:** Predictions output as "Likely Bot" or "Likely Genuine" with explicit notation of contemporary platform domain shift.

---

## 🛠️ Setup & Environment Configuration

### 1. Requirements & Dependencies
Ensure Python 3.10+ is installed. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Official X API Bearer Token
Obtain a Bearer Token from the [X Developer Portal](https://developer.x.com/) and set the environment variable:

**Windows (PowerShell):**
```powershell
$env:X_BEARER_TOKEN="your_x_bearer_token_here"
```

**Linux / macOS:**
```bash
export X_BEARER_TOKEN="your_x_bearer_token_here"
```

*(Note: Never hardcode or commit tokens to source code or git repositories).*

### Dataset files
Raw benchmark datasets and generated CSV analysis outputs are intentionally excluded from Git. Training and audit scripts may require you to obtain the Cresci datasets separately and place them locally. Pretrained model artifacts and compact aggregate feature-distribution metadata required by the application are included.

---

## 🖥️ Running the Application

Start the FastAPI application and web dashboard:
```bash
python app.py
```
Open your browser to: **`http://127.0.0.1:8000`**

### Web Dashboard Features:
1. **Live X (Twitter) Account Analyzer:**
   - Enter an X handle (e.g., `@NASA`, `@OpenAI`) or profile URL.
   - Click **"Analyze X Account"** to inspect live profile info, avatar, follower counts, prediction confidence, SHAP breakdown, and feature audit pills.
2. **Manual Feature Simulator:**
   - Adjust individual numeric sliders or select preset configurations ("Preset: Human", "Preset: Bot") to explore model boundaries and LIME explanations.

---

## 📡 API Endpoints

### 1. `POST /analyze_x` (Live X Account Analysis)
**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/analyze_x" \
     -H "Content-Type: application/json" \
     -d '{"username": "@NASA"}'
```

**Response Format:**
```json
{
  "platform": "X",
  "username": "NASA",
  "prediction": "Likely Genuine",
  "confidence": 0.9998,
  "probabilities": {
    "Human": 0.9998,
    "Bot": 0.0002
  },
  "profile_summary": {
    "id": "11348282",
    "name": "NASA",
    "username": "NASA",
    "description": "There's space for everybody.",
    "account_age_days": 6050.2,
    "followers_count": 82000000,
    "friends_count": 180,
    "status_count": 73000,
    "listed_count": 105000,
    "verified": true,
    "profile_image_url": "https://pbs.twimg.com/..."
  },
  "available_features": [
    "account_age_days", "followers_count", "friends_count", "status_count",
    "listed_count", "ffratio", "tweets_per_day", "description_binary",
    "verified", "default_profile_image", "profile_completeness"
  ],
  "missing_features": [
    "avg_favorites", "avg_hashtag", "avg_mentions", "avg_polarity", "avg_replies",
    "avg_retweets", "avg_sentence_length", "avg_subjectivity", "avg_url",
    "avg_user_engagement", "default_profile", "favorites_count", "geo_enabled",
    "hashtag_count", "mentions_count", "profile_background_tile", "punctuation_count",
    "punctuation_density", "reply_count", "retweet_count", "unique_word_count",
    "unique_word_use", "url_count"
  ],
  "explanation": [
    {
      "feature": "ffratio",
      "value": 0.9999,
      "attribution": -0.6214,
      "direction": "Towards GENUINE"
    },
    {
      "feature": "account_age_days",
      "value": 6050.2,
      "attribution": -0.4821,
      "direction": "Towards GENUINE"
    }
  ],
  "lime_explanations": [
    {
      "feature_rule": "ffratio > 0.25",
      "weight": -0.2541,
      "direction": "Towards GENUINE"
    }
  ],
  "status": "success",
  "scientific_disclaimer": "Predictions are probabilistic estimates derived from the Cresci benchmark training dataset. Contemporary X platform dynamics and bot behaviors may exhibit domain shift."
}
```

### 2. `POST /predict_explain` (Manual Feature Prediction)
**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/predict_explain" \
     -H "Content-Type: application/json" \
     -d '{
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
     }'
```

---

## 🧪 Comprehensive Verification & Test Suites

Run the live X analysis automated unit test suite (11 test cases covering mocked API responses, timeouts, rate limits, 401/404 handling, SHAP/LIME output verification, and regression tests):
```bash
python test_x_analysis.py
```

Run the end-to-end project readiness verification:
```bash
python test_complete_readiness.py
```
