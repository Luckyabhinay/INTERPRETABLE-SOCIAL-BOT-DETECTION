from collections.abc import Callable


def build_text_provenance_features(
    tweet_texts: list[str],
    description_texts: list[str],
    compute_linguistic_features: Callable,
    analyze_sentiment: Callable,
) -> dict:
    if tweet_texts:
        text_source = "tweet_text"
    elif description_texts:
        text_source = "description"
    else:
        text_source = "unavailable"

    features = {
        "text_source": text_source,
        "has_tweet_text": int(bool(tweet_texts)),
        "has_description_text": int(bool(description_texts)),
    }

    for source, texts in (("tweet", tweet_texts), ("description", description_texts)):
        prefix = f"{source}_text"
        if not texts:
            features.update({
                f"{prefix}_unique_word_count": None,
                f"{prefix}_unique_word_use": None,
                f"{prefix}_punctuation_count": None,
                f"{prefix}_avg_sentence_length": None,
                f"{prefix}_punctuation_density": None,
                f"{prefix}_avg_polarity": None,
                f"{prefix}_avg_subjectivity": None,
            })
            continue

        linguistic = compute_linguistic_features(texts)
        sentiments = [analyze_sentiment(text) for text in texts]
        features.update({
            f"{prefix}_unique_word_count": linguistic["unique_word_count"],
            f"{prefix}_unique_word_use": linguistic["unique_word_use"],
            f"{prefix}_punctuation_count": linguistic["punctuation_count"],
            f"{prefix}_avg_sentence_length": linguistic["avg_sentence_length"],
            f"{prefix}_punctuation_density": linguistic["punctuation_density"],
            f"{prefix}_avg_polarity": sum(polarity for polarity, _ in sentiments) / len(sentiments),
            f"{prefix}_avg_subjectivity": sum(subjectivity for _, subjectivity in sentiments) / len(sentiments),
        })

    return features
