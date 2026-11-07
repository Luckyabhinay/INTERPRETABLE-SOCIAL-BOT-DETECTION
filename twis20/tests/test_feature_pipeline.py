from __future__ import annotations

import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import joblib
from scipy import sparse

from twis20.feature_schema import (
    NETWORK_COLUMNS,
    PROFILE_COLUMNS,
    TEXT_SUMMARY_COLUMNS,
)
from twis20.network_features import extract_network_features
from twis20.pipeline import (
    build_feature_dataset,
    extract_from_sources,
    save_feature_dataset,
)
from twis20.preprocessing import FeaturePreprocessor
from twis20.profile_features import extract_profile_features
from twis20.records import AccountRecord
from twis20.text_features import extract_text_features, overlap_exposure


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _profile(**overrides: str) -> dict[str, str]:
    profile = {
        "followers_count": "10",
        "friends_count": "20",
        "statuses_count": "30",
        "favourites_count": "40",
        "listed_count": "2",
        "verified": "False ",
        "geo_enabled": "True ",
        "protected": "False ",
        "default_profile": "False ",
        "default_profile_image": "False ",
        "contributors_enabled": "False ",
        "has_extended_profile": "True ",
        "is_translation_enabled": "False ",
        "is_translator": "False ",
        "profile_background_tile": "False ",
        "profile_use_background_image": "True ",
        "created_at": "Tue Feb 04 12:13:10 +0000 2020 ",
        "description": "Hello @person #news https://t.co/x",
    }
    profile.update(overrides)
    return profile


class ProfileFeatureTests(unittest.TestCase):
    def test_order_missing_values_and_boolean_unknowns(self) -> None:
        frame = extract_profile_features(
            [
                _profile(followers_count="", verified="maybe"),
                _profile(description="None", created_at="bad date"),
            ]
        )
        self.assertEqual(tuple(frame.columns), PROFILE_COLUMNS)
        self.assertTrue(np.isnan(frame.loc[0, "profile.followers_count"]))
        self.assertEqual(frame.loc[0, "profile.followers_count__missing"], 1.0)
        self.assertEqual(frame.loc[0, "profile.verified__unknown"], 1.0)
        self.assertEqual(frame.loc[0, "profile.verified__true"], 0.0)
        self.assertEqual(frame.loc[0, "profile.created_at_year"], 2020.0)
        self.assertEqual(frame.loc[0, "profile.description_hashtag_count"], 1.0)
        self.assertEqual(frame.loc[1, "profile.description_present"], 0.0)
        self.assertTrue(np.isnan(frame.loc[1, "profile.description_char_count"]))
        self.assertEqual(frame.loc[1, "profile.description_char_count__missing"], 1.0)


class TextFeatureTests(unittest.TestCase):
    def test_repeated_tweets_are_counted_and_retained(self) -> None:
        summaries, documents = extract_text_features(
            [["repeat words", "repeat words", "", 4], None]
        )
        self.assertEqual(tuple(summaries.columns), TEXT_SUMMARY_COLUMNS)
        self.assertEqual(documents[0], ("repeat words", "repeat words", ""))
        self.assertEqual(summaries.loc[0, "tweet.valid_string_count"], 3.0)
        self.assertEqual(summaries.loc[0, "tweet.unique_exact_text_count"], 2.0)
        self.assertEqual(summaries.loc[0, "tweet.exact_duplicate_fraction"], 1 / 3)
        self.assertEqual(summaries.loc[0, "tweet.empty_or_invalid_item_count"], 2.0)
        self.assertEqual(summaries.loc[1, "tweet.list_available"], 0.0)
        self.assertEqual(documents[1], ())

    def test_overlap_flags_preserve_all_rows(self) -> None:
        documents = (
            ("Shared\tPost", "duplicate"),
            ("shared post",),
            ("different text",),
        )
        flags = overlap_exposure(documents, ("train", "dev", "test"))
        self.assertEqual(len(flags), 3)
        self.assertTrue(flags.loc[1, "text_overlap_normalized_train"])
        self.assertFalse(flags.loc[1, "text_overlap_exact_train"])
        self.assertFalse(flags.loc[2, "text_overlap_exact_train"])
        self.assertFalse(flags.loc[2, "text_overlap_normalized_train"])
        self.assertTrue(flags.loc[0].isna().all())

    def test_bigram_analysis_does_not_cross_tweet_boundaries(self) -> None:
        from twis20.text_features import analyze_tweet_document

        terms = set(analyze_tweet_document(("alpha", "beta")))
        self.assertEqual(terms, {"alpha", "beta"})


class PreprocessingTests(unittest.TestCase):
    def test_imputation_and_vocabulary_are_fit_on_train_only(self) -> None:
        profiles = extract_profile_features(
            [
                _profile(followers_count="2"),
                _profile(followers_count="4"),
                _profile(followers_count="999999"),
            ]
        )
        summaries, docs = extract_text_features(
            [
                ["shared common"],
                ["shared common"],
                ["heldoutword unseenword"],
            ]
        )
        preprocessor = FeaturePreprocessor().fit(
            profiles, summaries, docs, train_indices=(0, 1)
        )
        transformed = preprocessor.transform(profiles, summaries, docs)
        self.assertEqual(
            preprocessor.profile_medians_["profile.followers_count"], 3.0
        )
        self.assertEqual(
            transformed["profile"].loc[2, "profile.followers_count"], 999999.0
        )
        self.assertNotIn("heldoutword", preprocessor.feature_names_["text_tfidf"])
        self.assertNotIn("unseenword", preprocessor.feature_names_["text_tfidf"])
        self.assertEqual(transformed["text_tfidf"].shape[0], 3)

    def test_all_missing_train_column_has_stable_placeholder(self) -> None:
        profiles = extract_profile_features(
            [
                _profile(followers_count="None"),
                _profile(followers_count="invalid"),
                _profile(followers_count="500"),
            ]
        )
        summaries, docs = extract_text_features(
            [["shared token"], ["shared token"], ["heldout token"]]
        )
        preprocessor = FeaturePreprocessor().fit(
            profiles, summaries, docs, train_indices=(0, 1)
        )
        transformed = preprocessor.transform(profiles, summaries, docs)
        self.assertEqual(
            preprocessor.profile_medians_["profile.followers_count"], 0.0
        )
        self.assertEqual(
            transformed["profile"]["profile.followers_count"].tolist(),
            [0.0, 0.0, 500.0],
        )
        self.assertEqual(
            transformed["profile"]["profile.followers_count__missing"].tolist(),
            [1.0, 1.0, 0.0],
        )


class NetworkFeatureTests(unittest.TestCase):
    def test_distinct_topology_uses_support_and_ignores_post(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_csv = root / "split.csv"
            edge_csv = root / "edge.csv"
            _write_csv(
                split_csv,
                ["id", "split"],
                [
                    {"id": "u1", "split": "train"},
                    {"id": "u2", "split": "val"},
                    {"id": "u3", "split": "test"},
                    {"id": "u4", "split": "support"},
                ],
            )
            _write_csv(
                edge_csv,
                ["source_id", "relation", "target_id"],
                [
                    {"source_id": "u1", "relation": "friend", "target_id": "u4"},
                    {"source_id": "u1", "relation": "friend", "target_id": "u4"},
                    {"source_id": "u4", "relation": "friend", "target_id": "u2"},
                    {"source_id": "u1", "relation": "follow", "target_id": "u2"},
                    {"source_id": "u1", "relation": "post", "target_id": "u3"},
                ],
            )
            frame = extract_network_features(
                ["1", "2", "3"],
                ["train", "dev", "test"],
                split_csv,
                edge_csv,
            )
            self.assertEqual(tuple(frame.columns), NETWORK_COLUMNS)
            self.assertEqual(frame.loc[0, "network.friend_out_degree"], 1.0)
            self.assertEqual(frame.loc[0, "network.follow_out_degree"], 1.0)
            self.assertEqual(frame.loc[0, "network.combined_out_degree"], 2.0)
            self.assertEqual(frame.loc[1, "network.friend_in_degree"], 1.0)
            self.assertEqual(frame.loc[2, "network.isolate"], 1.0)
            self.assertTrue(frame.attrs["transductive"])
            self.assertFalse(frame.attrs["labels_used"])

    def test_roster_split_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_csv(
                root / "split.csv",
                ["id", "split"],
                [{"id": "u1", "split": "test"}],
            )
            _write_csv(
                root / "edge.csv",
                ["source_id", "relation", "target_id"],
                [],
            )
            with self.assertRaisesRegex(ValueError, "split mismatch"):
                extract_network_features(
                    ["1"], ["train"], root / "split.csv", root / "edge.csv"
                )


class PipelineIntegrationTests(unittest.TestCase):
    def test_official_split_integrity_and_leakage_separation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "official.zip"
            split_csv = root / "split.csv"
            edge_csv = root / "edge.csv"
            records = {
                "train": [
                    {
                        "ID": "1",
                        "label": "0",
                        "profile": _profile(),
                        "tweet": ["repeat token", "label text", "repeat token"],
                    },
                    {
                        "ID": "4",
                        "label": "1",
                        "profile": _profile(followers_count="None"),
                        "tweet": ["repeat token", "label text"],
                    },
                ],
                "dev": [
                    {
                        "ID": "2",
                        "label": "1",
                        "profile": _profile(),
                        "tweet": ["repeat token"],
                    }
                ],
                "test": [
                    {
                        "ID": "3",
                        "label": "0",
                        "profile": _profile(),
                        "tweet": ["heldouttoken only"],
                    }
                ],
            }
            with zipfile.ZipFile(archive_path, "w") as archive:
                for split, rows in records.items():
                    member = f"Twibot-20/{split}.json"
                    archive.writestr(member, json.dumps(rows))

            _write_csv(
                split_csv,
                ["id", "split"],
                [
                    {"id": "u1", "split": "train"},
                    {"id": "u4", "split": "train"},
                    {"id": "u2", "split": "val"},
                    {"id": "u3", "split": "test"},
                    {"id": "u5", "split": "support"},
                ],
            )
            _write_csv(
                edge_csv,
                ["source_id", "relation", "target_id"],
                [{"source_id": "u1", "relation": "friend", "target_id": "u5"}],
            )

            dataset = extract_from_sources(archive_path, split_csv, edge_csv)
            self.assertEqual(dataset.account_ids, ("1", "4", "2", "3"))
            self.assertEqual(dataset.splits, ("train", "train", "dev", "test"))
            self.assertEqual(dataset.labels, ("0", "1", "1", "0"))
            self.assertEqual(dataset.features["profile"].shape[0], 4)
            self.assertEqual(dataset.features["network"].shape[0], 4)
            self.assertTrue(sparse.isspmatrix_csr(dataset.features["text_tfidf"]))
            self.assertNotIn("heldouttoken", dataset.feature_names["text_tfidf"])
            self.assertEqual(dataset.split_indices("test").tolist(), [3])

            all_names = [
                name
                for names in dataset.feature_names.values()
                for name in names
            ]
            combined_names = dataset.experiment_feature_names(
                "profile_text_network"
            )
            self.assertEqual(combined_names, tuple(all_names))
            self.assertEqual(combined_names[0], PROFILE_COLUMNS[0])
            self.assertEqual(combined_names[-len(NETWORK_COLUMNS):], NETWORK_COLUMNS)
            self.assertNotIn("account_id", all_names)
            self.assertNotIn("label", all_names)
            self.assertNotIn("split", all_names)
            self.assertIn("tweet.tfidf.label", all_names)
            self.assertTrue(dataset.overlap_exposure.loc[2, "text_overlap_exact_train"])
            self.assertEqual(
                dataset.experiment_matrix("profile_text_network").shape[0], 4
            )
            metadata = dataset.metadata_frame()
            self.assertEqual(metadata["account_id"].tolist(), list(dataset.account_ids))
            self.assertEqual(metadata["label"].tolist(), list(dataset.labels))
            self.assertEqual(dataset.manifest["graph_labels_read"], False)
            self.assertTrue(dataset.manifest["graph_transductive"])
            output = root / "prepared_features"
            save_feature_dataset(dataset, output)
            self.assertTrue((output / "feature_manifest.json").is_file())
            self.assertTrue((output / "metadata.json").is_file())
            self.assertTrue((output / "preprocessor.joblib").is_file())
            saved_preprocessor = joblib.load(output / "preprocessor.joblib")
            self.assertEqual(
                saved_preprocessor.feature_names_["text_tfidf"],
                dataset.preprocessor.feature_names_["text_tfidf"],
            )
            self.assertEqual(
                sparse.load_npz(output / "features_text_tfidf.npz").shape,
                dataset.features["text_tfidf"].shape,
            )
            saved_metadata = json.loads(
                (output / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [row["account_id"] for row in saved_metadata],
                list(dataset.account_ids),
            )
            with self.assertRaises(FileExistsError):
                save_feature_dataset(dataset, output)

    def test_duplicate_account_ids_are_rejected(self) -> None:
        records = [
            AccountRecord("1", "train", "0", _profile(), ["same text"]),
            AccountRecord("1", "dev", "1", _profile(), ["same text"]),
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate account ID"):
            build_feature_dataset(records, "unused-split.csv", "unused-edge.csv")


if __name__ == "__main__":
    unittest.main()
