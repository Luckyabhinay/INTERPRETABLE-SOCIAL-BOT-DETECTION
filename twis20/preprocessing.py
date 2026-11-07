"""Train-only imputation and TF-IDF fitting for feature blocks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .profile_features import PROFILE_COLUMNS
from .text_features import (
    TEXT_SUMMARY_COLUMNS,
    create_tfidf_vectorizer,
)


@dataclass
class FeaturePreprocessor:
    """Impute dense blocks and fit lexical vocabulary using train rows only."""

    profile_columns: tuple[str, ...] = PROFILE_COLUMNS
    summary_columns: tuple[str, ...] = TEXT_SUMMARY_COLUMNS
    vectorizer: TfidfVectorizer = field(default_factory=create_tfidf_vectorizer)
    profile_medians_: dict[str, float] = field(default_factory=dict, init=False)
    summary_medians_: dict[str, float] = field(default_factory=dict, init=False)
    feature_names_: dict[str, tuple[str, ...]] = field(default_factory=dict, init=False)
    is_fitted_: bool = field(default=False, init=False)

    @staticmethod
    def _fit_medians(
        frame: pd.DataFrame,
        columns: Sequence[str],
        train_indices: Sequence[int],
        block_name: str,
    ) -> dict[str, float]:
        if tuple(frame.columns) != tuple(columns):
            raise ValueError(f"{block_name} columns do not match the declared schema")
        train_values = frame.iloc[list(train_indices)].to_numpy(dtype=np.float64)
        if not np.isfinite(train_values[~np.isnan(train_values)]).all():
            raise ValueError(f"{block_name} contains infinite values")
        medians: dict[str, float] = {}
        for index, column in enumerate(columns):
            values = train_values[:, index]
            finite_values = values[~np.isnan(values)]
            medians[column] = (
                float(np.median(finite_values)) if finite_values.size else 0.0
            )
        return medians

    @staticmethod
    def _apply_medians(
        frame: pd.DataFrame,
        columns: Sequence[str],
        medians: dict[str, float],
        block_name: str,
    ) -> pd.DataFrame:
        if tuple(frame.columns) != tuple(columns):
            raise ValueError(f"{block_name} transform columns changed")
        output = frame.copy().astype(np.float64)
        for column in columns:
            output[column] = output[column].fillna(medians[column])
        if not np.isfinite(output.to_numpy()).all():
            raise ValueError(f"{block_name} contains non-finite values after imputation")
        return output

    def fit(
        self,
        profile: pd.DataFrame,
        text_summary: pd.DataFrame,
        documents: Sequence[tuple[str, ...]],
        train_indices: Sequence[int],
    ) -> "FeaturePreprocessor":
        if not train_indices:
            raise ValueError("At least one training row is required")
        if len(set(train_indices)) != len(train_indices):
            raise ValueError("Training indices must be unique")
        if len(profile) != len(text_summary) or len(profile) != len(documents):
            raise ValueError("Feature blocks and text documents must have equal rows")
        if any(index < 0 or index >= len(profile) for index in train_indices):
            raise IndexError("Training index is outside the feature rows")

        self.profile_medians_ = self._fit_medians(
            profile, self.profile_columns, train_indices, "profile"
        )
        self.summary_medians_ = self._fit_medians(
            text_summary, self.summary_columns, train_indices, "text summary"
        )
        self.feature_names_ = {
            "profile": tuple(self.profile_columns),
            "text_summary": tuple(self.summary_columns),
        }
        try:
            self.vectorizer.fit([documents[index] for index in train_indices])
        except ValueError as exc:
            raise ValueError(
                "Unable to fit training-only TF-IDF vocabulary; "
                "training documents contain no eligible tokens"
            ) from exc
        self.feature_names_["text_tfidf"] = tuple(
            str(name) for name in self.vectorizer.get_feature_names_out()
        )
        self.is_fitted_ = True
        return self

    def transform(
        self,
        profile: pd.DataFrame,
        text_summary: pd.DataFrame,
        documents: Sequence[tuple[str, ...]],
    ) -> dict[str, pd.DataFrame | sparse.csr_matrix]:
        if not self.is_fitted_:
            raise RuntimeError("FeaturePreprocessor must be fit before transform")
        if len(profile) != len(text_summary) or len(profile) != len(documents):
            raise ValueError("Feature blocks and text documents must have equal rows")
        return {
            "profile": self._apply_medians(
                profile, self.profile_columns, self.profile_medians_, "profile"
            ),
            "text_summary": self._apply_medians(
                text_summary,
                self.summary_columns,
                self.summary_medians_,
                "text summary",
            ),
            "text_tfidf": self.vectorizer.transform(documents).tocsr(),
        }


__all__ = ["FeaturePreprocessor"]
