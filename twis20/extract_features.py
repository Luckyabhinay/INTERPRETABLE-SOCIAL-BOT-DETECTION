"""Command-line entry point for TwiBot-20 Phase 3 feature preparation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import extract_from_sources, save_feature_dataset

_ROOT = Path(__file__).resolve().parent
_DEFAULT_ARCHIVE = (
    _ROOT
    / "TwiBot-20-20261106T091806Z-1-002"
    / "TwiBot-20"
    / "Twibot-20.zip"
)
_DEFAULT_GRAPH = _ROOT / "TwiBot-20-Format22-003" / "Twibot-20"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract TwiBot-20 feature blocks and fit train-only preprocessing. "
            "This command does not train a model."
        )
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=_DEFAULT_ARCHIVE,
        help="Original TwiBot-20 archive containing official splits",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=_DEFAULT_GRAPH / "split.csv",
        help="Converted graph roster with labeled and support IDs",
    )
    parser.add_argument(
        "--edge-csv",
        type=Path,
        default=_DEFAULT_GRAPH / "edge.csv",
        help="Converted graph relationships",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory; an existing directory is never overwritten",
    )
    arguments = parser.parse_args()

    dataset = extract_from_sources(
        arguments.archive,
        arguments.split_csv,
        arguments.edge_csv,
    )
    output = save_feature_dataset(dataset, arguments.output)
    print(f"Saved feature blocks to {output}")
    print(f"Rows: {len(dataset.account_ids)}")
    print(f"Split counts: {dataset.manifest['split_counts']}")
    print("No model was trained.")


if __name__ == "__main__":
    main()
