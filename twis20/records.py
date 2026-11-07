"""Streaming readers and integrity checks for the official TwiBot-20 splits."""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, TextIO

SPLIT_MEMBERS = {
    "train": "Twibot-20/train.json",
    "dev": "Twibot-20/dev.json",
    "test": "Twibot-20/test.json",
}


@dataclass(frozen=True)
class AccountRecord:
    account_id: str
    split: str
    label: str
    profile: Mapping[str, Any]
    tweets: Any


def stream_json_array(stream: TextIO) -> Iterator[dict[str, Any]]:
    """Yield objects from a JSON array without loading the complete member."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    eof = False
    while True:
        if position > 1024 * 1024:
            buffer = buffer[position:]
            position = 0
        if not eof and len(buffer) - position < 2 * 1024 * 1024:
            chunk = stream.read(4 * 1024 * 1024)
            if chunk:
                buffer += chunk
            else:
                eof = True
        while position < len(buffer):
            while position < len(buffer) and (
                buffer[position].isspace() or buffer[position] in ",["
            ):
                position += 1
            if position >= len(buffer) or buffer[position] == "]":
                break
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                break
            position = end
            if not isinstance(value, dict):
                raise ValueError("Expected each source array item to be an object")
            yield value
        if eof and position >= len(buffer):
            return
        if eof and position < len(buffer) and buffer[position] == "]":
            return


def iter_official_records(archive_path: str) -> Iterator[AccountRecord]:
    """Read train/dev/test from the source archive in official member order."""
    seen_ids: set[str] = set()
    with zipfile.ZipFile(archive_path) as archive:
        members = set(archive.namelist())
        missing_members = set(SPLIT_MEMBERS.values()) - members
        if missing_members:
            raise ValueError(f"Official archive is missing members: {sorted(missing_members)}")
        for split, member in SPLIT_MEMBERS.items():
            with archive.open(member) as binary_stream:
                text_stream = io.TextIOWrapper(binary_stream, encoding="utf-8")
                for raw_record in stream_json_array(text_stream):
                    raw_id = raw_record.get("ID")
                    account_id = str(raw_id).strip() if raw_id is not None else ""
                    if not account_id:
                        raise ValueError(f"Blank account ID in official {split} split")
                    if account_id in seen_ids:
                        raise ValueError(f"Duplicate account ID across official splits: {account_id}")
                    seen_ids.add(account_id)

                    label = str(raw_record.get("label"))
                    if label not in {"0", "1"}:
                        raise ValueError(
                            f"Invalid label for account {account_id}: {label!r}"
                        )
                    raw_profile = raw_record.get("profile")
                    profile = raw_profile if isinstance(raw_profile, Mapping) else {}
                    yield AccountRecord(
                        account_id=account_id,
                        split=split,
                        label=label,
                        profile=profile,
                        tweets=raw_record.get("tweet"),
                    )
