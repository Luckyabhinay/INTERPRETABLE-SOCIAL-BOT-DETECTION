"""Reproducible, user-supplied live-account validation without training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


REGISTRY_FIELDS = (
    "record_id",
    "username",
    "reference_class",
    "reference_source",
    "evidence_reference",
    "reviewer",
    "reviewed_at",
)
REFERENCE_CLASSES = {
    "genuine_verified",
    "bot_verified",
    "suspicious_unverified",
}
PREDICTIONS = {"Likely Bot", "Likely Genuine"}
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_]{1,15}$")


def _normalized_username(value: str) -> str:
    candidate = value.strip()
    parsed = urlparse(candidate if "://" in candidate else "")
    if parsed.scheme:
        if (parsed.hostname or "").lower() not in {
            "x.com",
            "www.x.com",
            "twitter.com",
            "www.twitter.com",
        }:
            raise ValueError(f"unsupported profile URL: {value}")
        candidate = parsed.path.strip("/").split("/", 1)[0]
    candidate = candidate.removeprefix("@")
    if not USERNAME_PATTERN.fullmatch(candidate):
        raise ValueError(f"invalid X username or profile URL: {value}")
    return candidate.casefold()


def load_registry(path: Path) -> list[dict[str, str]]:
    """Load and validate the user's independent reference registry."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(REGISTRY_FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"registry must include columns: {', '.join(REGISTRY_FIELDS)}")
        rows = list(reader)

    if not 30 <= len(rows) <= 50:
        raise ValueError(f"registry must contain 30–50 accounts; found {len(rows)}")

    seen_ids: set[str] = set()
    seen_usernames: set[str] = set()
    cleaned = []
    for line_number, row in enumerate(rows, start=2):
        record = {field: (row.get(field) or "").strip() for field in REGISTRY_FIELDS}
        if any(not record[field] for field in REGISTRY_FIELDS):
            raise ValueError(f"registry line {line_number} has an empty required field")
        if record["record_id"] in seen_ids:
            raise ValueError(f"duplicate record_id on registry line {line_number}")
        normalized = _normalized_username(record["username"])
        if normalized in seen_usernames:
            raise ValueError(f"duplicate X account on registry line {line_number}")
        if record["reference_class"] not in REFERENCE_CLASSES:
            raise ValueError(
                f"invalid reference_class on registry line {line_number}: "
                f"{record['reference_class']}"
            )
        try:
            datetime.fromisoformat(record["reviewed_at"].replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"reviewed_at must be an ISO-8601 datetime on registry line {line_number}"
            ) from exc
        seen_ids.add(record["record_id"])
        seen_usernames.add(normalized)
        cleaned.append(record)
    return cleaned


def _safe_api_error(status_code: int | None, body: bytes) -> dict[str, Any]:
    try:
        response = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        response = {}
    return {
        "request_status": "http_error",
        "http_status": status_code,
        "error_code": response.get("error_code") if isinstance(response, dict) else None,
    }


def analyze_account(
    base_url: str,
    username: str,
    timeout: float,
) -> dict[str, Any]:
    """Call the local TwiBot-20 endpoint and retain only score/audit metadata."""
    parsed = urlparse(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or (parsed.hostname or "").lower() not in {"localhost", "127.0.0.1", "::1"}
    ):
        raise ValueError("base URL must be a loopback HTTP(S) origin without credentials or query data")
    endpoint = base_url.rstrip("/") + "/twibot20/analyze"
    payload = json.dumps({"username": username}).encode("utf-8")
    request = Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
            status_code = response.status
    except HTTPError as exc:
        with exc:
            return _safe_api_error(exc.code, exc.read())
    except (TimeoutError, URLError, OSError) as exc:
        reason = exc.reason if isinstance(exc, URLError) else exc
        error_code = "timeout" if isinstance(reason, TimeoutError) else "network_error"
        return {"request_status": "network_error", "error_code": error_code}

    try:
        response_data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"request_status": "invalid_response", "http_status": status_code}
    if not isinstance(response_data, dict):
        return {"request_status": "invalid_response", "http_status": status_code}
    if status_code != 200 or response_data.get("status") != "success":
        return {
            "request_status": "service_error",
            "http_status": status_code,
            "error_code": response_data.get("error_code"),
        }

    prediction = response_data.get("prediction")
    bot_score = response_data.get("model_score_uncalibrated")
    class_scores = response_data.get("class_scores_uncalibrated")
    threshold = response_data.get("threshold")
    audit = response_data.get("feature_audit")
    audit = audit if isinstance(audit, dict) else {}
    genuine_score = class_scores.get("Genuine") if isinstance(class_scores, dict) else None
    response_bot_score = class_scores.get("Bot") if isinstance(class_scores, dict) else None
    if (
        prediction not in PREDICTIONS
        or isinstance(bot_score, bool)
        or not isinstance(bot_score, (int, float))
        or not math.isfinite(bot_score)
        or not 0 <= bot_score <= 1
        or isinstance(genuine_score, bool)
        or isinstance(response_bot_score, bool)
        or not isinstance(genuine_score, (int, float))
        or not isinstance(response_bot_score, (int, float))
        or not math.isfinite(genuine_score)
        or not math.isfinite(response_bot_score)
        or not 0 <= genuine_score <= 1
        or not 0 <= response_bot_score <= 1
        or not math.isclose(
            genuine_score + response_bot_score, 1.0, abs_tol=1e-6
        )
        or not math.isclose(response_bot_score, bot_score, abs_tol=1e-6)
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
        or prediction != ("Likely Bot" if bot_score >= threshold else "Likely Genuine")
        or response_data.get("probability") is not None
        or response_data.get("confidence") is not None
        or response_data.get("feature_count_required") != 50072
        or response_data.get("feature_count_constructed") != 50072
        or audit.get("feature_order_validated") is not True
        or audit.get("preprocessing_validated") is not True
        or not isinstance(response_data.get("model"), str)
    ):
        return {"request_status": "invalid_response", "http_status": status_code}

    ood = response_data.get("ood_validation")
    warnings = ood.get("audit_warnings", []) if isinstance(ood, dict) else []
    if not isinstance(warnings, list):
        warnings = []
    return {
        "request_status": "success",
        "http_status": status_code,
        "model": response_data["model"],
        "prediction": prediction,
        "bot_score_uncalibrated": float(bot_score),
        "genuine_score_uncalibrated": float(genuine_score),
        "probability": response_data.get("probability"),
        "confidence": response_data.get("confidence"),
        "threshold": float(threshold),
        "feature_count_required": response_data.get("feature_count_required"),
        "feature_count_constructed": response_data.get("feature_count_constructed"),
        "feature_order_validated": audit.get("feature_order_validated"),
        "preprocessing_validated": audit.get("preprocessing_validated"),
        "ood_warnings": warnings,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in results if row["request_status"] == "success"]
    evaluated = [
        row
        for row in successful
        if row["reference_class"] in {"genuine_verified", "bot_verified"}
    ]
    confusion = {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0}
    for row in evaluated:
        truth_is_bot = row["reference_class"] == "bot_verified"
        prediction_is_bot = row["prediction"] == "Likely Bot"
        if truth_is_bot and prediction_is_bot:
            confusion["true_positive"] += 1
        elif truth_is_bot:
            confusion["false_negative"] += 1
        elif prediction_is_bot:
            confusion["false_positive"] += 1
        else:
            confusion["true_negative"] += 1
    bot_reference_count = confusion["true_positive"] + confusion["false_negative"]
    genuine_reference_count = confusion["true_negative"] + confusion["false_positive"]
    return {
        "registry_accounts": len(results),
        "successful_analyses": len(successful),
        "failed_analyses": len(results) - len(successful),
        "suspicious_unverified_successes": sum(
            row["request_status"] == "success"
            and row["reference_class"] == "suspicious_unverified"
            for row in results
        ),
        "independently_labeled_successes": len(evaluated),
        "confusion_counts": confusion,
        "false_positive_rate": (
            confusion["false_positive"] / genuine_reference_count
            if genuine_reference_count
            else None
        ),
        "false_negative_rate": (
            confusion["false_negative"] / bot_reference_count
            if bot_reference_count
            else None
        ),
        "metric_denominator_note": (
            "Only successful genuine_verified and bot_verified records are included. "
            "suspicious_unverified records are excluded from error metrics."
        ),
    }


Analyzer = Callable[[str, str, float], dict[str, Any]]


def run_validation(
    registry: list[dict[str, str]],
    output_path: Path,
    *,
    base_url: str,
    timeout: float = 60.0,
    delay_seconds: float = 1.0,
    analyzer: Analyzer = analyze_account,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run sequential analysis and write results without replacing prior runs."""
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    if timeout <= 0 or delay_seconds < 0:
        raise ValueError("timeout must be positive and delay_seconds must be non-negative")
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite existing summary: {summary_path}")

    results = []
    for index, record in enumerate(registry):
        response = analyzer(base_url, record["username"], timeout)
        results.append({**record, **response})
        print(
            f"[{index + 1}/{len(registry)}] {record['record_id']}: "
            f"{response.get('request_status', 'invalid_result')}"
        )
        if delay_seconds > 0 and index + 1 < len(registry):
            time.sleep(delay_seconds)

    summary = {
        "protocol": "TwiBot-20 live-account validation; no model training or tuning",
        "api_base_url": base_url,
        "models_observed": sorted(
            {
                row["model"]
                for row in results
                if row.get("request_status") == "success" and row.get("model")
            }
        ),
        **summarize_results(results),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(
            key for row in results for key in row
        )))
        writer.writeheader()
        for row in results:
            serialized = {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(serialized)
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    return results, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument(
        "--confirm-live-lookups",
        action="store_true",
        help="confirm that the run will make live provider/API requests",
    )
    args = parser.parse_args(argv)
    if not args.confirm_live_lookups:
        parser.error("pass --confirm-live-lookups to authorize live account lookups")
    try:
        registry = load_registry(args.registry)
        _, summary = run_validation(
            registry,
            args.output,
            base_url=args.base_url,
            timeout=args.timeout,
            delay_seconds=args.delay_seconds,
        )
    except (OSError, ValueError, FileExistsError) as exc:
        print(f"Validation stopped: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
