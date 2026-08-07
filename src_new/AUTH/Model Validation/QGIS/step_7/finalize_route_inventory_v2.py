#!/usr/bin/env python3
"""
Step 7 — Finalize a manually reviewed, route-based object inventory.

Key changes from the previous version
-------------------------------------
1. Physical objects are grouped globally by:
       physical_object_id + final_class
   They are NOT split by route_id.

   Example:
       R01_INBOUND observation  -> LM_001
       R01_OUTBOUND observation -> LM_001

   Both become one final physical-object row.

2. Route positions are summarized separately in:
       physical_object_route_positions.csv

   This avoids averaging incomparable inbound/outbound chainages.

3. The script auto-detects comma or semicolon CSV delimiters.

4. It counts false detections at:
   - observation/track level;
   - frame-detection level, using frames_seen;
   - predicted-class and false-positive-reason level.

5. Conflicting review rows are never silently accepted. They are written to:
       review_conflicts.csv

Typical command
---------------
python finalize_route_inventory_v2.py \
    --review observation_review_template_complete2.csv \
    --output final_route_inventory

Use:
    --fail-on-conflicts

to stop with a non-zero exit code when conflicts exist.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REQUIRED_COLUMNS = {
    "observation_id",
    "candidate_object_id",
    "video",
    "pass_id",
    "route_id",
    "track_id",
    "class_name",
    "representative_frame",
    "representative_time_s",
    "representative_chainage_m",
    "screen_side",
    "frames_seen",
    "mean_confidence",
    "review_decision",
    "corrected_class",
    "false_positive_reason",
    "manual_physical_object_id",
    "review_notes",
}

VALID_DECISIONS = {
    "TRUE_OBJECT",
    "FALSE_POSITIVE",
    "WRONG_CLASS",
    "UNSURE",
    "",
}

DECISION_ALIASES = {
    "TRUE": "TRUE_OBJECT",
    "TRUEOBJECT": "TRUE_OBJECT",
    "TRUE OBJECT": "TRUE_OBJECT",
    "FALSE": "FALSE_POSITIVE",
    "FALSEPOSITIVE": "FALSE_POSITIVE",
    "FALSE POSITIVE": "FALSE_POSITIVE",
    "WRONGCLASS": "WRONG_CLASS",
    "WRONG CLASS": "WRONG_CLASS",
    # This typo appears in the supplied Step 6 CSV.
    # "Wrong object" means the detected item is not a target object.
    "WRONG_OBLECT": "FALSE_POSITIVE",
    "WRONG OBJECT": "FALSE_POSITIVE",
    "NOT_SURE": "UNSURE",
    "NOT SURE": "UNSURE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize reviewed route observations into global physical-object "
            "and false-detection summaries."
        )
    )
    parser.add_argument(
        "--review",
        required=True,
        help="Completed observation-review CSV from Step 6",
    )
    parser.add_argument(
        "--output",
        default="final_route_inventory",
        help="Output directory (default: final_route_inventory)",
    )
    parser.add_argument(
        "--delimiter",
        choices=["auto", "comma", "semicolon", "tab"],
        default="auto",
        help="Input CSV delimiter (default: auto)",
    )
    parser.add_argument(
        "--fail-on-conflicts",
        action="store_true",
        help=(
            "Return exit code 2 when conflicting review rows are found. "
            "Diagnostic CSVs are still written."
        ),
    )
    return parser.parse_args()


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def parse_int(value: Any, default: int = 0) -> int:
    text = clean(value)
    if not text:
        return default
    try:
        return int(float(text.replace(",", ".")))
    except ValueError:
        return default


def parse_float(value: Any) -> float | None:
    text = clean(value)
    if not text:
        return None

    # Support both decimal point and decimal comma.
    if "," in text and "." not in text:
        text = text.replace(",", ".")

    try:
        number = float(text)
    except ValueError:
        return None

    return number if math.isfinite(number) else None


def detect_delimiter(path: Path, requested: str) -> str:
    if requested == "comma":
        return ","
    if requested == "semicolon":
        return ";"
    if requested == "tab":
        return "\t"

    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:8192]

    counts = {
        ";": sample.count(";"),
        ",": sample.count(","),
        "\t": sample.count("\t"),
    }

    delimiter = max(counts, key=counts.get)
    if counts[delimiter] == 0:
        raise ValueError("Could not detect the CSV delimiter")

    return delimiter


def read_csv_rows(
    path: Path,
    delimiter: str,
) -> tuple[list[str], list[dict[str, str]]]:
    """
    Read a CSV while safely ignoring duplicate blank columns at the end.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            raw_header = next(reader)
        except StopIteration as error:
            raise ValueError("The review CSV is empty") from error

        # Keep only the first occurrence of each non-empty header name.
        header_positions: dict[str, int] = {}
        ordered_headers: list[str] = []

        for index, value in enumerate(raw_header):
            name = clean(value)
            if not name or name in header_positions:
                continue
            header_positions[name] = index
            ordered_headers.append(name)

        missing = sorted(REQUIRED_COLUMNS - set(ordered_headers))
        if missing:
            raise ValueError(
                "The review CSV is missing required columns: "
                + ", ".join(missing)
            )

        rows: list[dict[str, str]] = []
        for raw_row in reader:
            if not any(clean(value) for value in raw_row):
                continue

            item = {
                name: clean(raw_row[index]) if index < len(raw_row) else ""
                for name, index in header_positions.items()
            }
            rows.append(item)

    return ordered_headers, rows


def normalize_decision(value: str) -> tuple[str, str]:
    """
    Return normalized decision and an optional normalization note.
    """
    original = clean(value)
    upper = original.upper().replace("-", "_")
    upper = " ".join(upper.split())

    if upper in VALID_DECISIONS:
        return upper, ""

    if upper in DECISION_ALIASES:
        normalized = DECISION_ALIASES[upper]
        return normalized, f"Normalized '{original}' to '{normalized}'"

    compact = upper.replace("_", "").replace(" ", "")
    if compact in DECISION_ALIASES:
        normalized = DECISION_ALIASES[compact]
        return normalized, f"Normalized '{original}' to '{normalized}'"

    return upper, ""


def is_false_id(value: str) -> bool:
    text = clean(value).upper()
    return (
        text.startswith("FALSE")
        or text.startswith("FP_")
        or text.startswith("FP-")
    )


def canonicalize_physical_object_id(value: str) -> tuple[str, str]:
    """
    Normalize common manual-ID variants.

    Examples:
        LM_01  -> LM_001
        TL-1   -> TL_001
        TS001  -> TS_001

    IDs outside the LM/TS/TL numeric convention are preserved unchanged.
    """
    original = clean(value)
    upper = original.upper()

    match = re.fullmatch(r"(LM|TS|TL)[_\- ]?0*(\d+)", upper)
    if not match:
        return original, ""

    prefix, number_text = match.groups()
    normalized = f"{prefix}_{int(number_text):03d}"

    if normalized != original:
        return normalized, f"Normalized physical ID '{original}' to '{normalized}'"

    return normalized, ""


def route_direction(route_id: str) -> str:
    upper = clean(route_id).upper()
    if "INBOUND" in upper:
        return "INBOUND"
    if "OUTBOUND" in upper:
        return "OUTBOUND"
    return "UNSPECIFIED"


def ordered_unique(values: Iterable[str]) -> list[str]:
    return sorted({clean(value) for value in values if clean(value)})


def average(values: Iterable[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def minimum(values: Iterable[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return min(valid) if valid else None


def maximum(values: Iterable[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return max(valid) if valid else None


def median(values: Iterable[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return statistics.median(valid) if valid else None


def format_number(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 9)
    return value


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: format_number(row.get(key, ""))
                for key in fieldnames
            })


def append_conflict(
    conflicts: list[dict[str, Any]],
    row: dict[str, Any],
    reason: str,
) -> None:
    conflict = dict(row)
    conflict["conflict_reason"] = reason
    conflicts.append(conflict)


def main() -> int:
    args = parse_args()

    review_path = Path(args.review)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if not review_path.exists():
        raise FileNotFoundError(f"Review CSV not found: {review_path}")

    delimiter = detect_delimiter(review_path, args.delimiter)
    original_headers, input_rows = read_csv_rows(review_path, delimiter)

    accepted: list[dict[str, Any]] = []
    false_positive: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    wrong_class_rows: list[dict[str, Any]] = []
    normalization_log: list[dict[str, Any]] = []
    physical_id_normalization_log: list[dict[str, Any]] = []

    for source_row in input_rows:
        row: dict[str, Any] = dict(source_row)

        decision, normalization_note = normalize_decision(
            row.get("review_decision", "")
        )
        row["normalized_review_decision"] = decision

        if normalization_note:
            normalization_log.append({
                "observation_id": row.get("observation_id", ""),
                "original_review_decision": row.get("review_decision", ""),
                "normalized_review_decision": decision,
                "normalization_note": normalization_note,
            })

        manual_id_original = clean(row.get("manual_physical_object_id"))
        manual_id, physical_id_note = canonicalize_physical_object_id(
            manual_id_original
        )
        row["manual_physical_object_id_original"] = manual_id_original
        row["manual_physical_object_id_normalized"] = manual_id

        if physical_id_note:
            physical_id_normalization_log.append({
                "observation_id": row.get("observation_id", ""),
                "original_physical_object_id": manual_id_original,
                "normalized_physical_object_id": manual_id,
                "normalization_note": physical_id_note,
            })

        corrected_class = clean(row.get("corrected_class"))
        predicted_class = clean(row.get("class_name"))

        row["frames_seen_numeric"] = parse_int(row.get("frames_seen"))
        row["mean_confidence_numeric"] = parse_float(
            row.get("mean_confidence")
        )
        row["representative_chainage_numeric"] = parse_float(
            row.get("representative_chainage_m")
        )

        if decision not in VALID_DECISIONS:
            append_conflict(
                conflicts,
                row,
                f"Unknown review_decision: {row.get('review_decision', '')}",
            )
            continue

        if decision in {"", "UNSURE"}:
            row["final_class"] = ""
            row["physical_object_id"] = ""
            unresolved.append(row)
            continue

        if decision == "FALSE_POSITIVE":
            row["final_class"] = ""
            row["physical_object_id"] = ""
            row["false_positive_reason"] = (
                clean(row.get("false_positive_reason"))
                or "UNSPECIFIED"
            )
            false_positive.append(row)
            continue

        # Accepted decisions must not use a FALSE_* physical-object ID.
        if is_false_id(manual_id):
            append_conflict(
                conflicts,
                row,
                (
                    f"Decision is {decision}, but manual_physical_object_id "
                    f"is '{manual_id}', which is a false-detection marker"
                ),
            )
            continue

        if not manual_id:
            append_conflict(
                conflicts,
                row,
                (
                    f"Decision is {decision}, but "
                    "manual_physical_object_id is blank"
                ),
            )
            continue

        if decision == "WRONG_CLASS":
            if not corrected_class:
                append_conflict(
                    conflicts,
                    row,
                    (
                        "review_decision is WRONG_CLASS, but "
                        "corrected_class is blank"
                    ),
                )
                continue

            row["final_class"] = corrected_class
            wrong_class_rows.append(row)
        else:
            row["final_class"] = predicted_class

        row["physical_object_id"] = manual_id
        accepted.append(row)

    # Ensure one global physical-object ID is not assigned to multiple classes.
    classes_by_id: dict[str, set[str]] = defaultdict(set)
    for row in accepted:
        classes_by_id[row["physical_object_id"]].add(row["final_class"])

    invalid_ids = {
        physical_id: classes
        for physical_id, classes in classes_by_id.items()
        if len(classes) > 1
    }

    if invalid_ids:
        still_accepted: list[dict[str, Any]] = []

        for row in accepted:
            physical_id = row["physical_object_id"]
            if physical_id in invalid_ids:
                append_conflict(
                    conflicts,
                    row,
                    (
                        f"physical_object_id '{physical_id}' is assigned to "
                        f"multiple final classes: "
                        f"{', '.join(sorted(invalid_ids[physical_id]))}"
                    ),
                )
            else:
                still_accepted.append(row)

        accepted = still_accepted

    # -----------------------
    # Global physical objects
    # -----------------------
    object_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        key = (row["physical_object_id"], row["final_class"])
        object_groups[key].append(row)

    inventory_rows: list[dict[str, Any]] = []

    for (physical_id, final_class), rows in sorted(object_groups.items()):
        routes = ordered_unique(row["route_id"] for row in rows)
        passes = ordered_unique(row["pass_id"] for row in rows)
        videos = ordered_unique(row["video"] for row in rows)
        directions = ordered_unique(route_direction(row["route_id"]) for row in rows)
        observations = ordered_unique(row["observation_id"] for row in rows)

        confidences = [
            row["mean_confidence_numeric"]
            for row in rows
        ]

        inventory_rows.append({
            "physical_object_id": physical_id,
            "final_class": final_class,
            "routes_seen": ",".join(routes),
            "route_count": len(routes),
            "directions_seen": ",".join(directions),
            "passes_seen": len(passes),
            "videos_seen": len(videos),
            "observation_count": len(rows),
            "total_detected_frames": sum(
                row["frames_seen_numeric"] for row in rows
            ),
            "mean_confidence": average(confidences),
            "minimum_confidence": minimum(confidences),
            "maximum_confidence": maximum(confidences),
            "source_observations": ",".join(observations),
            "source_videos": ",".join(videos),
            "evidence_status": (
                "CONFIRMED_MULTIPLE_PASSES"
                if len(passes) >= 2
                else "SINGLE_PASS_VALIDATED"
            ),
        })

    # -------------------------------------
    # Route-specific position summaries
    # -------------------------------------
    route_position_groups: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in accepted:
        key = (
            row["physical_object_id"],
            row["final_class"],
            row["route_id"],
        )
        route_position_groups[key].append(row)

    route_position_rows: list[dict[str, Any]] = []

    for (
        physical_id,
        final_class,
        route_id,
    ), rows in sorted(route_position_groups.items()):
        chainages = [
            row["representative_chainage_numeric"]
            for row in rows
        ]
        valid_chainages = [
            value for value in chainages if value is not None
        ]

        min_chainage = minimum(valid_chainages)
        max_chainage = maximum(valid_chainages)

        route_position_rows.append({
            "physical_object_id": physical_id,
            "final_class": final_class,
            "route_id": route_id,
            "direction": route_direction(route_id),
            "median_chainage_m": median(valid_chainages),
            "minimum_chainage_m": min_chainage,
            "maximum_chainage_m": max_chainage,
            "chainage_spread_m": (
                max_chainage - min_chainage
                if min_chainage is not None and max_chainage is not None
                else None
            ),
            "observation_count": len(rows),
            "passes_seen": len(ordered_unique(
                row["pass_id"] for row in rows
            )),
            "videos_seen": len(ordered_unique(
                row["video"] for row in rows
            )),
            "source_observations": ",".join(ordered_unique(
                row["observation_id"] for row in rows
            )),
        })

    # -----------------------
    # Unique-object counts
    # -----------------------
    total_counts: Counter[str] = Counter(
        row["final_class"] for row in inventory_rows
    )

    unique_counts_total_rows = [
        {
            "final_class": final_class,
            "unique_physical_objects": count,
        }
        for final_class, count in sorted(total_counts.items())
    ]

    route_object_sets: dict[
        tuple[str, str],
        set[str],
    ] = defaultdict(set)

    for row in accepted:
        route_object_sets[
            (row["route_id"], row["final_class"])
        ].add(row["physical_object_id"])

    unique_counts_by_route_rows = [
        {
            "route_id": route_id,
            "final_class": final_class,
            "unique_physical_objects": len(physical_ids),
        }
        for (
            route_id,
            final_class,
        ), physical_ids in sorted(route_object_sets.items())
    ]

    # -----------------------
    # False-detection counts
    # -----------------------
    false_groups: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in false_positive:
        false_groups[
            (
                clean(row.get("class_name")) or "UNSPECIFIED_CLASS",
                clean(row.get("false_positive_reason")) or "UNSPECIFIED",
            )
        ].append(row)

    false_by_reason_rows = []

    for (
        predicted_class,
        reason,
    ), rows in sorted(false_groups.items()):
        false_by_reason_rows.append({
            "predicted_class": predicted_class,
            "false_positive_reason": reason,
            "false_observations": len(rows),
            "false_frame_detections": sum(
                row["frames_seen_numeric"] for row in rows
            ),
            "mean_confidence": average(
                row["mean_confidence_numeric"] for row in rows
            ),
        })

    false_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in false_positive:
        false_by_class[
            clean(row.get("class_name")) or "UNSPECIFIED_CLASS"
        ].append(row)

    false_by_class_rows = [
        {
            "predicted_class": predicted_class,
            "false_observations": len(rows),
            "false_frame_detections": sum(
                row["frames_seen_numeric"] for row in rows
            ),
        }
        for predicted_class, rows in sorted(false_by_class.items())
    ]

    summary_rows = [
        {
            "measure": "input_observations",
            "value": len(input_rows),
        },
        {
            "measure": "accepted_observations",
            "value": len(accepted),
        },
        {
            "measure": "unique_physical_objects",
            "value": len(inventory_rows),
        },
        {
            "measure": "false_positive_observations",
            "value": len(false_positive),
        },
        {
            "measure": "false_frame_detections",
            "value": sum(
                row["frames_seen_numeric"] for row in false_positive
            ),
        },
        {
            "measure": "wrong_class_observations_accepted",
            "value": len([
                row for row in accepted
                if row["normalized_review_decision"] == "WRONG_CLASS"
            ]),
        },
        {
            "measure": "unresolved_observations",
            "value": len(unresolved),
        },
        {
            "measure": "conflicting_observations_excluded",
            "value": len(conflicts),
        },
        {
            "measure": "normalized_decision_values",
            "value": len(normalization_log),
        },
        {
            "measure": "normalized_physical_object_ids",
            "value": len(physical_id_normalization_log),
        },
    ]

    # -----------------------
    # Write outputs
    # -----------------------
    observation_fields = original_headers + [
        "manual_physical_object_id_original",
        "manual_physical_object_id_normalized",
        "normalized_review_decision",
        "final_class",
        "physical_object_id",
    ]

    conflict_fields = original_headers + [
        "manual_physical_object_id_original",
        "manual_physical_object_id_normalized",
        "normalized_review_decision",
        "conflict_reason",
    ]

    write_csv(
        output / "accepted_route_observations.csv",
        observation_fields,
        accepted,
    )

    write_csv(
        output / "physical_object_inventory.csv",
        [
            "physical_object_id",
            "final_class",
            "routes_seen",
            "route_count",
            "directions_seen",
            "passes_seen",
            "videos_seen",
            "observation_count",
            "total_detected_frames",
            "mean_confidence",
            "minimum_confidence",
            "maximum_confidence",
            "source_observations",
            "source_videos",
            "evidence_status",
        ],
        inventory_rows,
    )

    write_csv(
        output / "physical_object_route_positions.csv",
        [
            "physical_object_id",
            "final_class",
            "route_id",
            "direction",
            "median_chainage_m",
            "minimum_chainage_m",
            "maximum_chainage_m",
            "chainage_spread_m",
            "observation_count",
            "passes_seen",
            "videos_seen",
            "source_observations",
        ],
        route_position_rows,
    )

    write_csv(
        output / "unique_counts_total.csv",
        [
            "final_class",
            "unique_physical_objects",
        ],
        unique_counts_total_rows,
    )

    write_csv(
        output / "unique_counts_by_route.csv",
        [
            "route_id",
            "final_class",
            "unique_physical_objects",
        ],
        unique_counts_by_route_rows,
    )

    write_csv(
        output / "false_positive_observations.csv",
        observation_fields,
        false_positive,
    )

    write_csv(
        output / "false_detection_counts_by_reason.csv",
        [
            "predicted_class",
            "false_positive_reason",
            "false_observations",
            "false_frame_detections",
            "mean_confidence",
        ],
        false_by_reason_rows,
    )

    write_csv(
        output / "false_detection_counts_by_class.csv",
        [
            "predicted_class",
            "false_observations",
            "false_frame_detections",
        ],
        false_by_class_rows,
    )

    write_csv(
        output / "false_detection_summary.csv",
        ["measure", "value"],
        summary_rows,
    )

    write_csv(
        output / "wrong_class_observations.csv",
        observation_fields,
        [
            row for row in accepted
            if row["normalized_review_decision"] == "WRONG_CLASS"
        ],
    )

    write_csv(
        output / "unresolved_observations.csv",
        observation_fields,
        unresolved,
    )

    write_csv(
        output / "review_conflicts.csv",
        conflict_fields,
        conflicts,
    )

    write_csv(
        output / "decision_normalization_log.csv",
        [
            "observation_id",
            "original_review_decision",
            "normalized_review_decision",
            "normalization_note",
        ],
        normalization_log,
    )

    write_csv(
        output / "physical_id_normalization_log.csv",
        [
            "observation_id",
            "original_physical_object_id",
            "normalized_physical_object_id",
            "normalization_note",
        ],
        physical_id_normalization_log,
    )

    print(f"Input observations: {len(input_rows):,}")
    print(f"Accepted observations: {len(accepted):,}")
    print(f"Unique physical objects: {len(inventory_rows):,}")
    print(f"False-positive observations: {len(false_positive):,}")
    print(
        "False frame detections: "
        f"{sum(row['frames_seen_numeric'] for row in false_positive):,}"
    )
    print(f"Unresolved observations: {len(unresolved):,}")
    print(f"Conflicting observations excluded: {len(conflicts):,}")
    print(f"Results written to: {output.resolve()}")

    if conflicts:
        print(
            "\nReview review_conflicts.csv before treating the counts as final.",
            file=sys.stderr,
        )
        if args.fail_on_conflicts:
            return 2

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
