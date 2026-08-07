#!/usr/bin/env python3
"""
Count YOLO object encounters in one or more videos using Ultralytics tracking.

Definition:
    One encounter = one valid tracker ID within one video.

Important limitation:
    Encounters are NOT guaranteed to be unique physical objects across videos.
    The same real-world sign/light/landmark can be counted again in another video,
    and a fragmented track can occasionally count the same object twice.

Outputs:
    raw_detections.csv
    track_summary_all.csv
    track_summary_valid.csv
    encounter_counts_by_video.csv
    encounter_counts_total.csv
    annotated_videos/                  (optional)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
from ultralytics import YOLO


VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"
}

RAW_COLUMNS = [
    "video",
    "frame",
    "time_s",
    "fps",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "box_width",
    "box_height",
]

TRACK_COLUMNS = [
    "video",
    "track_id",
    "class_id",
    "class_name",
    "class_votes",
    "frames_seen",
    "class_agreement",
    "first_frame",
    "last_frame",
    "first_time_s",
    "last_time_s",
    "observed_span_s",
    "mean_confidence",
    "max_confidence",
    "mean_box_width",
    "mean_box_height",
    "valid_track",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track objects in videos and count each valid track once "
            "as an object encounter."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the custom YOLO weights, e.g. runs/detect/train/weights/best.pt",
    )
    parser.add_argument(
        "--videos",
        required=True,
        help="Directory containing the original video files",
    )
    parser.add_argument(
        "--output",
        default="tracking_results",
        help="Output directory (default: tracking_results)",
    )
    parser.add_argument(
        "--tracker",
        default="botsort.yaml",
        help="Ultralytics tracker configuration (default: botsort.yaml)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum detection confidence passed to the tracker (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.50,
        help="Detection IoU threshold (default: 0.50)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Inference image size. Larger values can help small signs but cost speed (default: 1280)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, e.g. 0, cpu, mps. Omit for automatic selection.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=3,
        help="Minimum number of frames required for a valid track (default: 3)",
    )
    parser.add_argument(
        "--min-mean-conf",
        type=float,
        default=0.30,
        help="Minimum mean confidence required for a valid track (default: 0.30)",
    )
    parser.add_argument(
        "--min-class-agreement",
        type=float,
        default=0.60,
        help=(
            "Minimum fraction of a track's detections that must agree with its "
            "dominant class (default: 0.60)"
        ),
    )
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        help="Save videos with bounding boxes and track IDs",
    )
    return parser.parse_args()


def find_videos(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Video directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"--videos must point to a directory: {directory}")

    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def probe_video(video_path: Path) -> tuple[float, int, int]:
    """Return FPS, width and height. Falls back to safe values when unavailable."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 30.0, 0, 0

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    if not fps or fps <= 0:
        fps = 30.0

    return fps, width, height


def class_name_from_names(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def create_writer(
    output_path: Path,
    fps: float,
    frame_width: int,
    frame_height: int,
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create annotated video: {output_path}")
    return writer


def process_video(
    video_path: Path,
    args: argparse.Namespace,
    annotated_directory: Path,
) -> list[dict[str, Any]]:
    """
    Process one video.

    A new model instance is created for every video so tracker state cannot leak
    from one unrelated recording into the next.
    """
    print(f"Processing: {video_path.name}", flush=True)

    model = YOLO(args.model)
    fps, width, height = probe_video(video_path)

    track_kwargs: dict[str, Any] = {
        "source": str(video_path),
        "stream": True,
        "tracker": args.tracker,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device is not None:
        track_kwargs["device"] = args.device

    results = model.track(**track_kwargs)

    observations: list[dict[str, Any]] = []
    writer: cv2.VideoWriter | None = None

    try:
        for frame_number, result in enumerate(results):
            if args.save_annotated:
                annotated_frame = result.plot()

                if writer is None:
                    frame_height, frame_width = annotated_frame.shape[:2]
                    output_video = (
                        annotated_directory / f"{video_path.stem}_tracked.mp4"
                    )
                    writer = create_writer(
                        output_video,
                        fps,
                        frame_width,
                        frame_height,
                    )

                writer.write(annotated_frame)

            boxes = result.boxes
            if boxes is None or len(boxes) == 0 or boxes.id is None:
                continue

            track_ids = boxes.id.int().cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.float().cpu().tolist()
            coordinates = boxes.xyxy.float().cpu().tolist()

            for track_id, class_id, confidence, xyxy in zip(
                track_ids,
                class_ids,
                confidences,
                coordinates,
            ):
                x1, y1, x2, y2 = map(float, xyxy)
                class_id = int(class_id)

                observations.append(
                    {
                        "video": video_path.name,
                        "frame": int(frame_number),
                        "time_s": float(frame_number / fps),
                        "fps": float(fps),
                        "track_id": int(track_id),
                        "class_id": class_id,
                        "class_name": class_name_from_names(
                            model.names,
                            class_id,
                        ),
                        "confidence": float(confidence),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "center_x": float((x1 + x2) / 2.0),
                        "center_y": float((y1 + y2) / 2.0),
                        "box_width": float(x2 - x1),
                        "box_height": float(y2 - y1),
                    }
                )
    finally:
        if writer is not None:
            writer.release()

    print(
        f"  Stored {len(observations):,} tracked frame detections.",
        flush=True,
    )
    return observations


def summarize_tracks(raw: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """
    Produce one row per (video, track_id).

    The assigned class is the class observed most often for that track.
    This prevents one tracker ID from being counted in multiple classes when
    occasional frame-level class changes occur.
    """
    if raw.empty:
        return pd.DataFrame(columns=TRACK_COLUMNS)

    track_keys = ["video", "track_id"]

    base = (
        raw.groupby(track_keys, as_index=False)
        .agg(
            frames_seen=("frame", "nunique"),
            first_frame=("frame", "min"),
            last_frame=("frame", "max"),
            first_time_s=("time_s", "min"),
            last_time_s=("time_s", "max"),
            mean_confidence=("confidence", "mean"),
            max_confidence=("confidence", "max"),
            mean_box_width=("box_width", "mean"),
            mean_box_height=("box_height", "mean"),
        )
    )

    class_votes = (
        raw.groupby(
            ["video", "track_id", "class_id", "class_name"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "class_votes"})
        .sort_values(
            ["video", "track_id", "class_votes", "class_id"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(track_keys, keep="first")
    )

    summary = base.merge(class_votes, on=track_keys, how="left")
    summary["class_agreement"] = (
        summary["class_votes"] / summary["frames_seen"]
    )
    summary["observed_span_s"] = (
        summary["last_time_s"] - summary["first_time_s"]
    )

    summary["valid_track"] = (
        (summary["frames_seen"] >= args.min_frames)
        & (summary["mean_confidence"] >= args.min_mean_conf)
        & (summary["class_agreement"] >= args.min_class_agreement)
    )

    summary = summary[
        [
            "video",
            "track_id",
            "class_id",
            "class_name",
            "class_votes",
            "frames_seen",
            "class_agreement",
            "first_frame",
            "last_frame",
            "first_time_s",
            "last_time_s",
            "observed_span_s",
            "mean_confidence",
            "max_confidence",
            "mean_box_width",
            "mean_box_height",
            "valid_track",
        ]
    ].sort_values(["video", "class_name", "track_id"])

    return summary


def save_outputs(
    output_directory: Path,
    raw: pd.DataFrame,
    tracks: pd.DataFrame,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)

    raw.to_csv(output_directory / "raw_detections.csv", index=False)
    tracks.to_csv(output_directory / "track_summary_all.csv", index=False)

    if tracks.empty:
        valid_tracks = tracks.copy()
        by_video = pd.DataFrame(
            columns=["video", "class_name", "object_encounters"]
        )
        totals = pd.DataFrame(
            columns=["class_name", "object_encounters"]
        )
    else:
        valid_tracks = tracks[tracks["valid_track"]].copy()

        by_video = (
            valid_tracks.groupby(
                ["video", "class_name"],
                as_index=False,
            )
            .size()
            .rename(columns={"size": "object_encounters"})
            .sort_values(["video", "class_name"])
        )

        totals = (
            valid_tracks.groupby("class_name", as_index=False)
            .size()
            .rename(columns={"size": "object_encounters"})
            .sort_values("class_name")
        )

    valid_tracks.to_csv(
        output_directory / "track_summary_valid.csv",
        index=False,
    )
    by_video.to_csv(
        output_directory / "encounter_counts_by_video.csv",
        index=False,
    )
    totals.to_csv(
        output_directory / "encounter_counts_total.csv",
        index=False,
    )

    print("\nTotal valid object encounters:")
    if totals.empty:
        print("  No valid tracks were found.")
    else:
        print(totals.to_string(index=False))

    print(f"\nResults saved to: {output_directory.resolve()}")


def main() -> int:
    args = parse_args()

    model_path = Path(args.model)
    video_directory = Path(args.videos)
    output_directory = Path(args.output)
    annotated_directory = output_directory / "annotated_videos"

    if not model_path.exists():
        print(f"ERROR: Model weights not found: {model_path}", file=sys.stderr)
        return 2

    try:
        videos = find_videos(video_directory)
    except (FileNotFoundError, NotADirectoryError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    if not videos:
        print(
            f"ERROR: No supported videos found under: {video_directory}",
            file=sys.stderr,
        )
        return 2

    print(f"Found {len(videos)} video(s).")
    print("Counting rule: one valid tracker ID per video = one encounter.")
    print(
        "These are encounter counts, not unique mapped physical-object counts.\n"
    )

    all_observations: list[dict[str, Any]] = []

    for video_path in videos:
        try:
            all_observations.extend(
                process_video(
                    video_path,
                    args,
                    annotated_directory,
                )
            )
        except Exception as error:
            print(
                f"ERROR while processing {video_path.name}: {error}",
                file=sys.stderr,
            )
            return 1

    raw = pd.DataFrame(all_observations, columns=RAW_COLUMNS)
    tracks = summarize_tracks(raw, args)
    save_outputs(output_directory, raw, tracks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
