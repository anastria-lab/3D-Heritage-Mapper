#!/usr/bin/env python3
"""
Associate YOLO tracklets with repeated route passes and create provisional
physical-object candidates.

The route geometry represents the vehicle path. The generated points are
approximate CAMERA/OBSERVATION positions, not exact sign/light coordinates.

Required QGIS inputs
--------------------
1. routes.gpkg / layer "routes"
   - route_id: unique route and travel direction, e.g. R01_NORTH
   - one LineString per route_id
   - projected CRS in metres

2. anchors.gpkg / layer "anchors"
   - video: exact video filename
   - frame: video frame number
   - Point geometry manually placed on the route at a recognisable video location

3. video_route_map.csv
   - video, route_id, pass_id
   - repeated recordings use the same route_id and different pass_id values

YOLO inputs
-----------
- raw_detections.csv
- track_summary_valid.csv

Outputs
-------
- computed_anchors.csv
- detection_route_points.gpkg
- detection_route_points.csv
- track_observations.gpkg
- track_observations.csv
- candidate_physical_objects.csv
- observation_review_template.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
from sklearn.cluster import DBSCAN


REQUIRED_ROUTE_COLUMNS = {"route_id"}
REQUIRED_ANCHOR_COLUMNS = {"video", "frame"}
REQUIRED_VIDEO_MAP_COLUMNS = {"video", "route_id", "pass_id"}
REQUIRED_RAW_COLUMNS = {
    "video", "frame", "track_id", "class_name", "confidence",
    "x1", "y1", "x2", "y2", "center_x", "center_y",
}
REQUIRED_TRACK_COLUMNS = {
    "video", "track_id", "class_name", "frames_seen",
    "first_frame", "last_frame", "mean_confidence",
}

DEFAULT_CLASS_EPS = {
    "traffic_signs": 20.0,
    "traffic_lights": 35.0,
    "auth_label": 40.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", required=True, help="QGIS GeoPackage containing route lines")
    parser.add_argument("--routes-layer", default="routes")
    parser.add_argument("--anchors", required=True, help="QGIS GeoPackage containing manual anchor points")
    parser.add_argument("--anchors-layer", default="anchors")
    parser.add_argument("--video-map", required=True, help="video_route_map.csv")
    parser.add_argument("--raw", required=True, help="raw_detections.csv")
    parser.add_argument("--tracks", required=True, help="track_summary_valid.csv")
    parser.add_argument("--output", default="route_inventory_work")
    parser.add_argument(
        "--class-eps-json",
        default=None,
        help=(
            "Optional JSON mapping of class to clustering tolerance in metres, "
            'e.g. \'{"traffic_signs":15,"traffic_lights":30,"auth_label":40}\''
        ),
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def normalize_route_geometry(geometry):
    if isinstance(geometry, LineString):
        return geometry
    if isinstance(geometry, MultiLineString):
        merged = linemerge(geometry)
        if isinstance(merged, LineString):
            return merged
    raise ValueError("Every route_id must resolve to one continuous LineString")


def route_dictionary(routes: gpd.GeoDataFrame) -> dict[str, LineString]:
    dissolved = routes.dissolve(by="route_id", as_index=False)
    result: dict[str, LineString] = {}
    for row in dissolved.itertuples(index=False):
        result[str(row.route_id)] = normalize_route_geometry(row.geometry)
    return result


def infer_video_width(group: pd.DataFrame) -> float:
    max_x = float(group["x2"].max())
    if max_x > 2200:
        return 3840.0
    if max_x > 1200:
        return 1920.0
    return max(max_x, 1.0)


def screen_side(normalized_x: float) -> str:
    if normalized_x < 0.40:
        return "LEFT"
    if normalized_x > 0.60:
        return "RIGHT"
    return "CENTER"


def interpolate_chainage(
    frames: np.ndarray,
    anchor_frames: np.ndarray,
    anchor_chainage: np.ndarray,
) -> np.ndarray:
    values = np.interp(frames, anchor_frames, anchor_chainage)
    outside = (frames < anchor_frames.min()) | (frames > anchor_frames.max())
    values[outside] = np.nan
    return values


def class_eps(args: argparse.Namespace) -> dict[str, float]:
    values = dict(DEFAULT_CLASS_EPS)
    if args.class_eps_json:
        supplied = json.loads(args.class_eps_json)
        values.update({str(key): float(value) for key, value in supplied.items()})
    return values


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    routes = gpd.read_file(args.routes, layer=args.routes_layer)
    anchors = gpd.read_file(args.anchors, layer=args.anchors_layer)
    video_map = pd.read_csv(args.video_map)
    raw = pd.read_csv(args.raw)
    tracks = pd.read_csv(args.tracks)

    require_columns(routes, REQUIRED_ROUTE_COLUMNS, "routes layer")
    require_columns(anchors, REQUIRED_ANCHOR_COLUMNS, "anchors layer")
    require_columns(video_map, REQUIRED_VIDEO_MAP_COLUMNS, "video route map")
    require_columns(raw, REQUIRED_RAW_COLUMNS, "raw detections")
    require_columns(tracks, REQUIRED_TRACK_COLUMNS, "valid track summary")

    if routes.crs is None or not routes.crs.is_projected:
        raise ValueError(
            "The routes layer must use a projected CRS measured in metres."
        )

    if anchors.crs is None:
        raise ValueError("The anchors layer must have a CRS.")
    anchors = anchors.to_crs(routes.crs)

    if video_map["video"].duplicated().any():
        duplicates = video_map.loc[video_map["video"].duplicated(), "video"].tolist()
        raise ValueError(f"Each video must occur once in video_route_map.csv: {duplicates}")

    route_geometries = route_dictionary(routes)
    unknown_routes = sorted(set(video_map["route_id"].astype(str)) - set(route_geometries))
    if unknown_routes:
        raise ValueError(f"video_route_map.csv references unknown routes: {unknown_routes}")

    anchors = anchors.merge(
        video_map[["video", "route_id", "pass_id"]],
        on="video",
        how="left",
        validate="many_to_one",
    )
    if anchors["route_id"].isna().any():
        missing = sorted(anchors.loc[anchors["route_id"].isna(), "video"].unique())
        raise ValueError(f"Anchor videos missing from video map: {missing}")

    anchors["chainage_m"] = [
        route_geometries[str(route_id)].project(point)
        for route_id, point in zip(anchors["route_id"], anchors.geometry)
    ]
    anchors["snapped_geometry"] = [
        route_geometries[str(route_id)].interpolate(chainage)
        for route_id, chainage in zip(anchors["route_id"], anchors["chainage_m"])
    ]

    anchor_rows = []
    for video, group in anchors.groupby("video"):
        group = group.sort_values("frame")
        if len(group) < 2:
            raise ValueError(f"{video} needs at least two route anchors")

        frame_diff = np.diff(group["frame"].to_numpy(dtype=float))
        if np.any(frame_diff <= 0):
            raise ValueError(f"{video} has duplicated or decreasing anchor frame numbers")

        chainage_diff = np.diff(group["chainage_m"].to_numpy(dtype=float))
        if np.any(chainage_diff < -1.0):
            raise ValueError(
                f"{video} has decreasing chainage. Draw a separate route in the "
                "actual travel direction, or reverse the line."
            )

        anchor_rows.append(group)

    anchors_checked = pd.concat(anchor_rows, ignore_index=True)
    anchors_checked.drop(columns="snapped_geometry").to_csv(
        output / "computed_anchors.csv",
        index=False,
    )

    valid_keys = tracks[["video", "track_id"]].drop_duplicates()
    detections = raw.merge(valid_keys, on=["video", "track_id"], how="inner")
    detections = detections.merge(
        video_map,
        on="video",
        how="inner",
        validate="many_to_one",
    )

    if detections["route_id"].isna().any():
        missing = sorted(detections.loc[detections["route_id"].isna(), "video"].unique())
        raise ValueError(f"Detection videos missing from video route map: {missing}")

    processed_parts = []

    for video, group in detections.groupby("video", sort=False):
        video_anchors = anchors_checked[anchors_checked["video"] == video].sort_values("frame")
        if len(video_anchors) < 2:
            raise ValueError(f"No usable anchors for {video}")

        frame_values = group["frame"].to_numpy(dtype=float)
        group = group.copy()
        group["chainage_m"] = interpolate_chainage(
            frame_values,
            video_anchors["frame"].to_numpy(dtype=float),
            video_anchors["chainage_m"].to_numpy(dtype=float),
        )

        line = route_geometries[str(group["route_id"].iloc[0])]
        group["geometry"] = [
            line.interpolate(float(value)) if np.isfinite(value) else None
            for value in group["chainage_m"]
        ]
        processed_parts.append(group)

    detection_points = pd.concat(processed_parts, ignore_index=True)
    detection_points["box_area"] = (
        (detection_points["x2"] - detection_points["x1"])
        * (detection_points["y2"] - detection_points["y1"])
    )

    detection_gdf = gpd.GeoDataFrame(
        detection_points,
        geometry="geometry",
        crs=routes.crs,
    )
    detection_gdf.to_file(
        output / "detection_route_points.gpkg",
        layer="detection_route_points",
        driver="GPKG",
    )

    detection_csv = detection_points.drop(columns="geometry").copy()
    detection_csv["point_x"] = detection_gdf.geometry.x
    detection_csv["point_y"] = detection_gdf.geometry.y
    detection_csv.to_csv(output / "detection_route_points.csv", index=False)

    track_lookup = tracks.set_index(["video", "track_id"])

    observation_rows: list[dict[str, Any]] = []

    for (video, track_id), group in detection_points.groupby(["video", "track_id"]):
        mapped = group[group["chainage_m"].notna()].copy()
        if mapped.empty:
            continue

        summary = track_lookup.loc[(video, track_id)]
        representative = mapped.loc[mapped["box_area"].idxmax()]
        width = infer_video_width(group)
        normalized_center_x = float(representative["center_x"]) / width

        line = route_geometries[str(representative["route_id"])]
        rep_chainage = float(representative["chainage_m"])
        rep_point = line.interpolate(rep_chainage)

        observation_rows.append({
            "observation_id": f"{Path(video).stem}-TRK-{int(track_id):05d}",
            "video": video,
            "pass_id": str(representative["pass_id"]),
            "route_id": str(representative["route_id"]),
            "track_id": int(track_id),
            "class_name": str(summary["class_name"]),
            "representative_frame": int(representative["frame"]),
            "representative_time_s": float(representative.get("time_s", np.nan)),
            "representative_chainage_m": rep_chainage,
            "median_chainage_m": float(mapped["chainage_m"].median()),
            "minimum_chainage_m": float(mapped["chainage_m"].min()),
            "maximum_chainage_m": float(mapped["chainage_m"].max()),
            "chainage_span_m": float(mapped["chainage_m"].max() - mapped["chainage_m"].min()),
            "screen_side": screen_side(normalized_center_x),
            "normalized_center_x": normalized_center_x,
            "frames_seen": int(summary["frames_seen"]),
            "first_frame": int(summary["first_frame"]),
            "last_frame": int(summary["last_frame"]),
            "mean_confidence": float(summary["mean_confidence"]),
            "representative_confidence": float(representative["confidence"]),
            "representative_box_area": float(representative["box_area"]),
            "geometry": rep_point,
        })

    observations = gpd.GeoDataFrame(observation_rows, geometry="geometry", crs=routes.crs)
    if observations.empty:
        raise ValueError("No track observations could be mapped between anchor frames")

    eps_values = class_eps(args)
    observations["provisional_cluster"] = -1

    next_cluster = 0
    for (route_id, class_name), index in observations.groupby(
        ["route_id", "class_name"]
    ).groups.items():
        group = observations.loc[index]
        eps = eps_values.get(str(class_name), 25.0)

        model = DBSCAN(eps=eps, min_samples=1)
        local_labels = model.fit_predict(
            group[["representative_chainage_m"]].to_numpy()
        )

        label_map = {}
        for local_label in sorted(set(local_labels)):
            label_map[local_label] = next_cluster
            next_cluster += 1

        observations.loc[index, "provisional_cluster"] = [
            label_map[label] for label in local_labels
        ]

    class_prefixes = {
        "traffic_signs": "TS",
        "traffic_lights": "TL",
        "auth_label": "LM",
    }

    observations["candidate_object_id"] = ""

    for (route_id, class_name), group in observations.groupby(["route_id", "class_name"]):
        cluster_order = (
            group.groupby("provisional_cluster")["representative_chainage_m"]
            .median()
            .sort_values()
            .index
            .tolist()
        )
        prefix = class_prefixes.get(str(class_name), "OB")
        cluster_to_id = {
            cluster: f"{route_id}-{prefix}-{number:03d}"
            for number, cluster in enumerate(cluster_order, start=1)
        }
        observations.loc[group.index, "candidate_object_id"] = [
            cluster_to_id[value] for value in group["provisional_cluster"]
        ]

    candidate_summary = (
        observations.groupby(
            ["candidate_object_id", "route_id", "class_name"],
            as_index=False,
        )
        .agg(
            median_chainage_m=("representative_chainage_m", "median"),
            minimum_chainage_m=("representative_chainage_m", "min"),
            maximum_chainage_m=("representative_chainage_m", "max"),
            observations=("observation_id", "size"),
            passes_seen=("pass_id", "nunique"),
            videos_seen=("video", "nunique"),
            mean_confidence=("mean_confidence", "mean"),
            minimum_confidence=("mean_confidence", "min"),
            total_detected_frames=("frames_seen", "sum"),
        )
    )
    candidate_summary["chainage_spread_m"] = (
        candidate_summary["maximum_chainage_m"]
        - candidate_summary["minimum_chainage_m"]
    )
    candidate_summary["review_priority"] = np.select(
        [
            candidate_summary["passes_seen"] >= 2,
            (candidate_summary["passes_seen"] == 1)
            & (candidate_summary["mean_confidence"] >= 0.50)
            & (candidate_summary["total_detected_frames"] >= 10),
        ],
        ["MULTI_PASS_CANDIDATE", "SINGLE_PASS_REVIEW"],
        default="HIGH_FALSE_POSITIVE_PRIORITY",
    )

    observations.to_file(
        output / "track_observations.gpkg",
        layer="track_observations",
        driver="GPKG",
    )

    observations_csv = observations.drop(columns="geometry").copy()
    observations_csv["point_x"] = observations.geometry.x
    observations_csv["point_y"] = observations.geometry.y
    observations_csv["crs"] = routes.crs.to_string()
    observations_csv.to_csv(output / "track_observations.csv", index=False)

    candidate_summary.to_csv(
        output / "candidate_physical_objects.csv",
        index=False,
    )

    review = observations_csv[
        [
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
            "point_x",
            "point_y",
            "crs",
        ]
    ].copy()
    review["review_decision"] = ""
    review["corrected_class"] = ""
    review["false_positive_reason"] = ""
    review["manual_physical_object_id"] = ""
    review["review_notes"] = ""
    review.to_csv(output / "observation_review_template.csv", index=False)

    print(f"Mapped detection rows: {len(detection_points):,}")
    print(f"Mapped track observations: {len(observations):,}")
    print(f"Provisional physical-object candidates: {len(candidate_summary):,}")
    print(f"Results: {output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
