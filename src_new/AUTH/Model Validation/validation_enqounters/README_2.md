# YOLO video encounter counter

## Definition

One encounter is one valid tracker ID within one video.

This does not count unique physical objects across different videos.

## Install

```bash
python -m pip install -r requirements.txt
```

## Folder example

```text
project/
├── count_video_encounters.py
├── requirements.txt
├── best.pt
└── videos/
    ├── route_01.MOV
    ├── route_02.MOV
    └── route_03.MOV
```

## Run

```bash
python count_video_encounters.py \
  --model best.pt \
  --videos videos \
  --output tracking_results \
  --save-annotated
```

For an Apple Silicon Mac, optionally add:

```bash
--device mps
```

For the first NVIDIA GPU, optionally add:

```bash
--device 0
```

## Main outputs

- `raw_detections.csv`: every tracked box in every frame.
- `track_summary_all.csv`: one row per tracker ID, including rejected tracks.
- `track_summary_valid.csv`: tracks that passed the filtering rules.
- `encounter_counts_by_video.csv`: encounter totals by video and class.
- `encounter_counts_total.csv`: encounter totals across all processed videos.
- `annotated_videos/`: visual checking videos, when `--save-annotated` is used.

## Important validation

Watch several annotated videos and manually compare counts. Tune:

- `--conf`
- `--min-frames`
- `--min-mean-conf`
- `--min-class-agreement`
- `--imgsz`

Do not interpret the total as a count of unique physical signs across routes.
