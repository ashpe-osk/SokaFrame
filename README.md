# SokaFrame Football Analysis

SokaFrame combines a football video-analysis pipeline with a FastAPI backend,
a React/Vite web app, and a background worker for uploaded-video analyses.
The command-line CV pipeline is at the repository root; the web application is
in `omniBoard/`.

## Clone and local setup

The repository intentionally does **not** include model weights, match videos,
local databases, or credentials. Obtain the required model files through an
approved private or licensed source and place them at the paths configured for
your setup. In particular, the React worker needs a compatible model at
`models/best.pt` unless `CV_DEFAULT_MODEL` points elsewhere.

For the Windows one-command React setup, install Node.js/npm, `uv`, Python
3.14 or newer for the backend, a separate Python 3.10 CV environment, and
FFmpeg/ffprobe. Follow [react_frontend.md](./react_frontend.md) for dependency
installation, private `.env` setup, database migrations, and the launcher:

```powershell
.\start-react.ps1
```

The launcher starts the API, analysis worker, and React/Vite frontend in
separate PowerShell windows. It expects the CV repository root,
Python 3.10 executable, and model path to be configured in
`omniBoard\.env`; it does not install dependencies or apply database
migrations.

For the standalone CV pipeline, see [onboarding.txt](./onboarding.txt) and
[Run and test](#run-and-test) below. Supply your own input video and model
weights; generated videos, uploads, and local data are excluded from Git.

### Repository layout

- `main.py`, `config/`, `utils/`, and `tests/`: football CV pipeline and tests.
- `omniBoard/backend/`: FastAPI API, database migrations, and worker.
- `omniBoard/frontend/`: React/TypeScript/Vite web application.
- `resources/`: SokaFrame brand assets.

The repository has no root license file. Confirm the licensing and
redistribution rights for the source code, brand assets, dependencies, and
separately distributed model weights before making the repository public.

---

## Football analysis pipeline

This project detects and tracks players, goalkeepers, referees, and the ball in a
football video. Its default output keeps the match footage unobscured and adds a
possession dashboard with a camera-space player radar below the frame. The
`main.py` entry point defaults to `models/best.pt` and
`data/vid_for_data_15s.mp4`, and writes the output to the next available
`output/run_N/` directory. Each run contains `tracked.mp4`, per-frame player
and ball observations in `observations.csv`, and per-frame rejected player
detections in `pitch_rejected_detections.csv`. Paths can be selected with
command-line arguments or environment variables, from either Windows or Linux.
Output finalization waits for up to five frames when needed so brief ball gaps
can be interpolated between actual detections before video, possession, and CSV
rows are written.

## Detection and player eligibility

The configured detector classes are ball, goalkeeper, player, and referee.
Detection confidence is `0.10`; only player and goalkeeper detections are
passed to the active tracker. Before BoT-SORT updates, the pipeline excludes
detections outside its fixed image-space playing-area polygon, detections
rejected by the optional grass mask, and player/GK boxes substantially
overlapping a confident referee box. Each rejected player/GK detection is
recorded with its frame, box, confidence, and rejection reason in
`pitch_rejected_detections.csv`. The ROI is not calibrated to pitch coordinates
and is not shot-aware. The visible-player threshold remains `0.60`. Raw
BoT-SORT IDs remain the tracker's internal IDs. A Global Identity Manager maps
eligible player/goalkeeper tracks to Global Player IDs for display and
downstream possession/radar outputs.

The active tracker is BoT-SORT, configured in
`config/custom_botsort.yaml`, with ReID and sparse-optical-flow GMC enabled.
Referee, ball, and other-class detections remain available to their existing
downstream processing, but are not sent to the player tracker and have no
player tracker ID.

These filters cannot identify an official whom the detector labels as a
player/goalkeeper when the box also passes the image-space pitch filters.
The `best.pt` classes do not include a linesman/staff category, so such
misclassifications require stronger detector output or a separately validated
official classifier; they should not be hidden by changing player colors.

## Team assignment and colors

Outfield team colors are fitted once from torso crops in LAB color space after
grass-like pixels are masked. Majority voting assigns and locks each track's
label. The initial team label requires six consistent observations once the
team centers are fitted and at least 75% vote consensus; player markers can
remain neutral during that brief warm-up. Team history remains keyed to the
active tracker ID.
Goalkeepers are assigned separately from outfield players and need three
consistent color matches to join a team; their existing goalkeeper visual
styling is retained. Referees do not receive a team assignment or Global
Player ID.

## Persistent player identity

`utils/global_identity_manager.py` keeps identity profiles, each with a bounded
eight-position trajectory, a smoothed BoT-SORT ReID feature
when available, a lightweight torso-color histogram fallback, team label,
bounding-box scale, and last-seen frame. A changed raw tracker ID is compared
against dormant profiles using appearance, team consistency, camera-stabilized
position/velocity, box scale, and elapsed time. Known opposing-team labels
reject a match; unknown team labels are neutral. Ambiguous or weak matches
create a new Global Player ID rather than force an unsafe merge. If a
high-confidence BoT-SORT ReID match is clearly stronger than other retained
profiles, position and motion can be treated as inconclusive; ambiguous
matches against multiple profiles are still rejected. The
re-identification window defaults to 54,000 frames (30 minutes at 30 FPS).
`--canonical-ids off` preserves the legacy 64-profile limit and time window;
`online` mode retains historical profiles without those limits, while `offline`
mode emits conservative unresolved tracklet evidence rather than claiming
unverified merges. The opt-in `--team-smoothing` mode requires online
canonical IDs and uses a 15-observation, confidence-weighted color-evidence
window. Its relative color-distance margin is evidence weighting, not a
calibrated probability.

The manager only receives players/goalkeepers that already passed the
pre-tracker playing-area/referee filters and post-tracker visibility checks.
Referee, ball, outside-area, and other detections cannot receive player Global
IDs. Raw IDs are retained on each `PlayerObservation` for diagnostics; the
Global Player ID is used for ellipse labels, radar labels, and possession
player identity. Pitch coordinates are currently unavailable, so pitch
matching is enabled only when a caller explicitly provides a reliable pitch
position.

`utils/visualization_utils.py`, the dashboard, and the radar use the shared
high-contrast display palette: Team 1, Team 2, goalkeeper, and ball each have a
consistent color. Ellipses have a dark outline to remain visible against the
grass.

## Dashboard and radar

The video frame is padded with a fixed-height panel beneath the footage. The
panel shows team-only possession bars (Team 1 + Team 2 = 100%), with Loose and
Unknown listed separately, plus the color legend and a fixed-size OpenCV radar.
No Matplotlib rendering or per-frame `plt.show()` is used.

The radar is labeled **CAMERA VIEW - NOT PITCH CALIBRATED**. It plots eligible
player foot points and the ball in normalized image space. Increasing source
image y maps downward on the panel, placing the near touchline at the bottom.
It displays only an outer boundary and halfway line; it does not claim to show
calibrated pitch coordinates. The active footage does not yet have validated
pitch calibration, so normalized observations set `pitch_x` and `pitch_y` to
`None` rather than fabricating coordinates. The existing homography utility is
not enabled in the active pipeline.

## Possession

Possession follows the existing distance-based logic in
`utils/possession_utils.py`, but consumes normalized observations keyed by
the persistent Global Player ID (falling back to raw tracker ID for legacy
callers).
Referees, off-pitch detections, and referee-duplicate tracks are excluded.
The estimate uses a 70-pixel nearest-player threshold, four-frame confirmation,
and retains the current possessor while within 1.25 times that distance.
Short ball gaps of up to five frames are linearly interpolated only when
bounded by real detections on both sides; leading, trailing, and longer gaps
remain missing. `observations.csv` reports ball sources as `real`,
`interpolated`, or `missing`; interpolated positions have no detector
confidence. The opt-in `--ball-gate` mode requires direct `real` ball detections
for possession and therefore excludes interpolated positions.
Dashboard team percentages use only Team 1 and Team 2 possession frames as the
denominator; Loose and Unknown are reported separately.

Each frame's player and ball data is represented by `PlayerObservation` and
`BallObservation` in `utils/player_observations.py`. Player observations carry
frame/timestamp, the current tracker ID and persistent Global Player ID, class,
team, image and foot position,
confidence, visibility, optional pitch coordinates, and current possession
state. The active loop sends these same observations to possession, radar, and
rendering; it does not retain the entire video or yet export the records.
`track.py` consumes precomputed team-control arrays for its cumulative
percentages; it does not implement the active possession decision. This MVP
does not infer passes, formations, or field-space control, and possession
remains an image-space distance estimate.

## Current limitations and longer clips

The `best.pt` model currently reports `ball`, `goalkeeper`, `player`, and
`referee`; all four names map to their intended semantic categories. Unknown
detector classes intentionally map to `other` and are not admitted to player
display or possession. Player-class detections are filtered by a broad
playing-area polygon and an adaptive grass mask, while referee classes and
player boxes overlapping confident referee boxes are excluded. These are
image-space filters, not a guarantee against every staff member or substitute
detected on or near the pitch.

Team labels are assigned from the fitted team-color centers and torso LAB
features. The distance decision uses all three LAB channels; six observations
with at least 75% agreement confirm a track's color, after which its ring no
longer stays neutral during the longer smoothing window.
The tracker can still switch/reuse IDs, so resulting ID/team switches must be
measured from tracker output and are not hidden by a second identity namespace.

The analysis loop reads, annotates, and writes one frame at a time and has no
fixed ten-minute duration limit. Per-frame debug logging is off by default.
The 449-frame, 15-second validation took about 7 minutes 34 seconds (roughly
1 frame per second) in this environment; at that rate a ten-minute clip would
take about five hours. Referee histories expire after the tracker buffer
interval, and tracker appearance profiles are pruned when ByteTrack removes a
track.
Long-clip throughput and output storage still need optimization and benchmarking
before production use. This is currently a command-line prototype using the
configured video path, not yet an
upload/download web service; downloadable match reports and pass exports in
JSON or CSV are not implemented.

## Run and test

Relative paths are resolved from the repository root, regardless of the
current working directory. The default remains the 15-second video:

```powershell
.\.venv\Scripts\python.exe main.py
```

Select either test clip and configure the output directory:

```powershell
.\.venv\Scripts\python.exe main.py --video data/vid_for_data_15s.mp4 --model models/best.pt --output-dir output
.\.venv\Scripts\python.exe main.py --video data/vid_for_data_long.mp4 --model models/best.pt --output-dir output
```

Set `--device 0` to explicitly request the first CUDA GPU; without `--device`,
The default `--backend auto` selects CUDA when the installed PyTorch can use an
NVIDIA GPU, otherwise OpenVINO when installed, otherwise PyTorch CPU. Before
using CUDA, the worker runs one small inference using the selected model and a
frame from the input video. Automatic mode reports a failed CUDA probe and
tries the next backend; explicit `--backend cuda` reports the error instead of
silently falling back. The selection applies to the machine running this
worker/backend, not the browser computer. An RTX 5060 therefore needs an NVIDIA
driver and a Blackwell-compatible CUDA-enabled PyTorch build (CUDA 12.8 or
newer) on the worker machine. For Intel CPU inference, install the
optional OpenVINO dependency:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-openvino.txt
```

The first OpenVINO run exports the selected `.pt` model to a neighboring
`*_openvino_model` directory and reuses that export on later runs. Choose a
backend explicitly with `--backend cuda`, `--backend openvino`, or
`--backend cpu`; `--device cpu` keeps CPU inference while still allowing the
OpenVINO preference. Environment overrides are `FOOTBALL_VIDEO`,
`FOOTBALL_MODEL`, `FOOTBALL_OUTPUT_DIR`, `FOOTBALL_DEVICE`, `FOOTBALL_BACKEND`,
`FOOTBALL_IMGSZ`, and `FOOTBALL_PROGRESS_EVERY`. `--progress-every 100`
controls progress reports, while `--progress-every 0` disables them. Inference
defaults to `--imgsz 384`. In a 200-frame benchmark, 384 processed at 3.31 FPS
versus 2.87 FPS at 512; detections fell from 2,833 to 2,651, so check small or
distant players and the ball in the resulting video. Use `--imgsz 512` or 640
when detection detail is more important than speed.
Use `--max-frames 60` for a bounded integration smoke test; the default `0`
processes the complete selected video.

On supported Intel hardware, OpenVINO can target the integrated GPU explicitly:

```powershell
.\.venv\Scripts\python.exe main.py --video data/vid_for_data_15s.mp4 --backend openvino --device intel:gpu
```

This target is validated against the OpenVINO devices available on the worker.
For the OmniBoard worker, set `CV_DEFAULT_DEVICE=intel:gpu` to opt in; the
default remains `cpu`.

### Phase 1 tracker diagnostics and replay

Capture the tracker output once, then replay it while iterating on downstream
team, identity, and possession logic:

```powershell
.\.venv\Scripts\python.exe main.py --video data/vid_for_data_15s.mp4 --cache-stub
.\.venv\Scripts\python.exe main.py --video data/vid_for_data_15s.mp4 --use-stub
```

The default cache is `output/run_N/tracking_stub.pkl`; use `--stub-path`
to choose another location. `--use-stub` still reads the source video for
pixels, but skips model loading, YOLO inference, and BoT-SORT updates. When
cache capture or replay is enabled, the diagnostic artifacts are written under
`output/run_N/phase1_diagnostics/`. Use the same source video and
dimensions for replay. A cache made with `--max-frames` contains only that many
frames and should be replayed with the same frame limit.

For Colab, open `notebooks/colab_setup_and_run.ipynb`, select a GPU runtime,
install the listed dependencies, provide `models/best.pt` and a video (locally
in `/content` or on mounted Drive), then run the 15-second GPU benchmark before
manually confirming the long-video run. The notebook does not automatically
start the long clip. Tracker updates remain sequential and
`config/custom_botsort.yaml` is unchanged.

Run the tracking and visualization tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_tracking_visualization tests.test_possession_utils tests.test_player_observations tests.test_ball_tracking
```
