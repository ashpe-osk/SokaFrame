import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from time import perf_counter, sleep

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from utils.ball_tracking import BallPositionTracker
from utils.camera_radar import CameraRadarRenderer
from utils.camera_motion import CameraMotionEstimator, CameraReference
from utils.dashboard import LAYOUT, compose_dashboard, output_size
from utils.detection_classes import map_class_categories
from utils.pitch_region import is_inside_playing_area
from utils.player_observations import BallObservation, PlayerObservation
from utils.global_identity_manager import GlobalIdentityManager
from utils.final_team_reconciliation import reconcile_final_team_labels
from utils.match_analytics import write_match_analytics
from utils.pass_detection import EvidenceBasedPassDetector
from utils.possession_utils import PossessionEstimator
from utils.referee_tracking import RefereeTrackManager
from utils.run_directories import create_next_run_directory
from utils.team_color_utils import (
    CanonicalTeamConflictTracker,
    CanonicalTeamSmoother,
    TeamColorAssigner,
)
from utils.tracking_input_filter import PlayerTrackingInputFilter
from utils.visualization_utils import draw_annotations, is_displayable_player
from utils.phase1_analysis import (
    Phase1Analyzer,
    TrackingStubReader,
    TrackingStubWriter,
    camera_motion_from_dict,
)
from utils.pipeline_checkpoint import (
    load_checkpoint,
    restore_checkpointed_file,
    save_checkpoint,
)
from utils.offline_identity_reconciliation import reconcile_observations

PROJECT_ROOT = Path(__file__).resolve().parent
_PROGRESS_REPLACE_RETRY_SECONDS = 0.1
_PROGRESS_REPLACE_RETRY_ATTEMPTS = 10
_PIPELINE_CHECKPOINT_INTERVAL = 100
OBSERVATION_CSV_FIELDS = (
    "frame_number",
    "timestamp_seconds",
    "object_type",
    "tracker_id",  
    "global_id",
    "player_type",
    "image_x",
    "image_y",
    "foot_x",
    "foot_y",
    "pitch_x",
    "pitch_y",
    "confidence",
    "is_visible",
    "bbox",
    "team_color",
    "possession_state",
    "possession_player_id",
    "source",
    "is_valid",
)
IDENTITY_EVIDENCE_CSV_FIELDS = (
    "frame_number",
    "current_tracker_id",
    "current_tracklet_ref",
    "candidate_global_id",
    "candidate_tracker_id",
    "candidate_tracklet_ref",
    "candidate_last_seen_frame",
    "gap_frames",
    "motion_score",
    "appearance_reid_score",
    "team_agreement",
    "team_conflict",
    "scale_consistency_score",
    "time_gap_score",
    "pitch_space_score",
    "distance_score",
    "combined_score",
    "best_candidate_score",
    "runner_up_score",
    "score_margin",
    "decision",
    "reason",
    "constraint",
    "algorithm_version",
)
PITCH_REJECTION_CSV_FIELDS = (
    "frame_number",
    "timestamp_seconds",
    "class_id",
    "object_type",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "rejection_reason",
)
MODEL_PATH = os.environ.get("FOOTBALL_MODEL", "models/best.pt")
VIDEO_PATH = os.environ.get("FOOTBALL_VIDEO", "data/vid_for_data_15s.mp4")
OUTPUT_DIR = os.environ.get("FOOTBALL_OUTPUT_DIR", "output")
TRACKER_CONFIG = "config/custom_botsort.yaml"
RADAR_MODE = "camera"
LAYOUT = "panel"
DETECTION_CONFIDENCE = 0.10
PLAYER_CONFIDENCE_THRESHOLD = 0.60
REFEREE_DUPLICATE_CONFIDENCE = 0.30
REFEREE_DUPLICATE_IOU = 0.50
POSSESSION_DISTANCE_THRESHOLD = 70.0
DEBUG_DETECTIONS = False
DEBUG_POSSESSION = False
ENABLE_OTHER_BUCKET = True
ENABLE_PITCH_MASK = True
IDENTITY_DIAGNOSTICS = (
    str(os.environ.get("IDENTITY_DIAGNOSTICS", "false")).strip().lower()
    == "true"
)


def _resolve_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _resolve_inference_backend(backend, device):
    import torch

    cuda_available = torch.cuda.is_available()
    openvino_available = importlib.util.find_spec("openvino") is not None
    requested_device = str(device).lower() if device is not None else None

    if backend == "auto":
        if requested_device == "cpu":
            backend = "openvino" if openvino_available else "cpu"
        elif requested_device is not None and requested_device.startswith("intel:"):
            backend = "openvino"
        elif requested_device is not None:
            backend = "cuda"
        elif cuda_available:
            backend = "cuda"
        elif openvino_available:
            backend = "openvino"
        else:
            backend = "cpu"

    if backend == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "CUDA was requested but is unavailable in this PyTorch "
                "environment. Install a CUDA-enabled PyTorch build and verify "
                "that the NVIDIA driver is available."
            )
        device_count = torch.cuda.device_count()
        if device_count <= 0:
            raise RuntimeError(
                "PyTorch reports CUDA as available, but it reports no NVIDIA "
                "devices. Check the driver and CUDA-enabled PyTorch install."
            )
        if requested_device == "cpu":
            raise ValueError("--device cpu cannot be used with --backend cuda")
        if requested_device in (None, "cuda"):
            return backend, 0
        device_index = (
            requested_device.removeprefix("cuda:")
            if requested_device.startswith("cuda:")
            else requested_device
        )
        if not device_index.isdigit():
            raise ValueError(
                "CUDA --device must be an index such as 0 or cuda:0"
            )
        device_index_value = int(device_index)
        if device_index_value >= device_count:
            raise ValueError(
                f"CUDA device index {device_index_value} is unavailable; "
                f"PyTorch detected {device_count} device(s)."
            )
        return backend, device_index_value

    if backend == "openvino":
        if not openvino_available:
            raise RuntimeError(
                "OpenVINO was requested but is not installed. Install it with "
                "'.venv\\Scripts\\python.exe -m pip install -r "
                "requirements-openvino.txt'."
            )
        if requested_device in (None, "cpu"):
            return backend, "cpu"
        if not requested_device.startswith("intel:"):
            raise ValueError(
                "OpenVINO devices must use 'cpu' or an Intel target such as "
                "'intel:gpu'."
            )
        openvino_device = requested_device.split(":", maxsplit=1)[1].upper()
        if not openvino_device:
            raise ValueError("OpenVINO Intel device target must not be empty.")
        from openvino import Core

        available_devices = Core().available_devices
        if not any(
            name == openvino_device
            or name.startswith(f"{openvino_device}.")
            for name in available_devices
        ):
            raise RuntimeError(
                f"OpenVINO device '{openvino_device}' is unavailable. "
                f"Available devices: {', '.join(available_devices)}"
            )
        return backend, f"intel:{openvino_device.lower()}"

    if backend == "cpu":
        if requested_device not in (None, "cpu"):
            raise ValueError(
                "PyTorch CPU inference cannot use a non-CPU --device"
            )
        return backend, "cpu"

    raise ValueError(f"Unsupported inference backend: {backend}")


def _load_inference_model(model_path, backend, device, imgsz):
    resolved_backend, inference_device = _resolve_inference_backend(
        backend, device
    )
    print(f"Inference backend: {resolved_backend} (device {inference_device})")

    if resolved_backend == "openvino":
        openvino_directory = model_path.with_name(
            f"{model_path.stem}_openvino_model"
        )
        openvino_xml = openvino_directory / model_path.with_suffix(".xml").name
        metadata_path = openvino_directory / "metadata.yaml"
        cached_metadata = (
            yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else None
        )
        cache_matches_model = (
            openvino_xml.is_file()
            and isinstance(cached_metadata, dict)
            and cached_metadata.get("imgsz") == [imgsz, imgsz]
            and model_path.stat().st_mtime <= openvino_xml.stat().st_mtime
        )
        if not cache_matches_model:
            print(f"Exporting model to OpenVINO: {openvino_directory}")
            exported_path = YOLO(str(model_path)).export(
                format="openvino",
                imgsz=imgsz,
                device="cpu",
            )
            openvino_directory = Path(exported_path)
            if not openvino_directory.is_absolute():
                openvino_directory = (Path.cwd() / openvino_directory).resolve()
            openvino_xml = (
                openvino_directory / model_path.with_suffix(".xml").name
            )
            if not openvino_xml.is_file():
                raise FileNotFoundError(
                    "OpenVINO export completed without the expected model "
                    f"file: {openvino_xml}"
                )
            metadata_path = openvino_directory / "metadata.yaml"
            cached_metadata = (
                yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
                if metadata_path.is_file()
                else None
            )
            if (
                not isinstance(cached_metadata, dict)
                or cached_metadata.get("imgsz") != [imgsz, imgsz]
            ):
                raise ValueError(
                    "OpenVINO export metadata does not match the requested "
                    f"inference size ({imgsz}): {metadata_path}"
                )
        model = YOLO(str(openvino_directory))
    else:
        model = YOLO(str(model_path))

    return model, inference_device, resolved_backend


def _load_auto_fallback(model_path, imgsz):
    if importlib.util.find_spec("openvino") is not None:
        try:
            return _load_inference_model(
                model_path, "openvino", None, imgsz
            )
        except (
            AssertionError,
            ImportError,
            OSError,
            RuntimeError,
            ValueError,
        ) as error:
            print(
                "OpenVINO could not be initialized; falling back to "
                f"PyTorch CPU: {error}"
            )
    return _load_inference_model(model_path, "cpu", None, imgsz)


def _read_cuda_probe_frame(video_path):
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise OSError(f"Could not open video for CUDA probe: {video_path}")
        success, frame = capture.read()
        if not success:
            raise OSError(f"Could not read a CUDA probe frame: {video_path}")
        return frame
    finally:
        capture.release()


def _probe_cuda_inference(model, frame, imgsz, device=0):
    import torch

    device_name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    print(
        f"CUDA device: {device_name}; compute capability: {capability}; "
        f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}"
    )
    try:
        model.predict(
            source=frame,
            device=device,
            imgsz=min(imgsz, 320),
            verbose=False,
            save=False,
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"CUDA inference probe failed on {device_name} "
            f"(capability {capability}, CUDA build {torch.version.cuda}): "
            f"{error}"
        ) from error
    print("CUDA inference probe succeeded.")

def _write_progress_sidecar(path, frames_processed, total_frames, status):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "status": status,
        "frames_processed": frames_processed,
        "total_frames": total_frames if total_frames > 0 else None,
    }
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, separators=(",", ":"))
        for attempt in range(_PROGRESS_REPLACE_RETRY_ATTEMPTS):
            try:
                os.replace(temporary_path, path)
                break
            except PermissionError:
                if attempt + 1 == _PROGRESS_REPLACE_RETRY_ATTEMPTS:
                    raise
                sleep(_PROGRESS_REPLACE_RETRY_SECONDS)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Track and annotate a football video."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit for a bounded test run; 0 processes the full video.",
    )
    parser.add_argument(
        "--canonical-ids",
        choices=("offline", "online", "off"),
        default="off",
        help=(
            "Identity strategy: off preserves legacy capped online IDs; "
            "online retains full-video identity state; offline writes "
            "conservative tracklet sidecars and renders unresolved IDs."
        ),
    )
    parser.add_argument(
        "--ball-gate",
        action="store_true",
        help=(
            "Conservatively gate possession to directly detected balls and "
            "known-team players; requires --canonical-ids online."
        ),
    )
    parser.add_argument(
        "--team-smoothing",
        action="store_true",
        help=(
            "Smooth color-evidence team labels over canonical identities; "
            "requires --canonical-ids online."
        ),
    )
    parser.add_argument(
        "--video",
        default=VIDEO_PATH,
        help="Input video path (default: FOOTBALL_VIDEO or the 15-second clip).",
    )
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help="YOLO model path (default: FOOTBALL_MODEL or models/best.pt).",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory for generated runs (default: FOOTBALL_OUTPUT_DIR or output/).",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("FOOTBALL_DEVICE"),
        help=(
            "Inference device, e.g. 0, cpu, or intel:gpu "
            "(default: chosen by --backend)."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "cuda", "openvino", "cpu"),
        default=os.environ.get("FOOTBALL_BACKEND", "auto").lower(),
        help=(
            "Inference runtime: auto prefers CUDA, then OpenVINO, then "
            "PyTorch CPU (default: FOOTBALL_BACKEND or auto)."
        ),
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=int(os.environ.get("FOOTBALL_IMGSZ", "384")),
        help=(
            "YOLO inference size in pixels; lower values are faster but may "
            "miss small objects (default: FOOTBALL_IMGSZ or 384)."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=int(os.environ.get("FOOTBALL_PROGRESS_EVERY", "100")),
        help="Print progress every N frames; use 0 to disable (default: 100).",
    )
    parser.add_argument(
        "--progress-file",
        default=None,
        help="Optional versioned JSON progress sidecar path for worker integration.",
    )
    parser.add_argument(
        "--profile-stages",
        action="store_true",
        help="Enable temporary per-stage timing for a profiling run.",
    )
    parser.add_argument(
        "--cache-stub",
        action="store_true",
        help="Save post-tracker detections and camera motion for later replay.",
    )
    parser.add_argument(
        "--use-stub",
        action="store_true",
        help="Replay a saved tracking stub, skipping YOLO and BoT-SORT.",
    )
    parser.add_argument(
        "--stub-path",
        default=None,
        help="Tracking-stub pickle path (default: output/<video>/tracking_stub.pkl).",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Optional path for automatic pipeline checkpoint and resume.",
    )
    parser.add_argument(
        "--ffmpeg-executable",
        default=os.environ.get("FFMPEG_EXECUTABLE", "ffmpeg"),
        help="FFmpeg executable used to assemble resumed video segments.",
    )
    args = parser.parse_args(argv)
    if args.progress_every < 0:
        parser.error("--progress-every must be zero or greater")
    if args.max_frames < 0:
        parser.error("--max-frames must be zero or greater")
    if args.imgsz <= 0:
        parser.error("--imgsz must be greater than zero")
    if args.ball_gate and args.canonical_ids != "online":
        parser.error("--ball-gate requires --canonical-ids online")
    if args.team_smoothing and args.canonical_ids != "online":
        parser.error("--team-smoothing requires --canonical-ids online")
    if (
        args.backend != "auto"
        and args.device is not None
        and (
            (args.backend == "cuda" and str(args.device).lower() == "cpu")
            or (
                args.backend in ("openvino", "cpu")
                and str(args.device).lower() != "cpu"
                and not (
                    args.backend == "openvino"
                    and str(args.device).lower().startswith("intel:")
                )
            )
        )
    ):
        parser.error("--device conflicts with the selected --backend")
    if args.cache_stub and args.use_stub:
        parser.error("--cache-stub and --use-stub are separate runs.")
    if args.resume_checkpoint and (args.use_stub or args.cache_stub):
        parser.error(
            "--resume-checkpoint cannot be combined with tracking-stub modes."
        )
    return args


def _class_label(class_names, class_id):
    if isinstance(class_names, dict):
        return str(class_names.get(class_id, class_id))
    if 0 <= class_id < len(class_names):
        return str(class_names[class_id])
    return str(class_id)


def _open_video_writer(path, fps, size):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    if not writer.isOpened():
        raise OSError(f"Could not initialize video writer: {path}")
    return writer


def _assemble_video_segments(segments, output_path, ffmpeg_executable):
    if not segments:
        raise OSError("No completed video segments are available to assemble.")
    if len(segments) == 1:
        shutil.copyfile(segments[0], output_path)
        return

    list_path = segments[0].parent / "concat-list.txt"
    temporary_output = output_path.with_name("tracked.tmp.mp4")
    try:
        list_path.write_text(
            "".join(f"file '{segment.name}'\n" for segment in segments),
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                ffmpeg_executable,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                str(temporary_output),
            ],
            cwd=segments[0].parent,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise OSError(
                "FFmpeg could not assemble resumed video segments: "
                f"{result.stderr.strip() or result.returncode}"
            )
        if not temporary_output.is_file() or temporary_output.stat().st_size <= 0:
            raise OSError("FFmpeg produced an empty resumed video.")
        os.replace(temporary_output, output_path)
    finally:
        list_path.unlink(missing_ok=True)
        temporary_output.unlink(missing_ok=True)


def _tracker_checkpoint_state(tracker):
    state = tracker.__dict__.copy()
    state.pop("encoder", None)
    return {
        "attributes": state,
        "next_track_id": tracker.track_class._count,
    }


def _restore_tracker_checkpoint_state(tracker, state):
    encoder = getattr(tracker, "encoder", None)
    tracker.__dict__.update(state["attributes"])
    if hasattr(tracker, "encoder"):
        tracker.encoder = encoder
    tracker.track_class._count = state["next_track_id"]


class _StageProfiler:
    def __init__(self):
        self.totals = defaultdict(float)
        self.samples = defaultdict(list)

    def record(self, stage, seconds):
        self.totals[stage] += seconds
        self.samples[stage].append(seconds)

    def report(self, frame_count, total_frames, elapsed, prefix="Profile"):
        processing_fps = frame_count / elapsed if elapsed else 0.0
        print(
            f"{prefix}: Frames {frame_count} / {total_frames} | "
            f"Overall FPS: {processing_fps:.2f}"
        )
        for stage in (
            "Model warm-up",
            "Video frame read",
            "Tracking cache read",
            "Camera motion compensation",
            "YOLO preprocessing",
            "YOLO inference",
            "YOLO postprocessing",
            "Pre-tracker filtering",
            "BoT-SORT update",
            "Predictor / callback overhead",
            "Tracker-ID / lifecycle bookkeeping",
            "Post-tracker referee / eligibility filtering",
            "Team classification",
            "Global identity assignment + appearance",
            "Ball tracking",
            "Player and ball observations",
            "Radar rendering",
            "Possession calculation",
            "Frame annotation + dashboard",
            "Observations CSV writing",
            "Phase 1 diagnostics",
            "Video writing",
            "Output finalization",
            "Other loop / reporting overhead",
        ):
            total_seconds = self.totals[stage]
            average_ms = total_seconds * 1000.0 / frame_count
            percentage = (
                100.0 * total_seconds / elapsed if elapsed else 0.0
            )
            print(
                f"  {stage}: {average_ms:.2f} ms/frame "
                f"({percentage:.1f}% of processing wall time)"
            )


def _install_tracker_profile_hooks(profiler, synchronize_cuda):
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.trackers.utils.gmc import GMC

    original_methods = []

    def install(target, method_name, stage, synchronize=False):
        original = getattr(target, method_name)

        def timed_method(self, *args, **kwargs):
            if synchronize:
                synchronize_cuda()
            started = perf_counter()
            try:
                return original(self, *args, **kwargs)
            finally:
                if synchronize:
                    synchronize_cuda()
                profiler.record(stage, perf_counter() - started)

        setattr(target, method_name, timed_method)
        original_methods.append((target, method_name, original))

    install(BOTSORT, "update", "BoT-SORT update", synchronize=True)
    install(
        BOTSORT,
        "init_track",
        "ReID feature extraction + track initialization",
        synchronize=False,
    )
    install(
        BOTSORT,
        "get_dists",
        "ReID / appearance association costs",
        synchronize=False,
    )
    install(GMC, "apply", "GMC optical-flow estimation")
    return original_methods


def _identity_transition_events(
    previous_global_by_tracker,
    previous_tracker_by_global,
    current_global_by_tracker,
    frame,
    camera_epoch=None,
    timestamp=None,
):
    events = []
    current_tracker_by_global = {}
    for tracker_id, global_id in current_global_by_tracker.items():
        current_tracker_by_global[int(global_id)] = int(tracker_id)

    for tracker_id in sorted(
        set(previous_global_by_tracker) | set(current_global_by_tracker)
    ):
        previous_global = previous_global_by_tracker.get(tracker_id)
        current_global = current_global_by_tracker.get(tracker_id)
        if previous_global is None and current_global is not None:
            events.append(
                {
                    "event": "new_tracker",
                    "frame": frame,
                    "tracker_id": tracker_id,
                    "global_id": current_global,
                    "camera_epoch": camera_epoch,
                    "timestamp": timestamp,
                }
            )
        elif (
            previous_global is not None
            and current_global is not None
            and previous_global != current_global
        ):
            events.append(
                {
                    "event": "tracker_global_changed",
                    "frame": frame,
                    "tracker_id": tracker_id,
                    "previous_global_id": previous_global,
                    "global_id": current_global,
                    "camera_epoch": camera_epoch,
                    "timestamp": timestamp,
                }
            )
    for global_id in sorted(
        set(previous_tracker_by_global) | set(current_tracker_by_global)
    ):
        previous_tracker = previous_tracker_by_global.get(global_id)
        current_tracker = current_tracker_by_global.get(global_id)
        if previous_tracker is None and current_tracker is not None:
            events.append(
                {
                    "event": "new_global",
                    "frame": frame,
                    "tracker_id": current_tracker,
                    "global_id": global_id,
                    "camera_epoch": camera_epoch,
                    "timestamp": timestamp,
                }
            )
        elif (
            previous_tracker is not None
            and current_tracker is not None
            and previous_tracker != current_tracker
        ):
            events.append(
                {
                    "event": "global_tracker_changed",
                    "frame": frame,
                    "global_id": global_id,
                    "previous_tracker_id": previous_tracker,
                    "tracker_id": current_tracker,
                    "camera_epoch": camera_epoch,
                    "timestamp": timestamp,
                }
            )
    return events


def _select_possession_ball_position(position, source, ball_gate):
    if ball_gate and source != "real":
        return None
    return position


def main(argv=None):
    args = _parse_args(argv)
    video_path = _resolve_path(args.video)
    model_path = _resolve_path(args.model)
    output_root = _resolve_path(args.output_dir)
    progress_path = _resolve_path(args.progress_file) if args.progress_file else None
    tracker_config = _resolve_path(TRACKER_CONFIG)
    checkpoint_path = (
        _resolve_path(args.resume_checkpoint)
        if args.resume_checkpoint
        else None
    )
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")
    if not args.use_stub and not model_path.is_file():
        raise FileNotFoundError(f"YOLO model does not exist: {model_path}")
    if not args.use_stub and not tracker_config.is_file():
        raise FileNotFoundError(f"Tracker configuration does not exist: {tracker_config}")

    output_root.mkdir(parents=True, exist_ok=True)
    run_output_dir = create_next_run_directory(output_root)
    output_path = run_output_dir / "tracked.mp4"
    resume_root = (
        checkpoint_path.parent
        if checkpoint_path is not None
        else run_output_dir
    )
    segments_dir = resume_root / "segments"
    resume_observations_path = resume_root / "observations.csv"
    identity_transition_path = resume_root / "identity_transitions.jsonl"
    identity_evidence_path = (
        resume_root / "identity_association_evidence.csv"
    )
    pitch_rejections_path = resume_root / "pitch_rejected_detections.csv"
    segments_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_data = None
    default_stub_path = run_output_dir / "tracking_stub.pkl"
    stub_path = (
        _resolve_path(args.stub_path)
        if args.stub_path
        else default_stub_path
    )
    if RADAR_MODE != "camera":
        raise ValueError("This pipeline uses camera-space radar; pitch calibration is disabled.")

    model = None
    model_initialization_seconds = 0.0
    class_names = None
    if args.use_stub:
        print(f"Loading tracking stub: {stub_path}")
    else:
        print("Initializing player, ball/referee models...")
        model_initialization_started = perf_counter()
        candidate_backend, _ = _resolve_inference_backend(
            args.backend, args.device
        )
        try:
            model, inference_device, inference_backend = (
                _load_inference_model(
                    model_path,
                    args.backend,
                    args.device,
                    args.imgsz,
                )
            )
        except (
            AssertionError,
            ImportError,
            OSError,
            RuntimeError,
            ValueError,
        ) as error:
            if args.backend != "auto" or candidate_backend != "openvino":
                raise
            print(
                "OpenVINO initialization failed in automatic mode; "
                f"falling back to PyTorch CPU: {error}"
            )
            model, inference_device, inference_backend = (
                _load_inference_model(
                    model_path,
                    "cpu",
                    None,
                    args.imgsz,
                )
            )
        if inference_backend == "cuda":
            try:
                _probe_cuda_inference(
                    model,
                    _read_cuda_probe_frame(video_path),
                    args.imgsz,
                    inference_device,
                )
            except RuntimeError as error:
                if args.backend != "auto" or args.device is not None:
                    raise
                print(
                    "CUDA inference probe failed in automatic mode; "
                    f"trying the next backend: {error}"
                )
                model, inference_device, inference_backend = (
                    _load_auto_fallback(model_path, args.imgsz)
                )
        model_initialization_seconds = perf_counter() - model_initialization_started
        class_names = model.names
        if args.profile_stages and model is not None:
            print(
                "Model initialization time: "
                f"{model_initialization_seconds:.3f} seconds"
            )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_width <= 0 or frame_height <= 0 or not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError(f"Invalid video properties for {video_path}")
    checkpoint_fingerprint = None
    if checkpoint_path is not None:
        video_stat = video_path.stat()
        model_stat = model_path.stat() if not args.use_stub else None
        checkpoint_fingerprint = {
            "video": str(video_path),
            "video_size": video_stat.st_size,
            "video_mtime_ns": video_stat.st_mtime_ns,
            "model": str(model_path),
            "model_size": model_stat.st_size if model_stat else None,
            "model_mtime_ns": model_stat.st_mtime_ns if model_stat else None,
            "frame_size": (frame_width, frame_height),
            "fps": fps,
            "total_frames": total_frames,
            "backend": args.backend,
            "device": args.device,
            "imgsz": args.imgsz,
            "tracker_config": str(tracker_config),
            "ball_interpolation": "linear-buffered-v1",
            "team_conflict_rendering": "v1",
        }
        if args.canonical_ids != "off":
            checkpoint_fingerprint["canonical_ids"] = args.canonical_ids
        if args.ball_gate:
            checkpoint_fingerprint["ball_gate"] = True
        if args.team_smoothing:
            checkpoint_fingerprint["team_smoothing"] = True
    if checkpoint_path is not None and checkpoint_path.is_file():
        checkpoint_frame, checkpoint_state = load_checkpoint(
            checkpoint_path, checkpoint_fingerprint
        )
        if checkpoint_frame >= total_frames:
            raise ValueError(
                "Pipeline checkpoint frame exceeds the source video length."
            )
        checkpoint_data = {
            "fingerprint": checkpoint_fingerprint,
            "frame_index": checkpoint_frame,
            "state": checkpoint_state,
        }
    elif checkpoint_path is not None:
        for stale_segment in segments_dir.glob("segment_*.mp4"):
            stale_segment.unlink()
        resume_observations_path.unlink(missing_ok=True)
        identity_transition_path.unlink(missing_ok=True)
        pitch_rejections_path.unlink(missing_ok=True)

    frame_index = (
        checkpoint_data["frame_index"] if checkpoint_data is not None else 0
    )
    if checkpoint_data is not None:
        print(f"Resuming pipeline from frame {frame_index}/{total_frames}.")
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            cap.release()
            raise OSError(
                f"Could not seek source video to frame {frame_index}."
            )
    if progress_path is not None:
        _write_progress_sidecar(progress_path, frame_index, total_frames, "running")

    stub_reader = (
        TrackingStubReader(
            stub_path,
            video_path=video_path,
            frame_size=(frame_width, frame_height),
        )
        if args.use_stub
        else None
    )
    if stub_reader is not None:
        class_names = stub_reader.metadata["class_names"]
    class_categories = map_class_categories(class_names)
    print(f"Loaded player-model classes: {class_names}")
    print(f"Semantic class map: {class_categories}")
    available_categories = set(class_categories.values())
    for expected in ("player", "goalkeeper", "referee", "ball"):
        if expected not in available_categories:
            print(f"WARNING: model has no recognized {expected!r} class.")

    output_width, output_height = output_size(frame_width, frame_height, LAYOUT)
    next_segment_number = frame_index // _PIPELINE_CHECKPOINT_INTERVAL + 1
    if checkpoint_path is not None:
        segment_path = segments_dir / f"segment_{next_segment_number:06d}.mp4"
        for stale_segment in segments_dir.glob("segment_*.mp4"):
            if stale_segment.name >= segment_path.name:
                stale_segment.unlink()
    else:
        segment_path = output_path
    writer = _open_video_writer(
        segment_path, fps, (output_width, output_height)
    )
    observations_mode = "r+" if checkpoint_data is not None else "w+"
    observations_file = resume_observations_path.open(
        observations_mode, newline="", encoding="utf-8"
    )
    if checkpoint_data is not None:
        restore_checkpointed_file(
            observations_file,
            checkpoint_data["state"]["observations_offset"],
        )
    observations_writer = csv.DictWriter(
        observations_file, fieldnames=OBSERVATION_CSV_FIELDS
    )
    if checkpoint_data is None:
        observations_writer.writeheader()

    team_assigner = TeamColorAssigner()
    team_smoother = CanonicalTeamSmoother()
    team_conflict_tracker = CanonicalTeamConflictTracker()
    referee_manager = RefereeTrackManager()
    tracking_input_filter = PlayerTrackingInputFilter(
        class_categories,
        REFEREE_DUPLICATE_CONFIDENCE,
        REFEREE_DUPLICATE_IOU,
        enable_pitch_mask=ENABLE_PITCH_MASK,
        diagnostics_enabled=IDENTITY_DIAGNOSTICS,
    )
    profiler = _StageProfiler() if args.profile_stages else None

    def apply_tracking_input_filter(predictor):
        if profiler is None:
            tracking_input_filter(predictor)
            return
        started = perf_counter()
        try:
            tracking_input_filter(predictor)
        finally:
            profiler.record(
                "Pre-tracker filtering", perf_counter() - started
            )

    if model is not None:
        model.add_callback(
            "on_predict_postprocess_end", apply_tracking_input_filter
        )
    ball_tracker = BallPositionTracker()
    camera_radar = CameraRadarRenderer()
    possession = PossessionEstimator(POSSESSION_DISTANCE_THRESHOLD)
    pass_detector = EvidenceBasedPassDetector(fps, image_width=frame_width)
    global_identity_manager = GlobalIdentityManager(
        max_identities=64 if args.canonical_ids == "off" else None,
        max_lost_frames=54000 if args.canonical_ids == "off" else None,
    )
    camera_motion_estimator = CameraMotionEstimator()
    camera_reference = CameraReference()
    pending_output_frames = {}
    stub_writer = (
        TrackingStubWriter(
            stub_path,
            video_path=video_path,
            class_names=class_names,
            fps=fps,
            frame_size=(frame_width, frame_height),
        )
        if args.cache_stub
        else None
    )
    analyzer = (
        Phase1Analyzer(resume_root / "phase1_diagnostics", fps)
        if args.cache_stub or args.use_stub
        else None
    )
    if stub_writer is not None:
        print(f"Caching post-tracker detections to: {stub_path}")
    if analyzer is not None:
        print(f"Phase 1 diagnostics: {run_output_dir / 'phase1_diagnostics'}")
    class_counts_total = Counter()
    unknown_goalkeeper_ids = set()
    goalkeeper_track_ids = set()
    unknown_player_ids = set()
    referee_frame_counts = Counter()
    player_filter_counts = Counter()
    player_tracker_ids = set()
    unique_player_tracker_ids = set()
    active_player_tracker_ids = set()
    active_global_player_ids = set()
    unique_global_player_ids = set()
    max_simultaneous_players = 0
    max_active_tracker_ids = 0
    camera_radar_frames = 0
    total_yolo_detections = 0
    track_loss_events = 0
    track_reacquisition_events = 0
    previous_lost_track_ids = set()
    previous_player_positions = {}
    large_displacement_events = []
    previous_global_by_tracker = {}
    previous_tracker_by_global = {}
    identity_transition_path = resume_root / "identity_transitions.jsonl"
    if checkpoint_data is not None:
        saved = checkpoint_data["state"]
        pending_output_frames.update(
            saved.get("pending_output_frames", {})
        )
        for name, component in (
            ("team_assigner", team_assigner),
            ("team_smoother", team_smoother),
            ("team_conflict_tracker", team_conflict_tracker),
            ("referee_manager", referee_manager),
            ("tracking_input_filter", tracking_input_filter),
            ("ball_tracker", ball_tracker),
            ("camera_radar", camera_radar),
            ("possession", possession),
            ("pass_detector", pass_detector),
            ("global_identity_manager", global_identity_manager),
            ("camera_motion_estimator", camera_motion_estimator),
            ("camera_reference", camera_reference),
        ):
            if name not in saved["components"]:
                continue
            if name == "global_identity_manager":
                component.restore_checkpoint_state(
                    saved["components"][name]
                )
                continue
            component.__dict__.update(saved["components"][name])
        if analyzer is not None and saved.get("analyzer") is not None:
            analyzer.__dict__.update(saved["analyzer"])
        counters = saved["counters"]
        class_counts_total = counters["class_counts_total"]
        unknown_goalkeeper_ids = counters["unknown_goalkeeper_ids"]
        goalkeeper_track_ids = counters["goalkeeper_track_ids"]
        unknown_player_ids = counters["unknown_player_ids"]
        referee_frame_counts = counters["referee_frame_counts"]
        player_filter_counts = counters["player_filter_counts"]
        player_tracker_ids = counters["player_tracker_ids"]
        unique_player_tracker_ids = counters["unique_player_tracker_ids"]
        active_player_tracker_ids = counters["active_player_tracker_ids"]
        active_global_player_ids = counters["active_global_player_ids"]
        unique_global_player_ids = counters["unique_global_player_ids"]
        max_simultaneous_players = counters["max_simultaneous_players"]
        max_active_tracker_ids = counters["max_active_tracker_ids"]
        camera_radar_frames = counters["camera_radar_frames"]
        total_yolo_detections = counters["total_yolo_detections"]
        track_loss_events = counters["track_loss_events"]
        track_reacquisition_events = counters["track_reacquisition_events"]
        previous_lost_track_ids = counters["previous_lost_track_ids"]
        previous_player_positions = counters["previous_player_positions"]
        large_displacement_events = counters["large_displacement_events"]
        previous_global_by_tracker = counters["previous_global_by_tracker"]
        previous_tracker_by_global = counters["previous_tracker_by_global"]
        frame_index = checkpoint_data["frame_index"]
        if IDENTITY_DIAGNOSTICS:
            with identity_transition_path.open("r+b") as handle:
                handle.truncate(saved["identity_transition_offset"])
    else:
        frame_index = 0
    identity_transition_file = (
        identity_transition_path.open("a", encoding="utf-8")
        if IDENTITY_DIAGNOSTICS
        else None
    )
    evidence_mode = (
        "r+"
        if checkpoint_data is not None and identity_evidence_path.is_file()
        else "w+"
    )
    identity_evidence_file = identity_evidence_path.open(
        evidence_mode, newline="", encoding="utf-8"
    )
    if (
        checkpoint_data is not None
        and evidence_mode == "r+"
    ):
        restore_checkpointed_file(
            identity_evidence_file,
            checkpoint_data["state"].get("identity_evidence_offset"),
        )
    elif checkpoint_data is not None:
        restore_checkpointed_file(identity_evidence_file)
    identity_evidence_writer = csv.DictWriter(
        identity_evidence_file,
        fieldnames=IDENTITY_EVIDENCE_CSV_FIELDS,
    )
    if evidence_mode == "w+" or identity_evidence_file.tell() == 0:
        identity_evidence_writer.writeheader()
    pitch_rejections_mode = (
        "r+"
        if checkpoint_data is not None and pitch_rejections_path.is_file()
        else "w+"
    )
    pitch_rejections_file = pitch_rejections_path.open(
        pitch_rejections_mode, newline="", encoding="utf-8"
    )
    if (
        checkpoint_data is not None
        and pitch_rejections_mode == "r+"
    ):
        restore_checkpointed_file(
            pitch_rejections_file,
            checkpoint_data["state"].get("pitch_rejections_offset"),
        )
    elif checkpoint_data is not None:
        restore_checkpointed_file(pitch_rejections_file)
    pitch_rejections_writer = csv.DictWriter(
        pitch_rejections_file,
        fieldnames=PITCH_REJECTION_CSV_FIELDS,
    )
    if pitch_rejections_mode == "w+" or pitch_rejections_file.tell() == 0:
        pitch_rejections_writer.writeheader()
    print(
        f"Processing {video_path}: {frame_width}x{frame_height} at {fps:g} FPS; "
        f"radar_mode={RADAR_MODE}; layout={LAYOUT}; "
        f"output={output_width}x{output_height}"
    )
    tracker_profile_hooks = []
    profile_cuda = False
    synchronize_cuda = lambda: None
    if args.profile_stages and model is not None:
        import torch

        profile_cuda = inference_backend == "cuda"
        def synchronize_profile_device():
            if profile_cuda:
                torch.cuda.synchronize(inference_device)

        synchronize_cuda = synchronize_profile_device
        if profile_cuda:
            device = inference_device
            torch.cuda.reset_peak_memory_stats(device)
            print(
                f"Profiling GPU: {torch.cuda.get_device_name(device)} "
                f"(CUDA {torch.version.cuda})"
            )
        else:
            print("Profiling device: CPU")
        tracker_profile_hooks = _install_tracker_profile_hooks(
            profiler, synchronize_cuda
        )
        warmup_wrapped_models = set()

        def profile_model_warmup(predictor):
            predictor_model = predictor.model
            model_identity = id(predictor_model)
            if model_identity in warmup_wrapped_models:
                return
            original_warmup = predictor_model.warmup

            def timed_warmup(*warmup_args, **warmup_kwargs):
                synchronize_cuda()
                warmup_started = perf_counter()
                try:
                    return original_warmup(*warmup_args, **warmup_kwargs)
                finally:
                    synchronize_cuda()
                    profiler.record(
                        "Model warm-up",
                        perf_counter() - warmup_started,
                    )

            predictor_model.warmup = timed_warmup
            warmup_wrapped_models.add(model_identity)

        model.add_callback("on_predict_start", profile_model_warmup)
    processing_started = perf_counter()
    try:
        if checkpoint_data is not None and model is not None:
            warm_success, warm_frame = cap.read()
            if not warm_success:
                raise OSError(
                    f"Could not read source frame {frame_index + 1} to restore tracking."
                )
            tracking_options = dict(
                persist=True,
                conf=DETECTION_CONFIDENCE,
                imgsz=args.imgsz,
                tracker=str(tracker_config),
                verbose=False,
            )
            if inference_device is not None:
                tracking_options["device"] = inference_device
            model.track(warm_frame, **tracking_options)
            tracker = model.predictor.trackers[0]
            _restore_tracker_checkpoint_state(
                tracker, checkpoint_data["state"]["tracker"]
            )
            tracking_input_filter.__dict__.update(
                checkpoint_data["state"]["components"]["tracking_input_filter"]
            )
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                raise OSError(
                    f"Could not seek source video to frame {frame_index} "
                    "after restoring tracking."
                )

        def save_pipeline_checkpoint():
            nonlocal writer, segment_path
            if checkpoint_path is None or model is None:
                return
            writer.release()
            observations_file.flush()
            os.fsync(observations_file.fileno())
            identity_offset = None
            if identity_transition_file is not None:
                identity_transition_file.flush()
                os.fsync(identity_transition_file.fileno())
                identity_offset = identity_transition_file.tell()
            identity_evidence_file.flush()
            os.fsync(identity_evidence_file.fileno())
            pitch_rejections_file.flush()
            os.fsync(pitch_rejections_file.fileno())
            tracker = model.predictor.trackers[0]
            components = {
                "team_assigner": team_assigner.__dict__.copy(),
                "team_smoother": team_smoother.__dict__.copy(),
                "team_conflict_tracker": team_conflict_tracker.__dict__.copy(),
                "referee_manager": referee_manager.__dict__.copy(),
                "tracking_input_filter": tracking_input_filter.__dict__.copy(),
                "ball_tracker": ball_tracker.__dict__.copy(),
                "camera_radar": camera_radar.__dict__.copy(),
                "possession": possession.__dict__.copy(),
                "pass_detector": pass_detector.__dict__.copy(),
                "global_identity_manager": (
                    global_identity_manager.checkpoint_state()
                ),
                "camera_motion_estimator": camera_motion_estimator.__dict__.copy(),
                "camera_reference": camera_reference.__dict__.copy(),
            }
            counters = {
                "class_counts_total": class_counts_total,
                "unknown_goalkeeper_ids": unknown_goalkeeper_ids,
                "goalkeeper_track_ids": goalkeeper_track_ids,
                "unknown_player_ids": unknown_player_ids,
                "referee_frame_counts": referee_frame_counts,
                "player_filter_counts": player_filter_counts,
                "player_tracker_ids": player_tracker_ids,
                "unique_player_tracker_ids": unique_player_tracker_ids,
                "active_player_tracker_ids": active_player_tracker_ids,
                "active_global_player_ids": active_global_player_ids,
                "unique_global_player_ids": unique_global_player_ids,
                "max_simultaneous_players": max_simultaneous_players,
                "max_active_tracker_ids": max_active_tracker_ids,
                "camera_radar_frames": camera_radar_frames,
                "total_yolo_detections": total_yolo_detections,
                "track_loss_events": track_loss_events,
                "track_reacquisition_events": track_reacquisition_events,
                "previous_lost_track_ids": previous_lost_track_ids,
                "previous_player_positions": previous_player_positions,
                "large_displacement_events": large_displacement_events,
                "previous_global_by_tracker": previous_global_by_tracker,
                "previous_tracker_by_global": previous_tracker_by_global,
            }
            save_checkpoint(
                checkpoint_path,
                checkpoint_fingerprint,
                frame_index,
                {
                    "components": components,
                    "counters": counters,
                    "tracker": _tracker_checkpoint_state(tracker),
                    "observations_offset": observations_file.tell(),
                    "identity_transition_offset": identity_offset,
                    "identity_evidence_offset": identity_evidence_file.tell(),
                    "pitch_rejections_offset": pitch_rejections_file.tell(),
                    "pending_output_frames": pending_output_frames,
                },
            )
            next_number = frame_index // _PIPELINE_CHECKPOINT_INTERVAL + 1
            segment_path = segments_dir / f"segment_{next_number:06d}.mp4"
            writer = _open_video_writer(
                segment_path, fps, (output_width, output_height)
            )

        def emit_finalized_frame(context, ball_result):
            nonlocal camera_radar_frames
            frame_number = context["frame_number"]
            frame = context["frame"]
            ball_position = ball_result["position"]
            ball_source = ball_result["source"]
            ball_on_pitch = (
                ball_position is not None
                and is_inside_playing_area(ball_position, frame.shape)
            )
            ball_observation = BallObservation(
                frame_number=frame_number,
                timestamp_seconds=(frame_number - 1) / fps,
                image_x=(
                    float(ball_position[0])
                    if ball_position is not None
                    else None
                ),
                image_y=(
                    float(ball_position[1])
                    if ball_position is not None
                    else None
                ),
                pitch_x=None,
                pitch_y=None,
                confidence=ball_result["confidence"],
                source=ball_source,
                is_valid=bool(ball_on_pitch),
            )
            radar_ball_position = ball_observation.position
            radar_started = perf_counter() if profiler is not None else 0.0
            radar = camera_radar.render(
                context["camera_markers"],
                frame.shape,
                ball_position=radar_ball_position,
                team_colors=context["team_colors"],
            )
            camera_radar_frames += 1
            if profiler is not None:
                profiler.record(
                    "Radar rendering", perf_counter() - radar_started
                )

            possession_started = (
                perf_counter() if profiler is not None else 0.0
            )
            player_observations = context["player_observations"]
            possession_candidates = [
                observation
                for observation in player_observations
                if args.canonical_ids != "offline"
                and observation.global_id is not None
                if observation.player_type
                in (TeamColorAssigner.TEAM_1, TeamColorAssigner.TEAM_2)
                and observation.object_type != "goalkeeper"
            ]
            possession_ball_position = radar_ball_position
            possession_ball_position = _select_possession_ball_position(
                possession_ball_position,
                ball_source,
                args.ball_gate,
            )
            possession_result = possession.update(
                (
                    None
                    if args.canonical_ids == "offline"
                    else possession_ball_position
                ),
                possession_candidates,
            )
            if profiler is not None:
                profiler.record(
                    "Possession calculation",
                    perf_counter() - possession_started,
                )

            player_observations = [
                replace(
                    observation,
                    possession_state=possession_result["state"],
                    possession_player_id=possession_result["player_id"],
                )
                for observation in player_observations
            ]
            pass_started = perf_counter() if profiler is not None else 0.0
            pass_detector.update(
                frame_number,
                ball_observation.timestamp_seconds,
                ball_observation.position,
                ball_observation.source,
                ball_observation.is_valid,
                player_observations,
            )
            if profiler is not None:
                profiler.record(
                    "Live pass detection", perf_counter() - pass_started
                )
            observation_dicts = [
                observation.to_dict()
                for observation in player_observations
            ]
            csv_write_started = (
                perf_counter() if profiler is not None else 0.0
            )
            csv_rows = []
            for observation, row in zip(
                player_observations, observation_dicts
            ):
                row = dict(row)
                row["object_type"] = observation.object_type
                row["bbox"] = json.dumps(row["bbox"])
                row["team_color"] = json.dumps(row["team_color"])
                csv_rows.append(row)
            ball_row = ball_observation.to_dict()
            ball_row["object_type"] = "ball"
            ball_row["possession_state"] = possession_result["state"]
            ball_row["possession_player_id"] = possession_result["player_id"]
            csv_rows.append(ball_row)
            observations_writer.writerows(csv_rows)
            if profiler is not None:
                profiler.record(
                    "Observations CSV writing",
                    perf_counter() - csv_write_started,
                )
            visualization_started = (
                perf_counter() if profiler is not None else 0.0
            )
            annotated_frame = draw_annotations(
                frame,
                observation_dicts,
                copy_frame=False,
                identity_resolved=context["identity_resolved"],
            )
            output_frame = compose_dashboard(
                annotated_frame,
                radar,
                possession,
                LAYOUT,
                team_colors=context["team_colors"],
                pass_counts=pass_detector.counts_by_team(),
            )
            if profiler is not None:
                profiler.record(
                    "Frame annotation + dashboard",
                    perf_counter() - visualization_started,
                )
            write_started = perf_counter() if profiler is not None else 0.0
            writer.write(output_frame)
            if profiler is not None:
                profiler.record(
                    "Video writing", perf_counter() - write_started
                )
            if DEBUG_POSSESSION:
                print(
                    f"Possession frame {frame_number}: "
                    f"ball={ball_position} ({ball_source}), "
                    f"closest_id={possession_result['closest_player_id']}, "
                    f"distance={possession_result['closest_distance']}, "
                    f"state={possession_result['state']}"
                )

        while True:
            if args.max_frames and frame_index >= args.max_frames:
                break
            read_started = perf_counter()
            success, frame = cap.read()
            if not success:
                break
            if profiler is not None:
                profiler.record(
                    "Video frame read", perf_counter() - read_started
                )
            frame_index += 1
            camera_started = perf_counter() if profiler is not None else 0.0
            stub_record = None
            stub_read_seconds = 0.0
            if stub_reader is not None:
                stub_read_started = (
                    perf_counter() if profiler is not None else 0.0
                )
                stub_record = stub_reader.read_frame(frame_index)
                if profiler is not None:
                    stub_read_seconds = perf_counter() - stub_read_started
                    profiler.record("Tracking cache read", stub_read_seconds)
                if stub_record is None:
                    break
                camera_motion = camera_motion_from_dict(
                    stub_record["camera_motion"]
                )
            else:
                camera_motion = camera_motion_estimator.update(frame)
            camera_reference.update(camera_motion)
            if profiler is not None:
                profiler.record(
                    "Camera motion compensation",
                    perf_counter() - camera_started - stub_read_seconds,
                )
            if stub_record is not None:
                detections = stub_record["detections"]
                player_filter_counts.update(
                    stub_record.get("pretracker_filter_counts", {})
                )
                tracking_input_filter.valid_detection_counts.clear()
                tracking_input_filter.valid_detection_counts.update(
                    stub_record.get("pretracker_valid_detection_counts", {})
                )
                tracking_input_filter.total_detection_count = int(
                    stub_record.get("pretracker_total_detection_count", 0)
                )
                current_lost_track_ids = set(
                    stub_record.get("lost_track_ids", ())
                )
            else:
                tracking_options = dict(
                    persist=True,
                    conf=DETECTION_CONFIDENCE,
                    imgsz=args.imgsz,
                    tracker=str(tracker_config),
                    verbose=False,
                )
                if inference_device is not None:
                    tracking_options["device"] = inference_device
                if profiler is not None:
                    synchronize_cuda()
                    track_started = perf_counter()
                    filtering_before = profiler.totals["Pre-tracker filtering"]
                    tracker_before = profiler.totals["BoT-SORT update"]
                    warmup_before = profiler.totals["Model warm-up"]
                tracking_input_filter.frame_number = frame_index
                model_results = model.track(frame, **tracking_options)
                for rejected in tracking_input_filter.rejected_detections:
                    x1, y1, x2, y2 = rejected["bbox"]
                    pitch_rejections_writer.writerow(
                        {
                            "frame_number": frame_index,
                            "timestamp_seconds": (frame_index - 1) / fps,
                            "class_id": rejected["class_id"],
                            "object_type": rejected["object_type"],
                            "confidence": rejected["confidence"],
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "rejection_reason": rejected[
                                "rejection_reason"
                            ],
                        }
                    )
                if profiler is not None:
                    synchronize_cuda()
                    track_wall_seconds = perf_counter() - track_started
                    yolo_profile_seconds = 0.0
                    for result in model_results:
                        speed = getattr(result, "speed", {})
                        for stage, key in (
                            ("YOLO preprocessing", "preprocess"),
                            ("YOLO inference", "inference"),
                            ("YOLO postprocessing", "postprocess"),
                        ):
                            stage_seconds = float(speed.get(key, 0.0)) / 1000.0
                            profiler.record(stage, stage_seconds)
                            yolo_profile_seconds += stage_seconds
                    filter_seconds = (
                        profiler.totals["Pre-tracker filtering"]
                        - filtering_before
                    )
                    tracker_seconds = (
                        profiler.totals["BoT-SORT update"] - tracker_before
                    )
                    warmup_seconds = (
                        profiler.totals["Model warm-up"] - warmup_before
                    )
                    predictor_overhead = max(
                        0.0,
                        track_wall_seconds
                        - filter_seconds
                        - tracker_seconds
                        - warmup_seconds
                        - yolo_profile_seconds,
                    )
                    profiler.record(
                        "Predictor / callback overhead",
                        predictor_overhead,
                    )
                tracker = model.predictor.trackers[0]
                detections = tracking_input_filter.attach_tracker_ids(tracker)
                tracker_features = {}
                refresh_reid_features = (
                    frame_index
                    % global_identity_manager.REID_FEATURE_REFRESH_INTERVAL_FRAMES
                    == 0
                )
                required_feature_ids = {
                    int(detection["tracker_id"])
                    for detection in detections
                    if detection.get("tracker_id") is not None
                    and (
                        refresh_reid_features
                        or int(detection["tracker_id"])
                        not in global_identity_manager.tracker_to_global
                    )
                }
                for active_track in tracker.tracked_stracks:
                    track_id = int(active_track.track_id)
                    if track_id not in required_feature_ids:
                        continue
                    feature = getattr(active_track, "smooth_feat", None)
                    if feature is None:
                        continue
                    if hasattr(feature, "detach"):
                        feature = feature.detach().cpu().numpy()
                    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
                    if feature.size and np.isfinite(feature).all():
                        tracker_features[track_id] = feature
                for detection in detections:
                    tracker_id = detection.get("tracker_id")
                    if tracker_id is not None and tracker_id in tracker_features:
                        detection["reid_feature"] = tracker_features[tracker_id]
                current_lost_track_ids = {
                    int(track.track_id) for track in tracker.lost_stracks
                }
                if stub_writer is not None:
                    stub_writer.write_frame(
                        frame_index,
                        detections,
                        camera_motion,
                        tracking_input_filter.filter_counts,
                        tracking_input_filter.valid_detection_counts,
                        tracking_input_filter.total_detection_count,
                        current_lost_track_ids,
                    )
            post_tracking_started = (
                perf_counter() if profiler is not None else 0.0
            )
            total_yolo_detections += len(detections)
            class_counts_total.update(
                _class_label(class_names, detection["class_id"])
                for detection in detections
            )
            if stub_record is None:
                player_filter_counts.update(
                    tracking_input_filter.filter_counts
                )
            active_player_tracker_ids = {
                detection["tracker_id"]
                for detection in detections
                if detection["object_type"] in ("player", "goalkeeper")
                and detection["tracker_id"] is not None
            }
            unique_player_tracker_ids.update(active_player_tracker_ids)
            max_active_tracker_ids = max(
                max_active_tracker_ids, len(active_player_tracker_ids)
            )
            track_loss_events += len(
                current_lost_track_ids - previous_lost_track_ids
            )
            track_reacquisition_events += len(
                previous_lost_track_ids & active_player_tracker_ids
            )
            previous_lost_track_ids = current_lost_track_ids
            for detection in detections:
                tracker_id = detection["tracker_id"]
                if (
                    detection["object_type"] not in ("player", "goalkeeper")
                    or tracker_id is None
                    or detection["confidence"] < PLAYER_CONFIDENCE_THRESHOLD
                ):
                    continue
                x1, y1, x2, y2 = detection["bbox"]
                current_position = (
                    (x1 + x2) / 2.0,
                    y2,
                    max(0.0, y2 - y1),
                )
                previous_position = previous_player_positions.get(tracker_id)
                if previous_position is not None:
                    previous_frame, previous_x, previous_y, previous_height = (
                        previous_position
                    )
                    frame_gap = frame_index - previous_frame
                    displacement = float(
                        np.hypot(
                            current_position[0] - previous_x,
                            current_position[1] - previous_y,
                        )
                    )
                    if frame_gap <= 2 and displacement > max(
                        60.0,
                        2.0
                        * max(previous_height, current_position[2]),
                    ):
                        large_displacement_events.append(
                            {
                                "tracker_id": tracker_id,
                                "from_frame": previous_frame,
                                "to_frame": frame_index,
                                "distance_px": round(displacement, 1),
                                "previous_box_height": round(
                                    previous_height, 1
                                ),
                                "current_box_height": round(
                                    current_position[2], 1
                                ),
                            }
                        )
                previous_player_positions[tracker_id] = (
                    frame_index,
                    *current_position,
                )
            if profiler is not None:
                profiler.record(
                    "Tracker-ID / lifecycle bookkeeping",
                    perf_counter() - post_tracking_started,
                )

            referee_started = perf_counter() if profiler is not None else 0.0
            visible_ids, referee_counts = referee_manager.update(detections)
            referee_ids = set(referee_manager.referee_reason)
            referee_ids.update(referee_manager.current_class_referee_ids)
            visible_ids.difference_update(referee_ids)
            referee_detections = [
                detection
                for detection in detections
                if detection["object_type"] == "referee"
            ]
            player_candidates = [
                detection
                for detection in detections
                if detection["object_type"] in ("player", "goalkeeper")
                and detection["tracker_id"] is not None
                and detection["confidence"] >= PLAYER_CONFIDENCE_THRESHOLD
            ]
            player_detections = [
                detection
                for detection in player_candidates
                if detection["tracker_id"] not in referee_ids
            ]

            if ENABLE_OTHER_BUCKET:
                team_assigner.observe_referee_detections(
                    frame, referee_detections
                )
            eligible_player_detections = [
                detection
                for detection in player_detections
                if detection["tracker_id"] in visible_ids
            ]
            if profiler is not None:
                profiler.record(
                    "Post-tracker referee / eligibility filtering",
                    perf_counter() - referee_started,
                )
            team_started = perf_counter() if profiler is not None else 0.0
            player_types = team_assigner.update(
                frame,
                eligible_player_detections,
                referee_ids=referee_ids,
                enable_other_bucket=ENABLE_OTHER_BUCKET,
            )
            if profiler is not None:
                profiler.record(
                    "Team classification",
                    perf_counter() - team_started,
                )
            identity_started = perf_counter() if profiler is not None else 0.0
            identity_detections = []
            for detection in eligible_player_detections:
                tracker_id = int(detection["tracker_id"])
                if tracker_id not in visible_ids:
                    continue
                player_type = player_types.get(
                    tracker_id,
                    TeamColorAssigner.UNKNOWN_GOALKEEPER
                    if detection["object_type"] == "goalkeeper"
                    else TeamColorAssigner.UNKNOWN,
                )
                if player_type == TeamColorAssigner.REFEREE:
                    continue
                team_is_provisional = team_assigner.is_provisional(
                    tracker_id
                )
                identity_team = (
                    TeamColorAssigner.UNKNOWN
                    if (
                        player_type == TeamColorAssigner.OTHER
                        or team_is_provisional
                    )
                    else player_type
                )
                identity_detections.append({
                    **detection,
                    "team": identity_team,
                    "team_confident": identity_team
                    in (TeamColorAssigner.TEAM_1, TeamColorAssigner.TEAM_2),
                    "camera_compensated_position": camera_reference.to_reference(
                        (
                            (detection["bbox"][0] + detection["bbox"][2]) / 2.0,
                            detection["bbox"][3],
                        )
                    ),
                    "camera_epoch": camera_reference.epoch,
                    "player_eligible": True,
                })
            previous_global_identity_teams = {
                global_id: (identity.tracker_id, identity.team)
                for global_id, identity in global_identity_manager.identities.items()
                if identity.team
                in (TeamColorAssigner.TEAM_1, TeamColorAssigner.TEAM_2)
            }
            if args.canonical_ids == "offline":
                global_id_by_tracker = {
                    int(detection["tracker_id"]): int(detection["tracker_id"])
                    for detection in identity_detections
                }
            else:
                global_id_by_tracker = global_identity_manager.update(
                    identity_detections,
                    frame_index,
                    frame=frame,
                    camera_epoch=camera_reference.epoch,
                )
            if global_identity_manager.last_association_evidence:
                identity_evidence_writer.writerows(
                    global_identity_manager.last_association_evidence
                )
            for tracker_id, global_id in global_id_by_tracker.items():
                previous_identity = previous_global_identity_teams.get(global_id)
                if (
                    previous_identity is None
                    or previous_identity[0] == tracker_id
                ):
                    continue
                stable_team = previous_identity[1]
                team_assigner.inherit_assignment(tracker_id, stable_team)
                player_types[tracker_id] = stable_team
            for tracker_id in team_assigner.last_goalkeeper_resets:
                team_assigner.clear_goalkeeper_assignment(tracker_id)
                player_types[tracker_id] = TeamColorAssigner.UNKNOWN_GOALKEEPER
                global_id = global_id_by_tracker.get(tracker_id)
                if global_id is None:
                    continue
                team_smoother.clear(global_id)
                team_conflict_tracker.clear(global_id)
                global_identity_manager.clear_team_label(global_id)
            if args.canonical_ids != "offline":
                for tracker_id, global_id in global_id_by_tracker.items():
                    identity = global_identity_manager.identities.get(
                        global_id
                    )
                    if identity is None or identity.team not in (
                        TeamColorAssigner.TEAM_1,
                        TeamColorAssigner.TEAM_2,
                    ):
                        continue
                    player_types[tracker_id] = identity.team
                    team_assigner.inherit_assignment(
                        tracker_id, identity.team
                    )
            if args.team_smoothing:
                for tracker_id, global_id in global_id_by_tracker.items():
                    if player_types.get(tracker_id) not in (
                        TeamColorAssigner.TEAM_1,
                        TeamColorAssigner.TEAM_2,
                    ):
                        continue
                    smoothed_team = team_smoother.update(
                        global_id,
                        frame_index,
                        team_assigner.last_team_evidence.get(tracker_id),
                    )
                    if smoothed_team is None:
                        continue
                    player_types[tracker_id] = smoothed_team
                    team_assigner.inherit_assignment(
                        tracker_id, smoothed_team
                    )
                    global_identity_manager.set_team_label(
                        global_id, smoothed_team
                    )
                    identity = global_identity_manager.identities.get(
                        global_id
                    )
                    if identity is not None and identity.team in (
                        TeamColorAssigner.TEAM_1,
                        TeamColorAssigner.TEAM_2,
                    ):
                        player_types[tracker_id] = identity.team
                        team_assigner.inherit_assignment(
                            tracker_id, identity.team
                        )
            if args.canonical_ids != "offline":
                player_object_types = {
                    int(detection["tracker_id"]): detection["object_type"]
                    for detection in eligible_player_detections
                    if detection.get("tracker_id") is not None
                }
                for tracker_id, global_id in global_id_by_tracker.items():
                    if player_object_types.get(int(tracker_id)) != "player":
                        continue
                    identity = global_identity_manager.identities.get(
                        global_id
                    )
                    canonical_team = (
                        identity.team
                        if identity is not None
                        else player_types.get(tracker_id)
                    )
                    if canonical_team not in (
                        TeamColorAssigner.TEAM_1,
                        TeamColorAssigner.TEAM_2,
                    ):
                        continue
                    if team_conflict_tracker.update(
                        global_id,
                        canonical_team,
                        frame_index,
                        team_assigner.last_observed_team_evidence.get(
                            tracker_id
                        ),
                    ):
                        player_types[tracker_id] = (
                            TeamColorAssigner.TEAM_CONFLICT
                        )
            active_global_player_ids = set(global_id_by_tracker.values())
            unique_global_player_ids.update(active_global_player_ids)
            if (
                IDENTITY_DIAGNOSTICS
                and args.canonical_ids != "offline"
                and global_id_by_tracker
            ):
                current_global_by_tracker = {
                    int(tracker_id): int(global_id)
                    for tracker_id, global_id in global_id_by_tracker.items()
                }
                current_tracker_by_global = {
                    int(global_id): int(tracker_id)
                    for tracker_id, global_id in current_global_by_tracker.items()
                }
                timestamp = (
                    round((frame_index - 1) / fps, 2)
                    if fps and fps > 0
                    else None
                )
                for event in _identity_transition_events(
                    previous_global_by_tracker,
                    previous_tracker_by_global,
                    current_global_by_tracker,
                    frame_index,
                    camera_epoch=camera_reference.epoch,
                    timestamp=timestamp,
                ):
                    identity_transition_file.write(
                        json.dumps(event, sort_keys=True) + "\n"
                    )
                    identity_transition_file.flush()
                previous_global_by_tracker = current_global_by_tracker.copy()
                previous_tracker_by_global = current_tracker_by_global.copy()
            diagnostics_seconds = 0.0
            if analyzer is not None:
                diagnostics_started = (
                    perf_counter() if profiler is not None else 0.0
                )
                analyzer.observe(
                    frame,
                    frame_index,
                    detections,
                    player_types,
                    global_id_by_tracker,
                    team_assigner,
                    camera_reference.epoch,
                )
                if profiler is not None:
                    diagnostics_seconds = (
                        perf_counter() - diagnostics_started
                    )
                    profiler.record(
                        "Phase 1 diagnostics",
                        diagnostics_seconds,
                    )
            if profiler is not None:
                profiler.record(
                    "Global identity assignment + appearance",
                    perf_counter() - identity_started - diagnostics_seconds,
                )
            visible_ids.difference_update(referee_ids)
            referee_counts["rejected_by_kit_color"] = sum(
                referee_manager.referee_reason.get(detection["tracker_id"])
                == "rejected_by_kit_color"
                for detection in detections
                if detection.get("tracker_id") is not None
            )
            referee_frame_counts.update(referee_counts)
            referee_frame_counts["detected_by_class"] += sum(
                detection["object_type"] == "referee"
                for detection in detections
            )

            ball_started = perf_counter() if profiler is not None else 0.0
            ball_detections = [
                detection
                for detection in detections
                if detection["object_type"] == "ball"
            ]
            resolved_ball_records = ball_tracker.update(
                ball_detections, frame_index
            )
            if profiler is not None:
                profiler.record(
                    "Ball tracking",
                    perf_counter() - ball_started,
                )

            observations_started = (
                perf_counter() if profiler is not None else 0.0
            )
            player_observations = []
            for detection in eligible_player_detections:
                tracker_id = detection["tracker_id"]
                if tracker_id not in visible_ids:
                    continue
                player_type = player_types.get(
                    tracker_id,
                    TeamColorAssigner.UNKNOWN_GOALKEEPER
                    if detection["object_type"] == "goalkeeper"
                    else TeamColorAssigner.UNKNOWN,
                )
                if player_type == TeamColorAssigner.REFEREE:
                    continue
                global_id = global_id_by_tracker.get(tracker_id)
                if (
                    player_type == TeamColorAssigner.OTHER
                    and (
                        global_id is None
                        or args.canonical_ids == "offline"
                    )
                ):
                    continue
                if args.canonical_ids == "offline":
                    global_id = None
                team_color = team_assigner.team_colors.get(
                    player_type,
                    TeamColorAssigner.TEAM_COLORS[
                        TeamColorAssigner.GOALKEEPER
                    ]
                    if detection["object_type"] == "goalkeeper"
                    else TeamColorAssigner.NEUTRAL_COLOR,
                )
                x1, _, x2, y2 = detection["bbox"]
                foot_x = (x1 + x2) / 2.0
                foot_y = y2
                player_observations.append(
                    PlayerObservation(
                        frame_number=frame_index,
                        timestamp_seconds=(frame_index - 1) / fps,
                        tracker_id=tracker_id,
                        object_type=detection["object_type"],
                        player_type=player_type,
                        image_x=foot_x,
                        image_y=foot_y,
                        foot_x=foot_x,
                        foot_y=foot_y,
                        pitch_x=None,
                        pitch_y=None,
                        confidence=detection["confidence"],
                        is_visible=True,
                        bbox=tuple(detection["bbox"]),
                        team_color=tuple(team_color),
                        global_id=global_id,
                    )
                )
                player_tracker_ids.add(tracker_id)
                if player_type == TeamColorAssigner.UNKNOWN_GOALKEEPER:
                    unknown_goalkeeper_ids.add(tracker_id)
                elif detection["object_type"] == "goalkeeper":
                    unknown_goalkeeper_ids.discard(tracker_id)
                elif player_type == TeamColorAssigner.UNKNOWN:
                    unknown_player_ids.add(tracker_id)
                else:
                    unknown_player_ids.discard(tracker_id)
                if detection["object_type"] == "goalkeeper":
                    goalkeeper_track_ids.add(tracker_id)

            max_simultaneous_players = max(
                max_simultaneous_players, len(player_observations)
            )
            camera_markers = {
                observation.tracker_id: (
                    observation.foot_x,
                    observation.foot_y,
                    observation.team_color,
                    observation.global_id,
                )
                for observation in player_observations
                if is_displayable_player(
                    {
                        "object_type": observation.object_type,
                        "global_id": observation.global_id,
                        "player_type": observation.player_type,
                    }
                )
            }
            if IDENTITY_DIAGNOSTICS and stub_record is None:
                rendered_tracker_ids = {
                    observation.tracker_id
                    for observation in player_observations
                }
                for detection in detections:
                    object_type = detection["object_type"]
                    tracker_id = detection.get("tracker_id")
                    global_id = (
                        global_id_by_tracker.get(tracker_id)
                        if tracker_id is not None
                        else None
                    )
                    team = (
                        player_types.get(tracker_id)
                        if tracker_id is not None
                        else None
                    )
                    if object_type not in ("player", "goalkeeper"):
                        stage = "non_player_class"
                    elif detection.get("pretracker_rejection_reason"):
                        stage = detection["pretracker_rejection_reason"]
                    elif not detection.get("pretracker_selected", False):
                        stage = "not_selected_for_tracking"
                    elif tracker_id is None:
                        stage = "not_tracked"
                    elif detection["confidence"] < PLAYER_CONFIDENCE_THRESHOLD:
                        stage = "below_player_confidence"
                    elif tracker_id not in visible_ids:
                        stage = "not_visible_or_referee_filtered"
                    elif args.canonical_ids != "offline" and global_id is None:
                        stage = "missing_canonical_id"
                    elif tracker_id not in rendered_tracker_ids:
                        stage = "not_rendered"
                    else:
                        stage = "rendered"
                    print(
                        "Detection flow: "
                        + json.dumps(
                            {
                                "frame": frame_index,
                                "class": object_type,
                                "confidence": detection["confidence"],
                                "bbox": detection["bbox"],
                                "filter_selected": detection.get(
                                    "pretracker_selected"
                                ),
                                "filter_rejection": detection.get(
                                    "pretracker_rejection_reason"
                                ),
                                "spatial_evidence": detection.get(
                                    "pretracker_spatial_evidence"
                                ),
                                "tracker_id": tracker_id,
                                "global_id": global_id,
                                "team": team,
                                "previous_team_assignment": (
                                    team_assigner.last_goalkeeper_resets.get(
                                        tracker_id
                                    )
                                ),
                                "team_color_distances": (
                                    team_assigner.last_goalkeeper_distances.get(
                                        tracker_id
                                    )
                                    if team_assigner.is_goalkeeper_tracker(
                                        tracker_id
                                    )
                                    else team_assigner.last_color_distances.get(
                                        tracker_id
                                    )
                                ),
                                "team_evidence": team_assigner.last_team_evidence.get(
                                    tracker_id
                                ),
                                "association": next(
                                    (
                                        evidence
                                        for evidence in global_identity_manager.last_association_evidence
                                        if evidence.get("current_tracker_id")
                                        == tracker_id
                                    ),
                                    None,
                                ),
                                "rendered": tracker_id
                                in rendered_tracker_ids,
                                "stage": stage,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            if profiler is not None:
                profiler.record(
                    "Player and ball observations",
                    perf_counter() - observations_started,
                )
            pending_output_frames[frame_index] = {
                "frame_number": frame_index,
                "frame": frame.copy(),
                "player_observations": tuple(player_observations),
                "camera_markers": camera_markers,
                "team_colors": dict(team_assigner.team_colors),
                "identity_resolved": args.canonical_ids != "offline",
            }
            for ball_result in resolved_ball_records:
                output_context = pending_output_frames.pop(
                    ball_result["frame_number"], None
                )
                if output_context is None:
                    raise RuntimeError(
                        "Ball tracker finalized a frame without buffered output "
                        f"data: {ball_result['frame_number']}"
                    )
                emit_finalized_frame(output_context, ball_result)

            referee_tracks_this_frame = sum(
                detection.get("tracker_id") in referee_manager.referee_reason
                or detection.get("tracker_id")
                in referee_manager.current_class_referee_ids
                for detection in detections
            )
            if frame_index <= 5:
                print(
                    f"Frame {frame_index}: class_counts="
                    f"{dict(Counter(_class_label(class_names, d['class_id']) for d in detections))}"
                )
            if DEBUG_DETECTIONS and frame_index % 30 == 0:
                print(
                    f"Frame {frame_index}/{total_frames}: "
                    f"referees={referee_tracks_this_frame} "
                    f"(class={referee_counts['detected_by_class']}, "
                    f"history={referee_counts['reclassified_by_class_history']}, "
                    f"kit={referee_counts['rejected_by_kit_color']}), "
                    f"visible_players={len(player_observations)}, "
                    f"goalkeepers={sum(o.object_type == 'goalkeeper' for o in player_observations)}, "
                    f"outside_pitch_filtered={player_filter_counts['outside_pitch']}, "
                    f"referee_duplicates_filtered={player_filter_counts['overlapping_referee']}, "
                    f"team_model_fitted={team_assigner.team_centers_fitted}, "
                    f"ball_detected={bool(ball_detections)}"
                )
            if DEBUG_DETECTIONS:
                for detection in eligible_player_detections:
                    if detection["object_type"] == "goalkeeper":
                        tracker_id = detection["tracker_id"]
                        assignment = player_types.get(
                            tracker_id, TeamColorAssigner.UNKNOWN_GOALKEEPER
                        )
                        print(
                            f"Goalkeeper track={tracker_id} "
                            f"team_color_distances="
                            f"{team_assigner.last_goalkeeper_distances.get(tracker_id)} "
                            f"assignment={assignment}"
                        )
                print(
                    f"Referee frame {frame_index}: "
                    f"class={referee_counts['detected_by_class']}, "
                    f"history={referee_counts['reclassified_by_class_history']}, "
                    f"kit={referee_counts['rejected_by_kit_color']}"
                )
            if (
                args.progress_every
                and frame_index % args.progress_every == 0
            ):
                elapsed = perf_counter() - processing_started
                processing_fps = frame_index / elapsed if elapsed else 0.0
                eta_seconds = (
                    (total_frames - frame_index) / processing_fps
                    if processing_fps and total_frames > frame_index
                    else 0.0
                )
                progress = (
                    f"{100.0 * frame_index / total_frames:.1f}%"
                    if total_frames > 0
                    else "unknown"
                )
                print(
                    f"Frame: {frame_index} / {total_frames or 'unknown'} | "
                    f"Progress: {progress} | "
                    f"Elapsed: {timedelta(seconds=int(elapsed))} | "
                    f"Processing FPS: {processing_fps:.2f} | "
                    f"ETA: {timedelta(seconds=int(eta_seconds))}",
                    flush=True,
                )
                if progress_path is not None:
                    _write_progress_sidecar(
                        progress_path, frame_index, total_frames, "running"
                    )
                if profiler is not None:
                    profiler.report(
                        frame_index,
                        total_frames,
                        elapsed,
                    )
            if (
                checkpoint_path is not None
                and frame_index % _PIPELINE_CHECKPOINT_INTERVAL == 0
                and frame_index < total_frames
            ):
                save_pipeline_checkpoint()
                if progress_path is not None:
                    _write_progress_sidecar(
                        progress_path, frame_index, total_frames, "running"
                    )
        for ball_result in ball_tracker.finish():
            output_context = pending_output_frames.pop(
                ball_result["frame_number"], None
            )
            if output_context is None:
                raise RuntimeError(
                    "Ball tracker finalized a frame without buffered output "
                    f"data: {ball_result['frame_number']}"
                )
            emit_finalized_frame(output_context, ball_result)
        if pending_output_frames:
            raise RuntimeError(
                "Output frames remained buffered after ball tracking ended: "
                f"{sorted(pending_output_frames)}"
            )
    finally:
        cap.release()
        finalization_started = perf_counter() if profiler is not None else 0.0
        writer.release()
        observations_file.close()
        identity_evidence_file.close()
        pitch_rejections_file.close()
        if identity_transition_file is not None:
            identity_transition_file.close()
        cv2.destroyAllWindows()
        if stub_writer is not None:
            stub_writer.close()
        if stub_reader is not None:
            stub_reader.close()
        if analyzer is not None:
            diagnostics_started = (
                perf_counter() if profiler is not None else 0.0
            )
            analyzer.finish(frame_index)
            if profiler is not None:
                profiler.record(
                    "Phase 1 diagnostics",
                    perf_counter() - diagnostics_started,
                )
        for target, method_name, original in reversed(
            tracker_profile_hooks
        ):
            setattr(target, method_name, original)
        if profiler is not None:
            profiler.record(
                "Output finalization",
                perf_counter() - finalization_started,
            )

    if checkpoint_path is not None:
        _assemble_video_segments(
            sorted(segments_dir.glob("segment_*.mp4")),
            output_path,
            args.ffmpeg_executable,
        )
        shutil.copyfile(resume_observations_path, run_output_dir / "observations.csv")
        shutil.copyfile(
            identity_evidence_path,
            run_output_dir / "identity_association_evidence.csv",
        )
        shutil.copyfile(
            pitch_rejections_path,
            run_output_dir / "pitch_rejected_detections.csv",
        )

    if args.canonical_ids == "offline":
        reconcile_observations(
            run_output_dir / "observations.csv",
            run_output_dir / "offline_reconciliation",
            fps=fps,
            video_path=output_path,
        )
    elif args.canonical_ids == "online":
        team_reconciliation = reconcile_final_team_labels(
            run_output_dir / "observations.csv",
            video_path,
            output_path,
            fps=fps,
            ball_gate=args.ball_gate,
            layout=LAYOUT,
        )
        (run_output_dir / "final_team_reconciliation.json").write_text(
            json.dumps(team_reconciliation, indent=2) + "\n",
            encoding="utf-8",
        )

    analytics_summary = write_match_analytics(
        run_output_dir / "observations.csv",
        run_output_dir,
        fps,
        run_output_dir / "phase1_diagnostics" / "events.jsonl",
        image_width=frame_width,
    )
    summary_payload = {
        "schema_version": "1.0",
        "status": "succeeded",
        "artifacts": {
            "tracked_video": {"path": "tracked.mp4"},
            "observations_csv": {"path": "observations.csv"},
            "match_summary": {"path": "match_summary.json"},
            "passes_csv": {"path": "passes.csv"},
            "passing_network_csv": {"path": "passing_network.csv"},
        },
        "analytics": analytics_summary,
    }
    (run_output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    if progress_path is not None:
        _write_progress_sidecar(
            progress_path, frame_index, total_frames, "completed"
        )
    processing_elapsed = perf_counter() - processing_started
    average_processing_fps = (
        frame_index / processing_elapsed if processing_elapsed else 0.0
    )
    if profiler is not None:
        exclusive_stages = {
            "Video frame read",
            "Tracking cache read",
            "Camera motion compensation",
            "Model warm-up",
            "YOLO preprocessing",
            "YOLO inference",
            "YOLO postprocessing",
            "Pre-tracker filtering",
            "BoT-SORT update",
            "Predictor / callback overhead",
            "Tracker-ID / lifecycle bookkeeping",
            "Post-tracker referee / eligibility filtering",
            "Team classification",
            "Global identity assignment + appearance",
            "Ball tracking",
            "Player and ball observations",
            "Radar rendering",
            "Possession calculation",
            "Frame annotation + dashboard",
            "Observations CSV writing",
            "Phase 1 diagnostics",
            "Video writing",
            "Output finalization",
        }
        measured_seconds = sum(
            profiler.totals[stage] for stage in exclusive_stages
        )
        profiler.record(
            "Other loop / reporting overhead",
            max(0.0, processing_elapsed - measured_seconds),
        )
        print(
            "Warm-up time: "
            f"{profiler.totals['Model warm-up']:.3f} seconds "
            "(first predictor warm-up; included in frame 1 wall time)"
        )
        profiler.report(
            frame_index,
            total_frames,
            processing_elapsed,
            prefix="Final cumulative profile",
        )
        print(
            "Nested tracker detail (included in BoT-SORT update; "
            "do not add these again to total time):"
        )
        tracker_seconds = profiler.totals["BoT-SORT update"]
        for stage in (
            "ReID feature extraction + track initialization",
            "ReID / appearance association costs",
            "GMC optical-flow estimation",
        ):
            seconds = profiler.totals[stage]
            share = 100.0 * seconds / tracker_seconds if tracker_seconds else 0.0
            print(
                f"  {stage}: {seconds * 1000.0 / frame_index:.2f} "
                f"ms/frame ({share:.1f}% of tracker update)"
            )
        if profile_cuda:
            device = args.device if args.device is not None else 0
            print(
                "GPU memory:",
                {
                    "allocated_mb": round(
                        torch.cuda.memory_allocated(device) / (1024**2), 1
                    ),
                    "reserved_mb": round(
                        torch.cuda.memory_reserved(device) / (1024**2), 1
                    ),
                    "peak_allocated_mb": round(
                        torch.cuda.max_memory_allocated(device) / (1024**2), 1
                    ),
                    "peak_reserved_mb": round(
                        torch.cuda.max_memory_reserved(device) / (1024**2), 1
                    ),
                },
            )
    percentages = possession.percentages()
    print(
        "Processing summary:",
        {
            "total_frames": total_frames,
            "processed_frames": frame_index,
            "elapsed": str(timedelta(seconds=int(processing_elapsed))),
            "processing_seconds": round(processing_elapsed, 2),
            "average_processing_fps": round(average_processing_fps, 2),
            "output_path": str(output_path),
        },
    )
    print(
        "Analytics artifacts:",
        {
            "match_summary": str(run_output_dir / "match_summary.json"),
            "team_statistics": str(run_output_dir / "team_statistics.csv"),
            "player_statistics": str(run_output_dir / "player_statistics.csv"),
            "possession_sequences": str(
                run_output_dir / "possession_sequences.csv"
            ),
            "passes": str(run_output_dir / "passes.csv"),
            "passing_network": str(
                run_output_dir / "passing_network.csv"
            ),
            "summary": str(run_output_dir / "summary.json"),
            "tracking_events": analytics_summary["outputs"][
                "tracking_events"
            ],
            "passes_supported": analytics_summary["passes"]["available"],
            "football_events_supported": analytics_summary[
                "football_events"
            ]["available"],
        },
    )
    print("\nDetection class counts:", dict(class_counts_total))
    print(f"Total YOLO detections: {total_yolo_detections}")
    print(
        "Pre-tracker eligibility:",
        {
            "valid_player_detections": tracking_input_filter.valid_detection_counts[
                "player"
            ],
            "valid_goalkeeper_detections": tracking_input_filter.valid_detection_counts[
                "goalkeeper"
            ],
            "rejected_outside_pitch": player_filter_counts["outside_pitch"],
            "rejected_referee_overlap": player_filter_counts[
                "overlapping_referee"
            ],
            "rejected_pitch_mask": player_filter_counts["pitch_mask"],
            "referee_class_detections_reaching_tracker": 0,
            "other_class_detections_reaching_tracker": 0,
        },
    )
    print(
        "Referee track-frame counts:",
        {
            "detected_by_class": referee_frame_counts["detected_by_class"],
            "reclassified_by_class_history": referee_frame_counts[
                "reclassified_by_class_history"
            ],
            "rejected_by_kit_color": referee_frame_counts[
                "rejected_by_kit_color"
            ],
        },
    )
    print(
        "Camera radar frames:",
        {
            "mode": RADAR_MODE,
            "rendered": camera_radar_frames,
        },
    )
    print(
        "Player filter counts:",
        {
            "outside_pitch": player_filter_counts["outside_pitch"],
            "overlapping_referee": player_filter_counts["overlapping_referee"],
            "known_referee_duplicate": player_filter_counts[
                "known_referee_duplicate"
            ],
            "pitch_mask": player_filter_counts["pitch_mask"],
        },
    )
    print(
        f"Tracking audit: unique_active_tracker_ids="
        f"{len(unique_player_tracker_ids)}, "
        f"active_ids_last_frame={len(active_player_tracker_ids)}, "
        f"max_active_ids={max_active_tracker_ids}, "
        f"rendered_tracker_ids={len(player_tracker_ids)}, "
        f"max_simultaneous_players={max_simultaneous_players}, "
        f"team_label_changes={team_assigner.team_label_change_count}, "
        f"team_label_changed_ids={sorted(team_assigner.team_label_changed_ids)}"
    )
    print(
        "Global identity audit:",
        {
            **global_identity_manager.snapshot(),
            "unique_global_ids_rendered": len(unique_global_player_ids),
            "active_global_ids_last_frame": len(active_global_player_ids),
        },
    )
    print(
        "Tracker lifecycle:",
        {
            "lost_track_events": track_loss_events,
            "lost_track_reacquisitions": track_reacquisition_events,
            "physical_track_fragments": "not measurable without identity ground truth",
            "verified_id_switches": "not measurable from tracker IDs alone",
            "large_displacement_candidates": len(large_displacement_events),
        },
    )
    print(
        "Large displacement candidates (first 20):",
        large_displacement_events[:20],
    )
    print(
        "Referee detection filtering:",
        {
            "detections": referee_frame_counts["detected_by_class"],
            "class_referee_tracker_ids": len(
                referee_manager.class_detected_track_ids
            ),
            "reclassified_player_tracker_ids": sum(
                reason == "reclassified_by_class_history"
                for reason in referee_manager.referee_reason.values()
            ),
        },
    )
    print(
        f"Ball position frames: real={ball_tracker.real_frames}, "
        f"interpolated={ball_tracker.interpolated_frames}, "
        f"missing={ball_tracker.unknown_frames}"
    )
    print(
        f"Goalkeeper tracks={len(goalkeeper_track_ids)}, "
        f"neutral goalkeepers={len(unknown_goalkeeper_ids)}, "
        f"unknown players={len(unknown_player_ids)}"
    )
    print(
        f"Possession percentages (all {possession.processed_frames} frames): "
        f"Team 1={percentages['team_1']:.1f}%, "
        f"Team 2={percentages['team_2']:.1f}%, "
        f"Loose={percentages['loose']:.1f}%, "
        f"Unknown={percentages['unknown']:.1f}%"
    )
    team_control = possession.team_control_percentages()
    print(
        "Team possession (confirmed outfield-player possession only): "
        f"Team 1={team_control['team_1']:.1f}%, "
        f"Team 2={team_control['team_2']:.1f}%"
    )
    print(f"Processing complete. Video saved to {output_path}")


if __name__ == "__main__":
    main()
