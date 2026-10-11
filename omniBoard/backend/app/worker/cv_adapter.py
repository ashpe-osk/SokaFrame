import ctypes
import hashlib
import json
import logging
import math
import os
import re
import shutil
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from threading import Event, Thread
from typing import BinaryIO, Literal
from uuid import UUID

from sqlalchemy.engine import Engine
from sqlmodel import Session

from app.core.config import Settings, settings
from app.models import Analysis, Artifact, ArtifactKind, Video
from app.worker.manager import (
    AnalysisJob,
    ProcessingCancelled,
    WorkerShutdownRequested,
)

logger = logging.getLogger(__name__)
_LOG_CHUNK_BYTES = 64 * 1024
_MAX_LOG_BYTES = 10 * 1024 * 1024
_LOG_TRUNCATION_MARKER = b"\n...[log truncated at 10 MiB]...\n"
_POLL_INTERVAL_SECONDS = 0.1
_PROGRESS_POLL_INTERVAL_SECONDS = 1.0
_PROGRESS_READ_RETRY_SECONDS = 0.1
_PROGRESS_READ_RETRY_ATTEMPTS = 10
_RUN_DIRECTORY_PATTERN = re.compile(r"run_[1-9][0-9]*\Z")


class CVAdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class CVAdapterConfig:
    python_executable: Path
    repo_root: Path
    entrypoint: Path
    default_model: Path
    termination_grace_seconds: float
    default_backend: Literal["auto", "cuda", "openvino", "cpu"] = "auto"
    canonical_ids: Literal["off", "online", "offline"] = "online"
    ball_gate: bool = True
    team_smoothing: bool = True

    @classmethod
    def from_settings(cls, configuration: Settings = settings) -> CVAdapterConfig:
        if configuration.CV_PYTHON_EXECUTABLE is None:
            raise CVAdapterError(
                "CV_PYTHON_EXECUTABLE must be set to the CV environment's Python."
            )
        if configuration.CV_REPO_ROOT is None:
            raise CVAdapterError("CV_REPO_ROOT must be set to the CV repository.")
        return cls(
            python_executable=configuration.CV_PYTHON_EXECUTABLE,
            repo_root=configuration.CV_REPO_ROOT,
            entrypoint=configuration.CV_ENTRYPOINT,
            default_model=configuration.CV_DEFAULT_MODEL,
            termination_grace_seconds=configuration.CV_TERMINATION_GRACE_SECONDS,
            default_backend=configuration.CV_DEFAULT_BACKEND,
            canonical_ids=configuration.CV_CANONICAL_IDS,
            ball_gate=configuration.CV_BALL_GATE,
            team_smoothing=configuration.CV_TEAM_SMOOTHING,
        )


@dataclass(frozen=True)
class CVProcessResult:
    return_code: int
    process_id: int
    output_root: Path
    run_directory: Path
    progress_json: Path
    tracked_video: Path
    observations_csv: Path
    summary_json: Path | None
    stdout_log: Path
    stderr_log: Path

    @property
    def has_supported_summary(self) -> bool:
        return self.summary_json is not None and self.summary_json.is_file()


class _WindowsProcessJob:
    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        from ctypes import wintypes

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.INT,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        self._kernel32 = kernel32
        self._handle = kernel32.CreateJobObjectW(None, None)
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())

        limits = ExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = 0x00002000
        if not kernel32.SetInformationJobObject(
            self._handle,
            9,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

        process_handle = kernel32.OpenProcess(0x0101, False, process.pid)
        if not process_handle:
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error
        try:
            if not kernel32.AssignProcessToJobObject(self._handle, process_handle):
                error = ctypes.WinError(ctypes.get_last_error())
                self.close()
                raise error
        finally:
            kernel32.CloseHandle(process_handle)

    def terminate(self, exit_code: int = 1) -> None:
        if self._handle and not self._kernel32.TerminateJobObject(
            self._handle, exit_code
        ):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        if self._handle:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


def _validated_absolute_path(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise CVAdapterError(f"{label} must be an absolute path: {path}")
    return path.expanduser().resolve()


def _resolve_configured_file(
    configured_path: Path, repo_root: Path, label: str
) -> Path:
    path = configured_path.expanduser()
    if not path.is_absolute():
        path = repo_root / path
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise CVAdapterError(f"Configured {label} does not exist: {path}") from exc
    if not resolved.is_file():
        raise CVAdapterError(f"Configured {label} is not a file: {resolved}")
    return resolved


def _capture_log(
    stream: BinaryIO, log_path: Path, errors: list[OSError]
) -> None:
    log_file = None
    try:
        log_file = log_path.open("wb")
    except OSError as exc:
        errors.append(exc)

    written = 0
    truncated = False
    payload_limit = _MAX_LOG_BYTES - len(_LOG_TRUNCATION_MARKER)
    try:
        while chunk := stream.read(_LOG_CHUNK_BYTES):
            if log_file is None:
                continue
            truncated_chunk = False
            if written < payload_limit:
                remaining = payload_limit - written
                payload = chunk[:remaining]
                truncated_chunk = len(payload) < len(chunk)
                try:
                    log_file.write(payload)
                    written += len(payload)
                except OSError as exc:
                    errors.append(exc)
                    log_file.close()
                    log_file = None
                    continue
            elif chunk:
                truncated_chunk = True
            if truncated_chunk and not truncated:
                try:
                    log_file.write(_LOG_TRUNCATION_MARKER)
                    truncated = True
                except OSError as exc:
                    errors.append(exc)
                    log_file.close()
                    log_file = None
    finally:
        stream.close()
        if log_file is not None:
            try:
                log_file.close()
            except OSError as exc:
                errors.append(exc)


class CVProcessor:
    """Runs the independent CV CLI in a separate configured Python process."""

    def __init__(
        self,
        *,
        database_engine: Engine,
        data_dir: Path,
        configuration: CVAdapterConfig | None = None,
    ) -> None:
        self.engine = database_engine
        self.data_dir = data_dir
        self.configuration = configuration or CVAdapterConfig.from_settings()
        (
            self.python_executable,
            self.repo_root,
            self.entrypoint,
            self.default_model,
        ) = self._validate_configuration(self.configuration)
        self._validate_runtime_options(self.configuration)
        if (
            not math.isfinite(self.configuration.termination_grace_seconds)
            or self.configuration.termination_grace_seconds <= 0
        ):
            raise CVAdapterError("CV_TERMINATION_GRACE_SECONDS must be positive.")

    @staticmethod
    def _validate_runtime_options(configuration: CVAdapterConfig) -> None:
        if configuration.default_backend not in (
            "auto",
            "cuda",
            "openvino",
            "cpu",
        ):
            raise CVAdapterError(
                f"Unsupported CV_DEFAULT_BACKEND: {configuration.default_backend}"
            )
        if configuration.canonical_ids not in ("off", "online", "offline"):
            raise CVAdapterError(
                f"Unsupported CV_CANONICAL_IDS: {configuration.canonical_ids}"
            )
        if not isinstance(configuration.ball_gate, bool):
            raise CVAdapterError("CV_BALL_GATE must be a boolean.")
        if not isinstance(configuration.team_smoothing, bool):
            raise CVAdapterError("CV_TEAM_SMOOTHING must be a boolean.")
        if (
            configuration.canonical_ids != "online"
            and (configuration.ball_gate or configuration.team_smoothing)
        ):
            raise CVAdapterError(
                "CV_BALL_GATE and CV_TEAM_SMOOTHING require "
                "CV_CANONICAL_IDS=online."
            )

    @staticmethod
    def _validate_configuration(
        configuration: CVAdapterConfig,
    ) -> tuple[Path, Path, Path, Path]:
        executable = _validated_absolute_path(
            configuration.python_executable, "CV_PYTHON_EXECUTABLE"
        )
        if not executable.is_file():
            raise CVAdapterError(
                f"CV_PYTHON_EXECUTABLE does not exist: {executable}"
            )
        if os.name != "nt" and not os.access(executable, os.X_OK):
            raise CVAdapterError(
                f"CV_PYTHON_EXECUTABLE is not executable: {executable}"
            )

        repo_root = _validated_absolute_path(configuration.repo_root, "CV_REPO_ROOT")
        if not repo_root.is_dir():
            raise CVAdapterError(f"CV_REPO_ROOT is not a directory: {repo_root}")

        entrypoint = _resolve_configured_file(
            configuration.entrypoint, repo_root, "CV_ENTRYPOINT"
        )
        try:
            entrypoint.relative_to(repo_root)
        except ValueError as exc:
            raise CVAdapterError("CV_ENTRYPOINT must be inside CV_REPO_ROOT.") from exc

        default_model = _resolve_configured_file(
            configuration.default_model, repo_root, "CV_DEFAULT_MODEL"
        )
        return executable, repo_root, entrypoint, default_model

    def _data_root(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir.resolve(strict=True)

    @staticmethod
    def _ensure_contained(path: Path, root: Path, label: str) -> Path:
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise CVAdapterError(f"{label} must remain inside LOCAL_DATA_DIR.") from exc
        return resolved

    def _video_path(self, video_id: UUID) -> Path:
        with Session(self.engine) as session:
            video = session.get(Video, video_id)
            if video is None:
                raise CVAdapterError(f"Source video metadata is missing: {video_id}")
            storage_path = video.storage_path

        posix_path = PurePosixPath(storage_path)
        windows_path = PureWindowsPath(storage_path)
        if (
            not storage_path
            or posix_path.is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or ".." in posix_path.parts
            or ".." in windows_path.parts
        ):
            raise CVAdapterError("Stored source video path is not a safe relative path.")

        data_root = self._data_root()
        candidate = data_root.joinpath(*posix_path.parts)
        try:
            video_path = candidate.resolve(strict=True)
        except OSError as exc:
            raise CVAdapterError(
                "Stored source video file is missing or unreadable."
            ) from exc
        self._ensure_contained(video_path, data_root, "Source video path")
        if not video_path.is_file():
            raise CVAdapterError("Stored source video is not a regular file.")
        try:
            with video_path.open("rb"):
                pass
        except OSError as exc:
            raise CVAdapterError("Stored source video file is unreadable.") from exc
        return video_path

    def _analysis_paths(self, analysis_id: UUID) -> tuple[Path, Path, Path]:
        data_root = self._data_root()
        analysis_root = self._ensure_contained(
            data_root / "analyses" / str(analysis_id),
            data_root,
            "Analysis output path",
        )
        output_root = self._ensure_contained(
            analysis_root / "cv-output", data_root, "CV output path"
        )
        log_root = self._ensure_contained(
            analysis_root / "logs", data_root, "CV log path"
        )
        output_root.mkdir(parents=True, exist_ok=True)
        log_root.mkdir(parents=True, exist_ok=True)
        return output_root, log_root / "cv-stdout.log", log_root / "cv-stderr.log"

    def _command(
        self,
        video_path: Path,
        output_root: Path,
        progress_path: Path,
        checkpoint_path: Path,
    ) -> list[str]:
        command = [
            str(self.python_executable),
            str(self.entrypoint),
            "--backend",
            self.configuration.default_backend,
            "--canonical-ids",
            self.configuration.canonical_ids,
            "--video",
            str(video_path),
            "--model",
            str(self.default_model),
            "--output-dir",
            str(output_root),
            "--progress-file",
            str(progress_path),
            "--resume-checkpoint",
            str(checkpoint_path),
            "--ffmpeg-executable",
            settings.FFMPEG_EXECUTABLE,
        ]
        if self.configuration.ball_gate:
            command.append("--ball-gate")
        if self.configuration.team_smoothing:
            command.append("--team-smoothing")
        return command

    def _consume_progress(self, progress_path: Path, analysis_id: UUID) -> bool:
        for attempt in range(_PROGRESS_READ_RETRY_ATTEMPTS):
            try:
                with progress_path.open("r", encoding="utf-8") as handle:
                    progress = json.load(handle)
                break
            except FileNotFoundError:
                return False
            except PermissionError as exc:
                if attempt + 1 == _PROGRESS_READ_RETRY_ATTEMPTS:
                    raise CVAdapterError(
                        f"Could not read CV progress sidecar: {progress_path}"
                    ) from exc
                time.sleep(_PROGRESS_READ_RETRY_SECONDS)
            except (OSError, json.JSONDecodeError) as exc:
                raise CVAdapterError(
                    f"Could not read CV progress sidecar: {progress_path}"
                ) from exc
        if not isinstance(progress, dict) or progress.get("schema_version") != "1.0":
            raise CVAdapterError("CV progress sidecar has an unsupported schema.")
        status = progress.get("status")
        frames_processed = progress.get("frames_processed")
        total_frames = progress.get("total_frames")
        if (
            status not in ("running", "completed")
            or not isinstance(frames_processed, int)
            or isinstance(frames_processed, bool)
            or frames_processed < 0
            or (
                total_frames is not None
                and (
                    not isinstance(total_frames, int)
                    or isinstance(total_frames, bool)
                    or total_frames < frames_processed
                )
            )
        ):
            raise CVAdapterError("CV progress sidecar has invalid progress values.")

        with Session(self.engine) as session:
            analysis = session.get(Analysis, analysis_id)
            if analysis is not None and (
                analysis.frames_processed != frames_processed
                or analysis.total_frames != total_frames
            ):
                analysis.frames_processed = frames_processed
                analysis.total_frames = total_frames
                session.add(analysis)
                session.commit()
        return status == "completed"

    @staticmethod
    def _child_environment() -> dict[str, str]:
        environment = os.environ.copy()
        for key in (
            "DATABASE_URL",
            "SECRET_KEY",
            "FIRST_SUPERUSER",
            "FIRST_SUPERUSER_PASSWORD",
            "SMTP_HOST",
            "SMTP_USER",
            "SMTP_PASSWORD",
            "SENTRY_DSN",
            "PYTHONPATH",
            "PYTHONHOME",
            "VIRTUAL_ENV",
            "FOOTBALL_BACKEND",
            "FOOTBALL_DEVICE",
        ):
            environment.pop(key, None)
        return environment

    def _terminate(
        self,
        process: subprocess.Popen[bytes],
        windows_job: _WindowsProcessJob | None,
    ) -> int:
        if os.name == "nt":
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT)
            except (OSError, ValueError):
                logger.warning(
                    "CV graceful process-group signal failed process_id=%s",
                    process.pid,
                    exc_info=True,
                )
            try:
                return_code = process.wait(
                    timeout=self.configuration.termination_grace_seconds
                )
            except subprocess.TimeoutExpired:
                return_code = None
            if windows_job is not None:
                windows_job.terminate()
            elif return_code is None:
                process.kill()
            if return_code is None:
                return_code = process.wait()
            return return_code

        try:
            self._signal_process_group(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            return_code = process.wait(
                timeout=self.configuration.termination_grace_seconds
            )
        except subprocess.TimeoutExpired:
            return_code = None
        force_signal = getattr(signal, "SIGKILL", None)
        if force_signal is None:
            raise CVAdapterError("SIGKILL is not available for process-group cleanup.")
        try:
            self._signal_process_group(process.pid, force_signal)
        except ProcessLookupError:
            pass
        if return_code is None:
            return_code = process.wait()
        return return_code

    @staticmethod
    def _signal_process_group(process_id: int, signal_number: int) -> None:
        kill_process_group = getattr(os, "killpg", None)
        if kill_process_group is None:
            raise CVAdapterError("Process-group signaling is unavailable.")
        kill_process_group(process_id, signal_number)

    def _run_directories(self, output_root: Path) -> set[Path]:
        return {
            path.resolve()
            for path in output_root.iterdir()
            if path.is_dir() and _RUN_DIRECTORY_PATTERN.fullmatch(path.name)
        }

    def _validate_outputs(
        self, output_root: Path, before: set[Path]
    ) -> tuple[Path, Path, Path, Path | None]:
        try:
            new_runs = [
                path.resolve(strict=True)
                for path in output_root.iterdir()
                if path.is_dir()
                and _RUN_DIRECTORY_PATTERN.fullmatch(path.name)
                and path.resolve() not in before
            ]
        except OSError as exc:
            raise CVAdapterError(f"Could not inspect CV output directory: {exc}") from exc
        if len(new_runs) != 1:
            raise CVAdapterError(
                "CV process did not create exactly one new run directory "
                "for this analysis."
            )

        run_directory = self._ensure_contained(
            new_runs[0], output_root.resolve(), "CV run directory"
        )
        tracked_video = run_directory / "tracked.mp4"
        observations_csv = run_directory / "observations.csv"
        summary_json = run_directory / "summary.json"
        for output_file in (tracked_video, observations_csv):
            self._ensure_contained(
                output_file, run_directory, f"{output_file.name} path"
            )
            try:
                has_content = output_file.is_file() and output_file.stat().st_size > 0
            except OSError as exc:
                raise CVAdapterError(
                    f"Could not inspect CV output {output_file.name}: {exc}"
                ) from exc
            if not has_content:
                raise CVAdapterError(
                    f"CV process did not create a non-empty {output_file.name}."
                )

        if summary_json.exists():
            if not summary_json.is_file():
                raise CVAdapterError("CV summary path exists but is not a file.")
            if summary_json.stat().st_size <= 0:
                raise CVAdapterError("CV summary file is empty.")
        return run_directory, tracked_video, observations_csv, summary_json

    def _ensure_summary_json(
        self,
        run_directory: Path,
        tracked_video: Path,
        observations_csv: Path,
        summary_json: Path | None,
    ) -> Path:
        if summary_json is not None and summary_json.exists():
            if not summary_json.is_file():
                raise CVAdapterError("CV summary path exists but is not a file.")
            if summary_json.stat().st_size <= 0:
                raise CVAdapterError("CV summary file is empty.")
            return summary_json

        summary_json = run_directory / "summary.json"
        payload = {
            "schema_version": "1.0",
            "status": "succeeded",
            "artifacts": {
                "tracked_video": {
                    "path": tracked_video.name,
                    "sha256": self._sha256(tracked_video),
                },
                "observations_csv": {
                    "path": observations_csv.name,
                    "sha256": self._sha256(observations_csv),
                },
            },
        }
        try:
            summary_json.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
        except OSError as exc:
            raise CVAdapterError(f"Could not write summary.json: {exc}") from exc
        return summary_json

    @staticmethod
    def _relative_to_data_root(path: Path, data_root: Path) -> str:
        try:
            relative = path.relative_to(data_root)
        except ValueError as exc:
            raise CVAdapterError(
                f"Path {path} is outside the configured local data root {data_root}."
            ) from exc
        return relative.as_posix()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _persist_artifacts(self, analysis_id: UUID, result: CVProcessResult) -> None:
        data_root = self._data_root()
        summary_path = result.summary_json
        artifact_records: list[tuple[ArtifactKind, Path, str]] = [
            (ArtifactKind.TRACKED_VIDEO, result.tracked_video, "video/mp4"),
            (
                ArtifactKind.BROWSER_VIDEO,
                result.run_directory / "browser.mp4",
                "video/mp4",
            ),
            (ArtifactKind.OBSERVATIONS_CSV, result.observations_csv, "text/csv"),
        ]
        if summary_path is not None and summary_path.exists():
            artifact_records.append((ArtifactKind.SUMMARY, summary_path, "application/json"))
        analytics_artifacts = (
            (
                ArtifactKind.MATCH_SUMMARY,
                "match_summary.json",
                "application/json",
            ),
            (ArtifactKind.PASSES_CSV, "passes.csv", "text/csv"),
            (
                ArtifactKind.PASSING_NETWORK_CSV,
                "passing_network.csv",
                "text/csv",
            ),
            (
                ArtifactKind.TEAM_STATISTICS_CSV,
                "team_statistics.csv",
                "text/csv",
            ),
            (
                ArtifactKind.PLAYER_STATISTICS_CSV,
                "player_statistics.csv",
                "text/csv",
            ),
            (
                ArtifactKind.POSSESSION_SEQUENCES_CSV,
                "possession_sequences.csv",
                "text/csv",
            ),
        )
        for kind, filename, media_type in analytics_artifacts:
            candidate = result.run_directory / filename
            if not candidate.exists():
                continue
            path = self._ensure_contained(
                candidate, result.run_directory, f"{filename} path"
            )
            if not path.is_file() or path.stat().st_size <= 0:
                raise CVAdapterError(
                    f"Optional analytics artifact is present but empty: {filename}"
                )
            artifact_records.append((kind, path, media_type))

        with Session(self.engine) as session:
            analysis = session.get(Analysis, analysis_id)
            if analysis is None:
                return
            if analysis.output_dir is None:
                analysis.output_dir = self._relative_to_data_root(result.output_root, data_root)
            for kind, path, media_type in artifact_records:
                if not path.is_file() or path.stat().st_size <= 0:
                    raise CVAdapterError(f"Artifact is missing or empty: {path}")
                artifact = Artifact(
                    analysis_id=analysis_id,
                    kind=kind,
                    storage_path=self._relative_to_data_root(path, data_root),
                    media_type=media_type,
                    file_size_bytes=path.stat().st_size,
                    sha256=self._sha256(path),
                )
                session.add(artifact)
            session.add(analysis)
            session.commit()

    @staticmethod
    def _read_summary(path: Path) -> dict[str, object]:
        if not path.exists():
            raise CVAdapterError(f"Missing summary file: {path}")
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise CVAdapterError(f"Summary is not valid JSON: {path}") from exc
        if not isinstance(data, dict):
            raise CVAdapterError("Summary JSON must be an object.")
        return data

    @staticmethod
    def _validate_summary(summary: dict[str, object]) -> None:
        schema_version = summary.get("schema_version")
        if schema_version not in ("1", 1, "1.0", 1.0):
            raise CVAdapterError("Summary JSON has unsupported schema_version.")
        required_top_level = {"status", "artifacts"}
        missing = sorted(required_top_level - set(summary))
        if missing:
            raise CVAdapterError(f"Summary JSON missing required fields: {missing}")

    @staticmethod
    def _prepare_browser_video(source: Path) -> Path:
        ffmpeg = shutil.which(settings.FFMPEG_EXECUTABLE)
        ffprobe = shutil.which(settings.FFPROBE_EXECUTABLE)
        if ffmpeg is None or ffprobe is None:
            raise CVAdapterError(
                "FFmpeg and ffprobe are required to prepare browser-compatible video. "
                "Install FFmpeg and add it to PATH, or configure "
                "FFMPEG_EXECUTABLE and FFPROBE_EXECUTABLE."
            )

        output = source.with_name("browser.mp4")
        temporary_output = source.with_name("browser.tmp.mp4")
        try:
            conversion = subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-profile:v",
                    "baseline",
                    "-level:v",
                    "5.0",
                    "-pix_fmt",
                    "yuv420p",
                    "-preset",
                    "fast",
                    "-crf",
                    "23",
                    "-movflags",
                    "+faststart",
                    str(temporary_output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if conversion.returncode != 0:
                raise CVAdapterError(
                    "FFmpeg could not create browser-compatible video: "
                    f"{conversion.stderr.strip() or conversion.returncode}"
                )

            probe = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,codec_tag_string,pix_fmt",
                    "-of",
                    "json",
                    str(temporary_output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if probe.returncode != 0:
                raise CVAdapterError(
                    "ffprobe could not validate browser-compatible video: "
                    f"{probe.stderr.strip() or probe.returncode}"
                )
            streams = json.loads(probe.stdout).get("streams", [])
            if not streams or (
                streams[0].get("codec_name") != "h264"
                or streams[0].get("codec_tag_string") != "avc1"
                or streams[0].get("pix_fmt") != "yuv420p"
            ):
                raise CVAdapterError(
                    "FFmpeg output is not browser-compatible H.264/AVC video."
                )
            if not temporary_output.is_file() or temporary_output.stat().st_size <= 0:
                raise CVAdapterError("FFmpeg produced an empty browser video.")
            temporary_output.replace(output)
        except (OSError, json.JSONDecodeError, AttributeError, TypeError) as exc:
            raise CVAdapterError(
                f"Could not prepare browser-compatible video: {exc}"
            ) from exc
        finally:
            temporary_output.unlink(missing_ok=True)

        return output

    def finalize_success(self, analysis_id: UUID, result: CVProcessResult) -> None:
        summary_path = result.summary_json
        if summary_path is None:
            raise CVAdapterError("CV process succeeded but no summary.json was produced.")
        summary = self._read_summary(summary_path)
        self._validate_summary(summary)
        required_artifacts = {
            "tracked.mp4": result.tracked_video,
            "observations.csv": result.observations_csv,
            "summary.json": summary_path,
        }
        for label, path in required_artifacts.items():
            if not path.is_file() or path.stat().st_size <= 0:
                raise CVAdapterError(f"Required artifact {label} is missing or empty.")
        self._prepare_browser_video(result.tracked_video)

        with Session(self.engine) as session:
            analysis = session.get(Analysis, analysis_id)
            if analysis is not None:
                analysis.summary = summary
                analysis.output_dir = (
                    self._relative_to_data_root(result.output_root, self._data_root())
                    if analysis.output_dir is None
                    else analysis.output_dir
                )
                analysis.finished_at = datetime.now(UTC)
                session.add(analysis)
                session.commit()
        self._persist_artifacts(analysis_id, result)

    def _artifact_record(self, analysis_id: UUID, kind: ArtifactKind, path: Path) -> Artifact:
        return Artifact(
            analysis_id=analysis_id,
            kind=kind,
            storage_path=self._relative_to_data_root(path, self._data_root()),
            media_type={
                ArtifactKind.TRACKED_VIDEO: "video/mp4",
                ArtifactKind.BROWSER_VIDEO: "video/mp4",
                ArtifactKind.OBSERVATIONS_CSV: "text/csv",
                ArtifactKind.SUMMARY: "application/json",
                ArtifactKind.LOG: "text/plain",
            }.get(kind, "application/octet-stream"),
            file_size_bytes=path.stat().st_size,
            sha256=self._sha256(path),
        )

    def process(
        self,
        job: AnalysisJob,
        cancellation_requested: Callable[[], bool],
        stop_event: Event,
    ) -> CVProcessResult:
        video_path = self._video_path(job.video_id)
        output_root, stdout_log, stderr_log = self._analysis_paths(job.id)
        progress_path = self._ensure_contained(
            output_root / "progress.json", output_root, "CV progress path"
        )
        checkpoint_path = self._ensure_contained(
            output_root / "pipeline.checkpoint",
            output_root,
            "CV checkpoint path",
        )
        try:
            progress_path.unlink(missing_ok=True)
        except OSError as exc:
            raise CVAdapterError(
                f"Could not remove stale CV progress sidecar: {progress_path}"
            ) from exc
        existing_runs = self._run_directories(output_root)
        command = self._command(
            video_path, output_root, progress_path, checkpoint_path
        )
        logger.info(
            "Starting CV process analysis_id=%s worker_id=%s cv_python=%s "
            "cv_repository=%s output_root=%s backend=%s",
            job.id,
            job.worker_id,
            self.python_executable,
            self.repo_root,
            output_root,
            self.configuration.default_backend,
        )

        creation_flags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            if os.name == "nt"
            else 0
        )
        try:
            process = subprocess.Popen(
                command,
                cwd=self.repo_root,
                env=self._child_environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                creationflags=creation_flags,
                start_new_session=os.name != "nt",
            )
        except OSError as exc:
            raise CVAdapterError(
                f"Could not start the configured CV process: {exc}"
            ) from exc

        windows_job = None
        if os.name == "nt":
            try:
                windows_job = _WindowsProcessJob(process)
            except OSError as exc:
                process.kill()
                process.wait()
                raise CVAdapterError(
                    f"Could not assign the CV process to a Windows job: {exc}"
                ) from exc

        if process.stdout is None or process.stderr is None:
            process.kill()
            process.wait()
            if windows_job is not None:
                windows_job.close()
            raise CVAdapterError("CV process output streams could not be captured.")

        log_errors: list[OSError] = []
        stdout_thread = Thread(
            target=_capture_log,
            args=(process.stdout, stdout_log, log_errors),
            name=f"cv-stdout-{job.id}",
        )
        stderr_thread = Thread(
            target=_capture_log,
            args=(process.stderr, stderr_log, log_errors),
            name=f"cv-stderr-{job.id}",
        )
        stdout_thread.start()
        stderr_thread.start()
        termination_reason: str | None = None
        next_progress_poll = 0.0
        try:
            while True:
                return_code = process.poll()
                if return_code is not None:
                    self._consume_progress(progress_path, job.id)
                    break
                now = time.monotonic()
                if now >= next_progress_poll:
                    self._consume_progress(progress_path, job.id)
                    next_progress_poll = now + _PROGRESS_POLL_INTERVAL_SECONDS
                if stop_event.is_set():
                    termination_reason = "worker_shutdown"
                    self._terminate(process, windows_job)
                    raise WorkerShutdownRequested
                if cancellation_requested():
                    termination_reason = "cancelled"
                    self._terminate(process, windows_job)
                    raise ProcessingCancelled
                stop_event.wait(_POLL_INTERVAL_SECONDS)
        except BaseException:
            if process.poll() is None:
                try:
                    self._terminate(process, windows_job)
                except Exception:
                    logger.exception(
                        "CV termination failed; forcing process cleanup "
                        "analysis_id=%s process_id=%s",
                        job.id,
                        process.pid,
                    )
                    if windows_job is not None:
                        windows_job.terminate()
                    elif os.name != "nt":
                        force_signal = getattr(signal, "SIGKILL", None)
                        if force_signal is None:
                            raise CVAdapterError(
                                "SIGKILL is not available for process-group cleanup."
                            )
                        try:
                            self._signal_process_group(process.pid, force_signal)
                        except ProcessLookupError:
                            pass
                    else:
                        process.kill()
                    process.wait()
            raise
        finally:
            if windows_job is not None:
                windows_job.close()
            elif os.name != "nt":
                force_signal = getattr(signal, "SIGKILL", None)
                if force_signal is None:
                    raise CVAdapterError(
                        "SIGKILL is not available for process-group cleanup."
                    )
                try:
                    self._signal_process_group(process.pid, force_signal)
                except ProcessLookupError:
                    pass
            stdout_thread.join()
            stderr_thread.join()

        if log_errors:
            raise CVAdapterError(
                f"Could not write CV process logs: {log_errors[0]}"
            ) from log_errors[0]

        log_fields = (
            f"analysis_id={job.id} worker_id={job.worker_id} "
            f"process_id={process.pid} return_code={return_code} "
            f"termination_reason={termination_reason or 'exited'} "
            f"output_root={output_root}"
        )
        if return_code != 0:
            logger.error("CV process failed %s", log_fields)
            raise CVAdapterError(
                f"CV process exited with return code {return_code}; "
                f"see {stderr_log}."
            )
        progress = self._consume_progress(progress_path, job.id)
        if not progress:
            raise CVAdapterError(
                "CV process exited successfully without a completed progress sidecar."
            )

        run_directory, tracked_video, observations_csv, summary_json = self._validate_outputs(
            output_root, existing_runs
        )
        summary_json = self._ensure_summary_json(
            run_directory, tracked_video, observations_csv, summary_json
        )
        logger.info("CV process completed %s", log_fields)
        result = CVProcessResult(
            return_code=return_code,
            process_id=process.pid,
            output_root=output_root,
            run_directory=run_directory,
            progress_json=progress_path,
            tracked_video=tracked_video,
            observations_csv=observations_csv,
            summary_json=summary_json,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
        )
        self.finalize_success(job.id, result)
        return result
