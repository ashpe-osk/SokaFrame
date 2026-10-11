import ctypes
import json
import os
import subprocess
import sys
import textwrap
import time
import uuid
from collections.abc import Generator
from pathlib import Path
from subprocess import Popen as PopenType
from threading import Event, Thread
from typing import Any, cast

import pytest
from pydantic import ValidationError
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, select

from app.core.config import Settings, settings
from app.core.database import create_database_engine
from app.models import Analysis, AnalysisStatus, Artifact, ArtifactKind, Video
from app.worker.cv_adapter import (
    _LOG_TRUNCATION_MARKER,
    _MAX_LOG_BYTES,
    CVAdapterConfig,
    CVAdapterError,
    CVProcessor,
)
from app.worker.manager import AnalysisJob, LocalJobManager, ProcessingCancelled

_FAKE_CV_SCRIPT = """
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--progress-file", required=True)
parser.add_argument("--resume-checkpoint", required=True)
parser.add_argument("--ffmpeg-executable", required=True)
parser.add_argument("--backend", choices=("auto", "cuda", "openvino", "cpu"), required=True)
parser.add_argument("--canonical-ids", choices=("off", "online", "offline"), required=True)
parser.add_argument("--ball-gate", action="store_true")
parser.add_argument("--team-smoothing", action="store_true")
args = parser.parse_args()
progress_path = Path(args.progress_file)
progress_path.write_text(json.dumps({
    "schema_version": "1.0",
    "status": "running",
    "frames_processed": 1,
    "total_frames": 10,
}))
print("captured stdout")
print("captured stderr", file=sys.stderr)
mode = os.environ.get("FAKE_CV_MODE", "success")
if mode == "failure":
    sys.exit(7)
if mode in ("sleep", "ignore_signal"):
    if mode == "ignore_signal":
        import signal
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    pid_file = os.environ.get("FAKE_CV_CHILD_PID_FILE")
    if pid_file:
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"]
        )
        Path(pid_file).write_text(str(child.pid))
    time.sleep(30)
run_directory = Path(args.output_dir) / "run_1"
run_directory.mkdir(parents=True)
if mode != "missing":
    if mode == "empty":
        (run_directory / "tracked.mp4").touch()
        (run_directory / "observations.csv").touch()
    else:
        (run_directory / "tracked.mp4").write_bytes(b"video")
        (run_directory / "observations.csv").write_bytes(b"frame_number\\n1\\n")
        (run_directory / "match_summary.json").write_text("{}")
        for name in (
            "passes.csv",
            "passing_network.csv",
            "team_statistics.csv",
            "player_statistics.csv",
            "possession_sequences.csv",
        ):
            (run_directory / name).write_text("header\\n")
if mode == "large_log":
    sys.stdout.write("x" * (12 * 1024 * 1024))
progress_path.write_text(json.dumps({
    "schema_version": "1.0",
    "status": "completed",
    "frames_processed": 10,
    "total_frames": 10,
}))
"""


@pytest.fixture
def adapter_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[tuple[CVProcessor, Engine, Path, uuid.UUID, Path]]:
    repository = tmp_path / "cv repository with spaces"
    repository.mkdir()
    entrypoint = repository / "fake cv entrypoint.py"
    entrypoint.write_text(textwrap.dedent(_FAKE_CV_SCRIPT), encoding="utf-8")
    model = repository / "model file.pt"
    model.write_bytes(b"test model")
    data_dir = tmp_path / "local data with spaces"
    video_path = data_dir / "uploads" / "source video.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"test video")
    engine = create_database_engine(
        f"sqlite:///{(tmp_path / 'adapter-test.db').as_posix()}"
    )
    SQLModel.metadata.create_all(engine)
    video_id = uuid.uuid4()
    with Session(engine) as session:
        session.add(
            Video(
                id=video_id,
                original_filename="source video.mp4",
                storage_path="uploads/source video.mp4",
                file_size_bytes=video_path.stat().st_size,
            )
        )
        session.commit()

    processor = CVProcessor(
        database_engine=engine,
        data_dir=data_dir,
        configuration=CVAdapterConfig(
            python_executable=Path(sys.executable),
            repo_root=repository,
            entrypoint=entrypoint,
            default_model=model,
            termination_grace_seconds=0.2,
        ),
    )

    def prepare_browser_video(source: Path) -> Path:
        browser_video = source.with_name("browser.mp4")
        browser_video.write_bytes(source.read_bytes())
        return browser_video

    monkeypatch.setattr(processor, "_prepare_browser_video", prepare_browser_video)
    yield processor, engine, data_dir, video_id, repository
    engine.dispose()


def _job(video_id: uuid.UUID, analysis_id: uuid.UUID | None = None) -> AnalysisJob:
    return AnalysisJob(
        id=analysis_id or uuid.uuid4(),
        video_id=video_id,
        options={},
        worker_id=uuid.uuid4(),
    )


def test_settings_reject_incompatible_cv_options() -> None:
    configuration = settings.model_dump()
    configuration["CV_CANONICAL_IDS"] = "offline"

    with pytest.raises(
        ValidationError,
        match="CV_BALL_GATE and CV_TEAM_SMOOTHING require CV_CANONICAL_IDS=online",
    ):
        Settings.model_validate(configuration)


def test_adapter_config_maps_approved_settings() -> None:
    expected = {
        "CV_REPO_ROOT": Path("C:/cv-repo"),
        "CV_PYTHON_EXECUTABLE": Path("C:/cv-repo/.venv/Scripts/python.exe"),
        "CV_ENTRYPOINT": Path("main.py"),
        "CV_DEFAULT_MODEL": Path("models/best.pt"),
        "CV_DEFAULT_BACKEND": "auto",
        "CV_CANONICAL_IDS": "online",
        "CV_BALL_GATE": True,
        "CV_TEAM_SMOOTHING": True,
    }
    adapter_config = CVAdapterConfig.from_settings(
        settings.model_copy(update=expected)
    )

    assert adapter_config.repo_root == expected["CV_REPO_ROOT"]
    assert adapter_config.python_executable == expected["CV_PYTHON_EXECUTABLE"]
    assert adapter_config.entrypoint == expected["CV_ENTRYPOINT"]
    assert adapter_config.default_model == expected["CV_DEFAULT_MODEL"]
    assert adapter_config.default_backend == "auto"
    assert adapter_config.canonical_ids == "online"
    assert adapter_config.ball_gate is True
    assert adapter_config.team_smoothing is True


def test_config_rejects_missing_executable_and_entrypoint(
    adapter_environment: tuple[CVProcessor, Engine, Path, uuid.UUID, Path],
    tmp_path: Path,
) -> None:
    processor, engine, data_dir, _, repository = adapter_environment
    configuration = processor.configuration
    with pytest.raises(CVAdapterError, match="CV_PYTHON_EXECUTABLE does not exist"):
        CVProcessor(
            database_engine=engine,
            data_dir=data_dir,
            configuration=CVAdapterConfig(
                python_executable=tmp_path / "missing-python.exe",
                repo_root=repository,
                entrypoint=configuration.entrypoint,
                default_model=configuration.default_model,
                termination_grace_seconds=1,
            ),
        )

    with pytest.raises(CVAdapterError, match="CV_ENTRYPOINT does not exist"):
        CVProcessor(
            database_engine=engine,
            data_dir=data_dir,
            configuration=CVAdapterConfig(
                python_executable=Path(sys.executable),
                repo_root=repository,
                entrypoint=Path("missing.py"),
                default_model=configuration.default_model,
                termination_grace_seconds=1,
            ),
        )


def test_config_requires_repo_and_model_paths(
    adapter_environment: tuple[CVProcessor, Engine, Path, uuid.UUID, Path],
) -> None:
    processor, engine, data_dir, _, repository = adapter_environment
    with pytest.raises(CVAdapterError, match="CV_REPO_ROOT must be set"):
        CVAdapterConfig.from_settings(
            settings.model_copy(
                update={"CV_REPO_ROOT": None, "CV_PYTHON_EXECUTABLE": Path(sys.executable)}
            )
        )
    with pytest.raises(CVAdapterError, match="CV_PYTHON_EXECUTABLE must be set"):
        CVAdapterConfig.from_settings(
            settings.model_copy(
                update={"CV_REPO_ROOT": repository, "CV_PYTHON_EXECUTABLE": None}
            )
        )

    with pytest.raises(CVAdapterError, match="CV_DEFAULT_MODEL does not exist"):
        CVProcessor(
            database_engine=engine,
            data_dir=data_dir,
            configuration=CVAdapterConfig(
                python_executable=Path(sys.executable),
                repo_root=repository,
                entrypoint=processor.entrypoint,
                default_model=Path("missing.pt"),
                termination_grace_seconds=1,
            ),
        )

    with pytest.raises(
        CVAdapterError,
        match="CV_BALL_GATE and CV_TEAM_SMOOTHING require CV_CANONICAL_IDS=online",
    ):
        CVProcessor(
            database_engine=engine,
            data_dir=data_dir,
            configuration=CVAdapterConfig(
                python_executable=Path(sys.executable),
                repo_root=repository,
                entrypoint=processor.entrypoint,
                default_model=processor.default_model,
                termination_grace_seconds=1,
                canonical_ids="offline",
            ),
        )
    with pytest.raises(CVAdapterError, match="CV_TERMINATION_GRACE_SECONDS"):
        CVProcessor(
            database_engine=engine,
            data_dir=data_dir,
            configuration=CVAdapterConfig(
                python_executable=Path(sys.executable),
                repo_root=repository,
                entrypoint=processor.entrypoint,
                default_model=processor.default_model,
                termination_grace_seconds=-1,
            ),
        )


def test_progress_read_retries_transient_permission_error(
    adapter_environment: tuple[CVProcessor, Engine, Path, uuid.UUID, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, _, _, _, _ = adapter_environment
    progress_path = tmp_path / "progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "completed",
                "frames_processed": 10,
                "total_frames": 10,
            }
        ),
        encoding="utf-8",
    )
    original_open = Path.open
    attempts = 0

    def temporarily_locked_open(
        path: Path, *args: Any, **kwargs: Any
    ) -> Any:
        nonlocal attempts
        if path == progress_path and attempts < 2:
            attempts += 1
            raise PermissionError(13, "File is temporarily locked", str(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", temporarily_locked_open)

    assert processor._consume_progress(progress_path, uuid.uuid4()) is True
    assert attempts == 2


def test_launch_failure_is_reported(
    adapter_environment: tuple[CVProcessor, Engine, Path, uuid.UUID, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, _, _, video_id, _ = adapter_environment

    def fail_to_launch(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("test executable disappeared")

    monkeypatch.setattr("app.worker.cv_adapter.subprocess.Popen", fail_to_launch)
    with pytest.raises(CVAdapterError, match="Could not start the configured CV process"):
        processor.process(_job(video_id), lambda: False, Event())


def test_command_uses_explicit_paths_and_is_shell_free(
    adapter_environment: tuple[CVProcessor, Engine, Path, uuid.UUID, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, _, _, video_id, repository = adapter_environment
    monkeypatch.setenv("FOOTBALL_BACKEND", "cuda")
    monkeypatch.setenv("FOOTBALL_DEVICE", "0")
    actual_popen = subprocess.Popen
    observed: dict[str, object] = {}

    def recording_popen(
        command: list[str], **kwargs: Any
    ) -> PopenType[bytes]:
        observed["command"] = command
        observed.update(kwargs)
        return cast(PopenType[bytes], actual_popen(command, **kwargs))

    monkeypatch.setattr("app.worker.cv_adapter.subprocess.Popen", recording_popen)
    result = processor.process(_job(video_id), lambda: False, Event())

    command = observed["command"]
    assert isinstance(command, list)
    assert command[0] == str(Path(sys.executable).resolve())
    assert command[1] == str((repository / "fake cv entrypoint.py").resolve())
    assert command[2:] == [
        "--backend",
        "auto",
        "--canonical-ids",
        "online",
        "--video",
        str((processor.data_dir / "uploads" / "source video.mp4").resolve()),
        "--model",
        str((repository / "model file.pt").resolve()),
        "--output-dir",
        str(result.output_root),
        "--progress-file",
        str(result.progress_json),
        "--resume-checkpoint",
        str(result.output_root / "pipeline.checkpoint"),
        "--ffmpeg-executable",
        settings.FFMPEG_EXECUTABLE,
        "--ball-gate",
        "--team-smoothing",
    ]
    assert observed["cwd"] == repository.resolve()
    assert observed["shell"] is False
    assert observed["stdin"] == subprocess.DEVNULL
    assert observed["stdout"] == subprocess.PIPE
    assert observed["stderr"] == subprocess.PIPE
    child_environment = observed["env"]
    assert isinstance(child_environment, dict)
    assert "FOOTBALL_BACKEND" not in child_environment
    assert "FOOTBALL_DEVICE" not in child_environment
    assert "--use-stub" not in command
    assert "--cache-stub" not in command


def test_output_and_logs_are_isolated_per_analysis(
    adapter_environment: tuple[CVProcessor, Engine, Path, uuid.UUID, Path],
) -> None:
    processor, _, data_dir, video_id, _ = adapter_environment
    first = processor.process(_job(video_id), lambda: False, Event())
    second = processor.process(_job(video_id), lambda: False, Event())

    assert first.output_root == data_dir / "analyses" / str(first.run_directory.parent.parent.name) / "cv-output"
    assert first.output_root != second.output_root
    assert first.run_directory != second.run_directory
    assert first.output_root.name == second.output_root.name == "cv-output"
    assert first.tracked_video.read_bytes() == b"video"
    assert first.observations_csv.read_bytes() == b"frame_number\n1\n"
    assert first.progress_json.is_file()
    assert json.loads(first.progress_json.read_text(encoding="utf-8")) == {
        "schema_version": "1.0",
        "status": "completed",
        "frames_processed": 10,
        "total_frames": 10,
    }
    assert first.summary_json is not None
    assert first.summary_json.is_file()
    assert first.stdout_log.read_text(encoding="utf-8").find("captured stdout") >= 0
    assert first.stderr_log.read_text(encoding="utf-8").find("captured stderr") >= 0


def test_nonzero_exit_and_missing_outputs_fail(
    adapter_environment: tuple[CVProcessor, Engine, Path, uuid.UUID, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, _, _, video_id, _ = adapter_environment
    monkeypatch.setenv("FAKE_CV_MODE", "failure")
    with pytest.raises(CVAdapterError, match="return code 7"):
        processor.process(_job(video_id), lambda: False, Event())

    monkeypatch.setenv("FAKE_CV_MODE", "missing")
    with pytest.raises(CVAdapterError, match="non-empty tracked.mp4"):
        processor.process(_job(video_id), lambda: False, Event())
    monkeypatch.setenv("FAKE_CV_MODE", "empty")
    with pytest.raises(CVAdapterError, match="non-empty tracked.mp4"):
        processor.process(_job(video_id), lambda: False, Event())


def test_successful_exit_without_generated_summary_fails(
    adapter_environment: tuple[CVProcessor, Engine, Path, uuid.UUID, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, _, _, video_id, _ = adapter_environment
    monkeypatch.setattr(processor, "_ensure_summary_json", lambda *_args: None)

    with pytest.raises(CVAdapterError, match="no summary.json was produced"):
        processor.process(_job(video_id), lambda: False, Event())


def test_logs_are_capped_while_streams_are_drained(
    adapter_environment: tuple[CVProcessor, Engine, Path, uuid.UUID, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, _, _, video_id, _ = adapter_environment
    monkeypatch.setenv("FAKE_CV_MODE", "large_log")
    result = processor.process(_job(video_id), lambda: False, Event())

    assert result.stdout_log.stat().st_size == _MAX_LOG_BYTES
    assert result.stderr_log.stat().st_size < _MAX_LOG_BYTES
    assert result.stdout_log.read_bytes().endswith(_LOG_TRUNCATION_MARKER)


def test_cancellation_terminates_process_group_and_child(
    adapter_environment: tuple[CVProcessor, Engine, Path, uuid.UUID, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    processor, _, _, video_id, _ = adapter_environment
    child_pid_file = tmp_path / "child pid.txt"
    monkeypatch.setenv("FAKE_CV_MODE", "ignore_signal")
    monkeypatch.setenv("FAKE_CV_CHILD_PID_FILE", str(child_pid_file))
    process_started = time.monotonic()

    started = time.monotonic()
    with pytest.raises(ProcessingCancelled):
        processor.process(
            _job(video_id),
            lambda: time.monotonic() - process_started > 0.5,
            Event(),
        )
    assert time.monotonic() - started < 5

    child_pid = int(child_pid_file.read_text(encoding="utf-8"))
    if os.name == "nt":
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        process_handle = kernel32.OpenProcess(0x1000, False, child_pid)
        if process_handle:
            exit_code = wintypes.DWORD()
            try:
                assert kernel32.GetExitCodeProcess(
                    process_handle, ctypes.byref(exit_code)
                )
                assert exit_code.value != 259
            finally:
                kernel32.CloseHandle(process_handle)
        else:
            assert ctypes.get_last_error() == 87
    else:
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
def test_worker_records_cv_success_failure_and_cancellation(
    adapter_environment: tuple[CVProcessor, Engine, Path, uuid.UUID, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, engine, data_dir, video_id, _ = adapter_environment
    analysis_ids = [uuid.uuid4() for _ in range(3)]

    def prepare_browser_video(source: Path) -> Path:
        browser_video = source.with_name("browser.mp4")
        browser_video.write_bytes(source.read_bytes())
        return browser_video

    monkeypatch.setattr(processor, "_prepare_browser_video", prepare_browser_video)
    with Session(engine) as session:
        for analysis_id in analysis_ids:
            session.add(Analysis(id=analysis_id, video_id=video_id))
        session.commit()

    manager = LocalJobManager(
        processor=processor,
        database_engine=engine,
        data_dir=data_dir,
        poll_seconds=0.01,
    )
    assert manager.run_once() == analysis_ids[0]
    with Session(engine) as session:
        analysis = session.get(Analysis, analysis_ids[0])
        assert analysis is not None
        assert analysis.status == AnalysisStatus.SUCCEEDED
        assert analysis.frames_processed == 10
        assert analysis.total_frames == 10
        artifacts = session.exec(
            select(Artifact).where(Artifact.analysis_id == analysis_ids[0])
        ).all()
        assert {artifact.kind for artifact in artifacts} == {
            ArtifactKind.TRACKED_VIDEO,
            ArtifactKind.BROWSER_VIDEO,
            ArtifactKind.OBSERVATIONS_CSV,
            ArtifactKind.SUMMARY,
            ArtifactKind.MATCH_SUMMARY,
            ArtifactKind.PASSES_CSV,
            ArtifactKind.PASSING_NETWORK_CSV,
            ArtifactKind.TEAM_STATISTICS_CSV,
            ArtifactKind.PLAYER_STATISTICS_CSV,
            ArtifactKind.POSSESSION_SEQUENCES_CSV,
        }
        assert all(artifact.sha256 is not None for artifact in artifacts)
        assert all(
            artifact.storage_path.startswith(
                f"analyses/{analysis_ids[0]}/"
            )
            for artifact in artifacts
        )

    monkeypatch.setenv("FAKE_CV_MODE", "failure")
    assert manager.run_once() == analysis_ids[1]
    with Session(engine) as session:
        analysis = session.get(Analysis, analysis_ids[1])
        assert analysis is not None
        assert analysis.status == AnalysisStatus.FAILED

    monkeypatch.setenv("FAKE_CV_MODE", "sleep")
    cancellation_thread = Thread(
        target=_request_cancel,
        args=(engine, analysis_ids[2]),
    )
    cancellation_thread.start()
    assert manager.run_once() == analysis_ids[2]
    cancellation_thread.join(timeout=3)
    assert not cancellation_thread.is_alive()
    with Session(engine) as session:
        analysis = session.get(Analysis, analysis_ids[2])
        assert analysis is not None
        assert analysis.status == AnalysisStatus.CANCELLED
        assert analysis.frames_processed == 1
        assert analysis.total_frames == 10


def test_browser_video_is_validated_as_h264(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "tracked.mp4"
    source.write_bytes(b"cv output")
    calls: list[list[str]] = []

    def fake_run(
        arguments: list[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        if arguments[0].endswith("ffmpeg"):
            Path(arguments[-1]).write_bytes(b"browser video")
            return subprocess.CompletedProcess(arguments, 0, "", "")
        return subprocess.CompletedProcess(
            arguments,
            0,
            json.dumps(
                {
                    "streams": [
                        {
                            "codec_name": "h264",
                            "codec_tag_string": "avc1",
                            "pix_fmt": "yuv420p",
                        }
                    ]
                }
            ),
            "",
        )

    monkeypatch.setattr("app.worker.cv_adapter.shutil.which", lambda name: name)
    monkeypatch.setattr("app.worker.cv_adapter.subprocess.run", fake_run)

    browser_video = CVProcessor._prepare_browser_video(source)

    assert browser_video.name == "browser.mp4"
    assert browser_video.read_bytes() == b"browser video"
    assert calls[0][calls[0].index("-c:v") + 1] == "libx264"
    assert calls[0][-2:] == ["+faststart", str(tmp_path / "browser.tmp.mp4")]


def _request_cancel(engine: Engine, analysis_id: uuid.UUID) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        with Session(engine) as session:
            analysis = session.get(Analysis, analysis_id)
            if (
                analysis is not None
                and analysis.status == AnalysisStatus.RUNNING
                and analysis.frames_processed > 0
            ):
                analysis.status = AnalysisStatus.CANCEL_REQUESTED
                session.add(analysis)
                session.commit()
                return
        time.sleep(0.01)
    raise AssertionError("Analysis did not enter running state before timeout")
