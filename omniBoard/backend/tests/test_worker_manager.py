import uuid
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event

import pytest
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel

from app.core.database import create_database_engine
from app.models import (
    Analysis,
    AnalysisStatus,
    Project,
    User,
    Video,
    get_datetime_utc,
)
from app.worker.manager import (
    FakeProcessor,
    LocalJobManager,
    ProcessingCancelled,
    Processor,
    WorkerAlreadyRunning,
    WorkerLock,
    WorkerShutdownRequested,
)


@pytest.fixture
def worker_engine(tmp_path: Path) -> Generator[Engine]:
    database_path = tmp_path / "worker-test.db"
    test_engine = create_database_engine(f"sqlite:///{database_path.as_posix()}")
    SQLModel.metadata.create_all(test_engine)
    yield test_engine
    test_engine.dispose()


@pytest.fixture
def job_factory(worker_engine: Engine) -> Generator[Callable[..., uuid.UUID]]:
    owner_id = uuid.uuid4()
    with Session(worker_engine) as session:
        session.add(
            User(
                id=owner_id,
                email=f"worker-{owner_id.hex}@example.com",
                hashed_password="not-used",
            )
        )
        session.commit()

    def create_job(
        *,
        created_at: datetime | None = None,
        status: AnalysisStatus = AnalysisStatus.QUEUED,
    ) -> uuid.UUID:
        unique = uuid.uuid4().hex
        project = Project(name=f"Worker test {unique}", owner_id=owner_id)
        video = Video(
            project=project,
            original_filename="test.mp4",
            storage_path=f"uploads/worker-tests/{unique}.mp4",
            file_size_bytes=16,
            sha256="a" * 64,
        )
        analysis = Analysis(
            video=video,
            status=status,
            created_at=created_at or get_datetime_utc(),
        )
        with Session(worker_engine) as session:
            session.add(analysis)
            session.commit()
            session.refresh(analysis)
        assert analysis.id is not None
        return analysis.id

    yield create_job


def _status(database_engine: Engine, analysis_id: uuid.UUID) -> Analysis:
    with Session(database_engine) as session:
        analysis = session.get(Analysis, analysis_id)
        assert analysis is not None
        return analysis


def _manager(
    tmp_path: Path,
    database_engine: Engine,
    processor: Processor | None = None,
) -> LocalJobManager:
    if processor is None:
        processor = FakeProcessor()
    return LocalJobManager(
        processor=processor,
        database_engine=database_engine,
        data_dir=tmp_path,
        poll_seconds=0.02,
    )


def test_fifo_claim_and_running_state_committed_before_processor(
    job_factory: Callable[..., uuid.UUID], tmp_path: Path, worker_engine: Engine
) -> None:
    older = job_factory(created_at=get_datetime_utc() - timedelta(minutes=2))
    newer = job_factory(created_at=get_datetime_utc() - timedelta(minutes=1))
    observed: list[tuple[uuid.UUID, AnalysisStatus]] = []

    class InspectingProcessor:
        def process(self, job, cancellation_requested, stop_event) -> None:
            del cancellation_requested, stop_event
            observed.append((job.id, _status(worker_engine, job.id).status))

    manager = _manager(tmp_path, worker_engine, InspectingProcessor())

    assert manager.run_once() == older
    assert manager.run_once() == newer
    assert observed == [
        (older, AnalysisStatus.RUNNING),
        (newer, AnalysisStatus.RUNNING),
    ]
    assert _status(worker_engine, older).started_at is not None
    assert _status(worker_engine, older).finished_at is not None
    assert _status(worker_engine, older).status == AnalysisStatus.SUCCEEDED


def test_two_claimers_never_claim_the_same_analysis(
    job_factory: Callable[..., uuid.UUID], tmp_path: Path, worker_engine: Engine
) -> None:
    first = job_factory()
    second = job_factory()
    managers = (
        _manager(tmp_path, worker_engine),
        _manager(tmp_path, worker_engine),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(executor.map(lambda manager: manager.claim_next(), managers))

    claimed_ids = [job.id for job in claims if job is not None]
    assert set(claimed_ids) == {first, second}
    assert len(claimed_ids) == 2
    assert all(
        _status(worker_engine, analysis_id).status == AnalysisStatus.RUNNING
        for analysis_id in claimed_ids
    )


def test_worker_writes_ready_file_after_startup(
    tmp_path: Path, worker_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready_file = tmp_path / "worker-ready.txt"
    monkeypatch.setenv("OMNIBOARD_WORKER_READY_FILE", str(ready_file))
    stop_event = Event()
    stop_event.set()

    _manager(tmp_path, worker_engine).run(stop_event)

    assert ready_file.read_text(encoding="ascii") == "ready"


def test_jobs_are_processed_serially_and_fake_processor_creates_no_artifacts(
    job_factory: Callable[..., uuid.UUID], tmp_path: Path, worker_engine: Engine
) -> None:
    first = job_factory()
    second = job_factory()
    manager = _manager(tmp_path, worker_engine, FakeProcessor(duration_seconds=0.01))

    assert manager.run_once() == first
    assert _status(worker_engine, first).status == AnalysisStatus.SUCCEEDED
    assert _status(worker_engine, second).status == AnalysisStatus.QUEUED
    assert manager.run_once() == second
    assert _status(worker_engine, second).status == AnalysisStatus.SUCCEEDED


def test_failed_job_does_not_stop_next_job(
    job_factory: Callable[..., uuid.UUID], tmp_path: Path, worker_engine: Engine
) -> None:
    failed_id = job_factory()
    succeeded_id = job_factory()

    class FailFirstProcessor:
        def process(self, job, cancellation_requested, stop_event) -> None:
            del cancellation_requested, stop_event
            if job.id == failed_id:
                raise RuntimeError("test processor failure")

    manager = _manager(tmp_path, worker_engine, FailFirstProcessor())

    assert manager.run_once() == failed_id
    failed = _status(worker_engine, failed_id)
    assert failed.status == AnalysisStatus.FAILED
    assert failed.error_code == "processing_failed"
    assert failed.error_message == "Processing failed. See worker logs for details."
    assert "test processor failure" not in failed.error_message

    assert manager.run_once() == succeeded_id
    assert _status(worker_engine, succeeded_id).status == AnalysisStatus.SUCCEEDED


def test_cancel_requested_before_claim_never_reaches_processor(
    job_factory: Callable[..., uuid.UUID], tmp_path: Path, worker_engine: Engine
) -> None:
    cancelled_id = job_factory(status=AnalysisStatus.CANCEL_REQUESTED)
    processed: list[uuid.UUID] = []

    class RecordingProcessor:
        def process(self, job, cancellation_requested, stop_event) -> None:
            del cancellation_requested, stop_event
            processed.append(job.id)

    manager = _manager(tmp_path, worker_engine, RecordingProcessor())

    assert manager.run_once() is None
    assert _status(worker_engine, cancelled_id).status == AnalysisStatus.CANCELLED
    assert _status(worker_engine, cancelled_id).finished_at is not None
    assert processed == []


def test_running_job_supports_cooperative_cancellation(
    job_factory: Callable[..., uuid.UUID], tmp_path: Path, worker_engine: Engine
) -> None:
    analysis_id = job_factory()

    class CancellingProcessor:
        def process(self, job, cancellation_requested, stop_event) -> None:
            del stop_event
            with Session(worker_engine) as session:
                analysis = session.get(Analysis, job.id)
                assert analysis is not None
                analysis.status = AnalysisStatus.CANCEL_REQUESTED
                session.add(analysis)
                session.commit()
            if cancellation_requested():
                raise ProcessingCancelled

    manager = _manager(tmp_path, worker_engine, CancellingProcessor())

    assert manager.run_once() == analysis_id
    assert _status(worker_engine, analysis_id).status == AnalysisStatus.CANCELLED


def test_worker_restart_requeues_running_jobs_with_or_without_checkpoint(
    job_factory: Callable[..., uuid.UUID], tmp_path: Path, worker_engine: Engine
) -> None:
    stale_running = job_factory(status=AnalysisStatus.RUNNING)
    queued = job_factory()
    with Session(worker_engine) as session:
        analysis = session.get(Analysis, stale_running)
        assert analysis is not None
        analysis.started_at = get_datetime_utc() - timedelta(minutes=10)
        session.add(analysis)
        session.commit()
    manager = _manager(tmp_path, worker_engine)

    assert manager.recover_interrupted() == 1
    interrupted = _status(worker_engine, stale_running)
    assert interrupted.status == AnalysisStatus.QUEUED
    assert interrupted.started_at is None
    assert interrupted.finished_at is None
    assert interrupted.error_code is None
    assert _status(worker_engine, queued).status == AnalysisStatus.QUEUED
    assert manager.recover_interrupted() == 0


def test_worker_restart_requeues_only_jobs_with_resume_checkpoints(
    job_factory: Callable[..., uuid.UUID],
    tmp_path: Path,
    worker_engine: Engine,
) -> None:
    resumable_id = job_factory(status=AnalysisStatus.RUNNING)
    interrupted_id = job_factory(status=AnalysisStatus.INTERRUPTED)
    checkpoint = (
        tmp_path
        / "analyses"
        / str(resumable_id)
        / "cv-output"
        / "pipeline.checkpoint"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    with Session(worker_engine) as session:
        analysis = session.get(Analysis, interrupted_id)
        assert analysis is not None
        analysis.error_code = "worker_interrupted"
        session.add(analysis)
        session.commit()

    manager = _manager(tmp_path, worker_engine)

    assert manager.recover_interrupted() == 2
    assert _status(worker_engine, resumable_id).status == AnalysisStatus.QUEUED
    assert _status(worker_engine, interrupted_id).status == AnalysisStatus.QUEUED


def test_second_worker_lock_is_rejected_and_lock_is_released(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / ".analysis-worker.lock"
    with WorkerLock(lock_path):
        with pytest.raises(WorkerAlreadyRunning):
            WorkerLock(lock_path).acquire()

    with WorkerLock(lock_path):
        assert lock_path.exists()


def test_worker_shutdown_marks_active_analysis_interrupted(
    job_factory: Callable[..., uuid.UUID], tmp_path: Path, worker_engine: Engine
) -> None:
    analysis_id = job_factory()
    started = Event()
    stop_event = Event()

    class WaitingProcessor:
        def process(self, job, cancellation_requested, stop_event) -> None:
            del job, cancellation_requested
            started.set()
            stop_event.set()
            raise WorkerShutdownRequested

    manager = _manager(tmp_path, worker_engine, WaitingProcessor())
    assert _status(worker_engine, analysis_id).status == AnalysisStatus.QUEUED
    assert manager.run_once(stop_event) == analysis_id

    assert started.is_set()
    assert _status(worker_engine, analysis_id).status == AnalysisStatus.INTERRUPTED
    with WorkerLock(tmp_path / ".analysis-worker.lock"):
        pass


def test_empty_queue_polls_and_waits(
    tmp_path: Path, worker_engine: Engine
) -> None:
    class StopAfterPolls(Event):
        def __init__(self) -> None:
            super().__init__()
            self.polls = 0

        def wait(self, timeout: float | None = None) -> bool:
            assert timeout == pytest.approx(0.02)
            self.polls += 1
            if self.polls >= 2:
                self.set()
            return self.is_set()

    stop_event = StopAfterPolls()
    manager = _manager(tmp_path, worker_engine)

    manager.run(stop_event)

    assert stop_event.polls == 2
