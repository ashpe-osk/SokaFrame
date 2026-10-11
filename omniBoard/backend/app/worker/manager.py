import errno
import logging
import os
import signal
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Protocol

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from sqlalchemy import or_, update
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from app.core.config import settings
from app.core.db import engine as application_engine
from app.models import Analysis, AnalysisStatus, get_datetime_utc

logger = logging.getLogger(__name__)


class WorkerAlreadyRunning(RuntimeError):
    pass


class ProcessingCancelled(Exception):
    pass


class WorkerShutdownRequested(Exception):
    pass


@dataclass(frozen=True)
class AnalysisJob:
    id: uuid.UUID
    video_id: uuid.UUID
    options: dict[str, object]
    worker_id: uuid.UUID


class Processor(Protocol):
    def process(
        self,
        job: AnalysisJob,
        cancellation_requested: Callable[[], bool],
        stop_event: Event,
    ) -> object | None: ...


class FakeProcessor:
    """Small lifecycle-only processor; it produces no CV outputs."""

    def __init__(self, duration_seconds: float = 0.1) -> None:
        self.duration_seconds = duration_seconds

    def process(
        self,
        job: AnalysisJob,
        cancellation_requested: Callable[[], bool],
        stop_event: Event,
    ) -> None:
        del job
        deadline = time.monotonic() + self.duration_seconds
        while True:
            if stop_event.is_set():
                raise WorkerShutdownRequested
            if cancellation_requested():
                raise ProcessingCancelled
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            stop_event.wait(min(remaining, 0.05))


class WorkerLock:
    """Process-lifetime lock; the OS releases it when a worker exits or crashes."""

    def __init__(self, lock_path: Path) -> None:
        self.lock_path = lock_path
        self._file = None

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.lock_path.open("a+b")
        self._file.seek(0, os.SEEK_END)
        if self._file.tell() == 0:
            self._file.write(b"\0")
            self._file.flush()
        self._file.seek(0)

        try:
            if os.name == "nt":
                msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._file.close()
            self._file = None
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK, None):
                raise WorkerAlreadyRunning("Another local analysis worker is active") from None
            raise

    def release(self) -> None:
        if self._file is None:
            return
        try:
            self._file.seek(0)
            if os.name == "nt":
                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None

    def __enter__(self) -> WorkerLock:
        self.acquire()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.release()


class LocalJobManager:
    def __init__(
        self,
        *,
        processor: Processor,
        database_engine: Engine = application_engine,
        data_dir: Path | None = None,
        poll_seconds: float | None = None,
        worker_id: uuid.UUID | None = None,
    ) -> None:
        self.engine = database_engine
        self.data_dir = data_dir or settings.LOCAL_DATA_DIR
        self.processor = processor
        self.poll_seconds = (
            poll_seconds if poll_seconds is not None else settings.WORKER_POLL_SECONDS
        )
        if self.poll_seconds <= 0:
            raise ValueError("Worker polling interval must be positive")
        self.worker_id = worker_id or uuid.uuid4()

    def recover_interrupted(self) -> int:
        with Session(self.engine) as session:
            interrupted = session.exec(
                select(Analysis.id).where(
                    or_(
                        Analysis.status == AnalysisStatus.RUNNING,
                        (
                            (Analysis.status == AnalysisStatus.INTERRUPTED)
                            & (Analysis.error_code == "worker_interrupted")
                        ),
                    )
                )
            ).all()
        recovered_count = 0
        checkpointed_count = 0
        with self.engine.begin() as connection:
            for analysis_id in interrupted:
                checkpoint = (
                    self.data_dir
                    / "analyses"
                    / str(analysis_id)
                    / "cv-output"
                    / "pipeline.checkpoint"
                )
                resumable = checkpoint.is_file()
                checkpointed_count += int(resumable)
                result = connection.execute(
                    update(Analysis)
                    .where(
                        Analysis.id == analysis_id,
                        Analysis.status.in_(
                            (
                                AnalysisStatus.RUNNING,
                                AnalysisStatus.INTERRUPTED,
                            )
                        ),
                    )
                    .values(
                        status=AnalysisStatus.QUEUED,
                        started_at=None,
                        finished_at=None,
                        error_code=None,
                        error_message=None,
                    )
                )
                recovered_count += result.rowcount or 0
        if recovered_count:
            logger.warning(
                "Recovered %d interrupted analyses (%d resumable)",
                recovered_count,
                checkpointed_count,
            )
        return recovered_count

    def cancel_unclaimed(self) -> int:
        statement = (
            update(Analysis)
            .where(
                Analysis.status == AnalysisStatus.CANCEL_REQUESTED,
                Analysis.started_at.is_(None),
            )
            .values(
                status=AnalysisStatus.CANCELLED,
                finished_at=get_datetime_utc(),
                error_code=None,
                error_message=None,
            )
        )
        with self.engine.begin() as connection:
            result = connection.execute(statement)
            return result.rowcount or 0

    def claim_next(self) -> AnalysisJob | None:
        with self.engine.connect() as connection:
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            try:
                analysis_id = connection.execute(
                    select(Analysis.id)
                    .where(Analysis.status == AnalysisStatus.QUEUED)
                    .order_by(Analysis.created_at.asc(), Analysis.id.asc())
                    .limit(1)
                ).scalar_one_or_none()
                if analysis_id is None:
                    connection.commit()
                    return None

                now = get_datetime_utc()
                claim_result = connection.execute(
                    update(Analysis)
                    .where(
                        Analysis.id == analysis_id,
                        Analysis.status == AnalysisStatus.QUEUED,
                    )
                    .values(
                        status=AnalysisStatus.RUNNING,
                        started_at=now,
                        finished_at=None,
                        error_code=None,
                        error_message=None,
                    )
                )
                if claim_result.rowcount != 1:
                    connection.commit()
                    return None

                row = connection.execute(
                    select(Analysis.id, Analysis.video_id, Analysis.options).where(
                        Analysis.id == analysis_id
                    )
                ).one()
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

        return AnalysisJob(
            id=row.id,
            video_id=row.video_id,
            options=row.options,
            worker_id=self.worker_id,
        )

    def run_once(self, stop_event: Event | None = None) -> uuid.UUID | None:
        stop_event = stop_event or Event()
        self.cancel_unclaimed()
        job = self.claim_next()
        if job is None:
            return None

        logger.info("Claimed analysis %s", job.id)
        try:
            self.processor.process(
                job,
                cancellation_requested=lambda: self._cancellation_requested(job.id),
                stop_event=stop_event,
            )
        except ProcessingCancelled:
            self._mark_cancelled(job.id)
            logger.info("Analysis %s cancelled", job.id)
        except WorkerShutdownRequested:
            self._mark_interrupted(job.id)
            logger.info("Analysis %s interrupted during worker shutdown", job.id)
        except Exception:
            logger.exception("Analysis %s failed during processing", job.id)
            self._mark_failed(job.id)
        else:
            if stop_event.is_set():
                self._mark_interrupted(job.id)
                logger.info("Analysis %s interrupted during worker shutdown", job.id)
            else:
                self._mark_succeeded(job.id)
                logger.info("Analysis %s succeeded", job.id)
        return job.id

    def run(self, stop_event: Event | None = None) -> None:
        stop_event = stop_event or Event()
        lock = WorkerLock(self.data_dir / ".analysis-worker.lock")
        with lock:
            logger.info("Worker started: %s", self.worker_id)
            self.recover_interrupted()
            ready_file = os.environ.get("OMNIBOARD_WORKER_READY_FILE")
            if ready_file:
                ready_path = Path(ready_file)
                ready_path.parent.mkdir(parents=True, exist_ok=True)
                ready_path.write_text("ready", encoding="ascii")
            try:
                while not stop_event.is_set():
                    if self.run_once(stop_event) is None:
                        logger.debug("Waiting for queued analyses")
                        stop_event.wait(self.poll_seconds)
            finally:
                logger.info("Worker shutting down: %s", self.worker_id)

    def _cancellation_requested(self, analysis_id: uuid.UUID) -> bool:
        with Session(self.engine) as session:
            analysis = session.get(Analysis, analysis_id)
            return (
                analysis is None
                or analysis.status == AnalysisStatus.CANCEL_REQUESTED
            )

    def _set_terminal_state(
        self,
        analysis_id: uuid.UUID,
        status: AnalysisStatus,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self.engine.begin() as connection:
            current_status = connection.execute(
                select(Analysis.status).where(Analysis.id == analysis_id)
            ).scalar_one_or_none()
            if current_status is None:
                return
            terminal_status = (
                AnalysisStatus.CANCELLED
                if current_status == AnalysisStatus.CANCEL_REQUESTED
                else status
            )
            connection.execute(
                update(Analysis)
                .where(Analysis.id == analysis_id)
                .values(
                    status=terminal_status,
                    finished_at=get_datetime_utc(),
                    error_code=error_code if terminal_status == status else None,
                    error_message=error_message if terminal_status == status else None,
                )
            )

    def _mark_succeeded(self, analysis_id: uuid.UUID) -> None:
        self._set_terminal_state(analysis_id, AnalysisStatus.SUCCEEDED)

    def _mark_cancelled(self, analysis_id: uuid.UUID) -> None:
        self._set_terminal_state(analysis_id, AnalysisStatus.CANCELLED)

    def _mark_interrupted(self, analysis_id: uuid.UUID) -> None:
        self._set_terminal_state(
            analysis_id,
            AnalysisStatus.INTERRUPTED,
            error_code="worker_interrupted",
            error_message="Worker stopped while this analysis was running.",
        )

    def _mark_failed(self, analysis_id: uuid.UUID) -> None:
        self._set_terminal_state(
            analysis_id,
            AnalysisStatus.FAILED,
            error_code="processing_failed",
            error_message="Processing failed. See worker logs for details.",
        )


def install_shutdown_handlers(stop_event: Event) -> None:
    def request_shutdown(signum: int, _frame: object) -> None:
        logger.info("Received shutdown signal %s", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, request_shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_shutdown)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from app.worker.cv_adapter import CVProcessor

    processor = CVProcessor(
        database_engine=application_engine,
        data_dir=settings.LOCAL_DATA_DIR,
    )
    stop_event = Event()
    install_shutdown_handlers(stop_event)
    manager = LocalJobManager(processor=processor)
    try:
        manager.run(stop_event)
    except WorkerAlreadyRunning:
        logger.error("Another local analysis worker is already running")
        raise SystemExit(1) from None
