import logging
import shutil
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi import (
    Path as PathParam,
)
from fastapi.responses import FileResponse
from pydantic import JsonValue
from sqlalchemy import delete, exists, update
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Field, Session, SQLModel, func, select

from app.api.deps import CurrentUser, SessionDep
from app.core.config import settings
from app.models import (
    AnalysesPublic,
    Analysis,
    AnalysisArtifactPublic,
    AnalysisPublic,
    AnalysisStatus,
    APIError,
    Artifact,
    ArtifactKind,
    Project,
    Video,
    get_datetime_utc,
)
from app.services.video_storage import (
    InvalidVideoFile,
    StoredVideo,
    UnsupportedVideoFormat,
    UploadTooLarge,
    store_video_upload,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyses", tags=["analyses"])
_TERMINAL_ANALYSIS_STATUSES = (
    AnalysisStatus.SUCCEEDED,
    AnalysisStatus.FAILED,
    AnalysisStatus.CANCELLED,
    AnalysisStatus.INTERRUPTED,
)


def _remove_analysis_files(
    analysis_ids: list[uuid.UUID], artifact_paths: list[str]
) -> None:
    data_root = settings.LOCAL_DATA_DIR.expanduser().resolve()
    errors: list[OSError] = []

    def ensure_contained(path: Path, label: str) -> Path:
        resolved = path.resolve()
        try:
            resolved.relative_to(data_root)
        except ValueError as exc:
            raise OSError(f"{label} is outside the configured data directory.") from exc
        return path

    for storage_path in artifact_paths:
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
            errors.append(OSError("An artifact has an unsafe stored path."))
            continue
        artifact_path = data_root.joinpath(*posix_path.parts)
        try:
            ensure_contained(artifact_path, "Artifact path")
            if artifact_path.is_symlink():
                artifact_path.unlink()
            elif artifact_path.is_file():
                artifact_path.unlink()
            elif artifact_path.exists():
                raise OSError("An artifact path is not a regular file.")
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.exception("Failed to remove analysis artifact %s", artifact_path)
            errors.append(exc)

    for analysis_id in analysis_ids:
        for parent in ("analyses", "outputs"):
            analysis_directory = data_root / parent / str(analysis_id)
            try:
                ensure_contained(analysis_directory, "Analysis output directory")
                if analysis_directory.is_symlink():
                    analysis_directory.unlink()
                elif analysis_directory.is_dir():
                    shutil.rmtree(analysis_directory)
            except FileNotFoundError:
                continue
            except OSError as exc:
                logger.exception(
                    "Failed to remove analysis output directory %s",
                    analysis_directory,
                )
                errors.append(exc)

    if errors:
        raise OSError(
            f"Could not remove {len(errors)} file(s) for deleted analysis history."
        ) from errors[0]


def _remove_uploaded_video_files(storage_paths: list[str]) -> None:
    data_root = settings.LOCAL_DATA_DIR.expanduser().resolve()
    uploads_root = data_root / "uploads"
    errors: list[OSError] = []

    for storage_path in storage_paths:
        posix_path = PurePosixPath(storage_path)
        windows_path = PureWindowsPath(storage_path)
        if (
            not storage_path
            or posix_path.is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or ".." in posix_path.parts
            or ".." in windows_path.parts
            or len(posix_path.parts) < 2
            or posix_path.parts[0] != "uploads"
        ):
            errors.append(OSError("An uploaded video has an unsafe stored path."))
            continue

        video_path = data_root.joinpath(*posix_path.parts)
        try:
            try:
                video_path.parent.resolve().relative_to(uploads_root.resolve())
            except ValueError as exc:
                raise OSError(
                    "Uploaded video path is outside the uploads directory."
                ) from exc
            if video_path.is_symlink():
                video_path.unlink()
            elif video_path.is_file():
                video_path.unlink()
            elif video_path.exists():
                raise OSError("An uploaded video path is not a regular file.")
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.exception("Failed to remove uploaded video %s", video_path)
            errors.append(exc)

    if errors:
        raise OSError(
            f"Could not remove {len(errors)} uploaded video file(s)."
        ) from errors[0]


class AnalysisFromVideoCreate(SQLModel):
    project_id: uuid.UUID
    video_id: uuid.UUID
    options: dict[str, JsonValue] = Field(default_factory=dict)


def _analysis_public(
    analysis: Analysis,
    video: Video,
    project_id: uuid.UUID,
    result_available: bool,
    artifacts: list[Artifact] | None = None,
) -> AnalysisPublic:
    return AnalysisPublic(
        id=analysis.id,
        project_id=project_id,
        video_id=video.id,
        options=analysis.options,
        status=analysis.status,
        frames_processed=analysis.frames_processed,
        total_frames=analysis.total_frames,
        created_at=analysis.created_at,
        started_at=analysis.started_at,
        finished_at=analysis.finished_at,
        error_code=analysis.error_code,
        result_available=result_available,
        video_original_filename=video.original_filename,
        video_media_type=video.media_type,
        video_size_bytes=video.file_size_bytes,
        video_sha256=video.sha256,
        summary=analysis.summary,
        artifacts=[
            AnalysisArtifactPublic(
                id=artifact.id,
                kind=artifact.kind,
                media_type=artifact.media_type,
                file_size_bytes=artifact.file_size_bytes,
                sha256=artifact.sha256,
            )
            for artifact in (artifacts or [])
        ],
    )


@router.post(
    "/from-video",
    response_model=AnalysisPublic,
    status_code=201,
    responses={
        401: {"model": APIError, "description": "Authentication required"},
        403: {"model": APIError, "description": "Project belongs to another user"},
        404: {"model": APIError, "description": "Project or video not found"},
        500: {"model": APIError, "description": "Analysis could not be queued"},
    },
)
def create_analysis_from_video(
    request: AnalysisFromVideoCreate,
    session: SessionDep,
    current_user: CurrentUser,
) -> AnalysisPublic:
    project = session.get(Project, request.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Project belongs to another user")

    video = session.get(Video, request.video_id)
    if video is None or video.project_id != project.id:
        raise HTTPException(
            status_code=404, detail="Video does not belong to this project"
        )

    analysis = Analysis(video_id=video.id, options=request.options)
    session.add(analysis)
    try:
        session.commit()
        session.refresh(analysis)
    except SQLAlchemyError:
        session.rollback()
        logger.exception("Failed to queue analysis for stored video")
        raise HTTPException(
            status_code=500, detail="The analysis could not be queued."
        ) from None
    return _analysis_public(
        analysis, video, project.id, result_available=False, artifacts=[]
    )


def _analysis_artifacts(session: Session, analysis_id: uuid.UUID) -> list[Artifact]:
    return list(
        session.exec(
            select(Artifact)
            .where(Artifact.analysis_id == analysis_id)
            .order_by(Artifact.kind)
        ).all()
    )


@router.post(
    "/",
    response_model=AnalysisPublic,
    status_code=201,
    responses={
        400: {"model": APIError, "description": "Invalid video file"},
        401: {"model": APIError, "description": "Authentication required"},
        403: {"model": APIError, "description": "Project belongs to another user"},
        404: {"model": APIError, "description": "Project not found"},
        413: {"model": APIError, "description": "Video upload exceeds the size limit"},
        415: {"model": APIError, "description": "Unsupported video format"},
        500: {"model": APIError, "description": "Upload or metadata storage failed"},
    },
)
async def create_analysis(
    session: SessionDep,
    current_user: CurrentUser,
    project_id: Annotated[uuid.UUID, Form()],
    video_file: Annotated[UploadFile, File()],
) -> AnalysisPublic:
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Project belongs to another user")

    try:
        stored_video = await store_video_upload(
            video_file,
            settings.LOCAL_DATA_DIR,
            settings.MAX_VIDEO_UPLOAD_BYTES,
        )
    except UnsupportedVideoFormat:
        raise HTTPException(
            status_code=415,
            detail="Unsupported video format. Upload an MP4, MOV, or AVI video.",
        ) from None
    except UploadTooLarge:
        raise HTTPException(
            status_code=413,
            detail=f"Video exceeds the {settings.MAX_VIDEO_UPLOAD_BYTES}-byte upload limit.",
        ) from None
    except InvalidVideoFile:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file does not contain a supported video header.",
        ) from None
    except OSError:
        logger.exception("Failed to store uploaded video")
        raise HTTPException(
            status_code=500, detail="The uploaded video could not be stored."
        ) from None

    return _persist_analysis(session, project, stored_video)


def _persist_analysis(
    session: Session, project: Project, stored_video: StoredVideo
) -> AnalysisPublic:
    video = Video(
        project_id=project.id,
        original_filename=stored_video.original_filename,
        media_type=stored_video.media_type,
        storage_path=stored_video.storage_path,
        file_size_bytes=stored_video.file_size_bytes,
        sha256=stored_video.sha256,
    )
    analysis = Analysis(video_id=video.id, status=AnalysisStatus.QUEUED)
    session.add(video)
    session.add(analysis)
    public_analysis = _analysis_public(
        analysis, video, project.id, result_available=False, artifacts=[]
    )
    try:
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        try:
            stored_video.remove()
        except OSError:
            logger.exception("Failed to clean up video after database commit failure")
        logger.exception("Failed to persist uploaded video and analysis metadata")
        raise HTTPException(
            status_code=500, detail="The analysis could not be created."
        ) from None

    return public_analysis


def _list_analyses(
    session: Session,
    user_id: uuid.UUID,
    skip: int,
    limit: int,
    project_id: uuid.UUID | None = None,
) -> AnalysesPublic:
    filters = [Project.owner_id == user_id]
    if project_id is not None:
        filters.append(Project.id == project_id)
    count_statement = (
        select(func.count())
        .select_from(Analysis)
        .join(Video, Analysis.video_id == Video.id)
        .join(Project, Video.project_id == Project.id)
        .where(*filters)
    )
    count = session.exec(count_statement).one()
    result_available = exists(
        select(Artifact.id).where(Artifact.analysis_id == Analysis.id)
    )
    statement = (
        select(Analysis, Video, Project, result_available)
        .join(Video, Analysis.video_id == Video.id)
        .join(Project, Video.project_id == Project.id)
        .where(*filters)
        .order_by(Analysis.created_at.desc(), Analysis.id.desc())
        .offset(skip)
        .limit(limit)
    )
    rows = session.exec(statement).all()
    artifacts_by_analysis: dict[uuid.UUID, list[Artifact]] = {}
    analysis_ids = [analysis.id for analysis, _, _, _ in rows]
    if analysis_ids:
        artifacts = session.exec(
            select(Artifact)
            .where(Artifact.analysis_id.in_(analysis_ids))
            .order_by(Artifact.kind)
        ).all()
        for artifact in artifacts:
            artifacts_by_analysis.setdefault(artifact.analysis_id, []).append(artifact)
    return AnalysesPublic(
        data=[
            _analysis_public(
                analysis,
                video,
                project.id,
                has_artifacts,
                artifacts_by_analysis.get(analysis.id, []),
            )
            for analysis, video, project, has_artifacts in rows
        ],
        count=count,
    )


@router.get(
    "/",
    response_model=AnalysesPublic,
    responses={401: {"model": APIError, "description": "Authentication required"}},
)
def read_analyses(
    session: SessionDep,
    current_user: CurrentUser,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
) -> AnalysesPublic:
    return _list_analyses(session, current_user.id, skip, limit)


@router.get(
    "/by-project/{project_type}/{client_id}",
    response_model=AnalysesPublic,
    responses={
        401: {"model": APIError, "description": "Authentication required"},
        404: {"model": APIError, "description": "Project not found"},
    },
)
def read_project_analyses(
    project_type: Literal["board", "media"],
    client_id: Annotated[str, PathParam(min_length=1, max_length=255)],
    session: SessionDep,
    current_user: CurrentUser,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
) -> AnalysesPublic:
    project_id = session.exec(
        select(Project.id).where(
            Project.owner_id == current_user.id,
            Project.type == project_type,
            Project.client_id == client_id,
        )
    ).first()
    if project_id is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return _list_analyses(session, current_user.id, skip, limit, project_id)


@router.delete(
    "/by-project/{project_type}/{client_id}/completed",
    responses={
        401: {"model": APIError, "description": "Authentication required"},
        404: {"model": APIError, "description": "Project not found"},
        500: {"model": APIError, "description": "Analysis files could not be removed"},
    },
)
def delete_completed_project_analyses(
    project_type: Literal["board", "media"],
    client_id: Annotated[str, PathParam(min_length=1, max_length=255)],
    session: SessionDep,
    current_user: CurrentUser,
) -> dict[str, int]:
    project_id = session.exec(
        select(Project.id).where(
            Project.owner_id == current_user.id,
            Project.type == project_type,
            Project.client_id == client_id,
        )
    ).first()
    if project_id is None:
        raise HTTPException(status_code=404, detail="Project not found")

    analyses = session.exec(
        select(Analysis)
        .join(Video, Analysis.video_id == Video.id)
        .where(
            Video.project_id == project_id,
            Analysis.status.in_(_TERMINAL_ANALYSIS_STATUSES),
        )
    ).all()
    analysis_ids = [analysis.id for analysis in analyses]
    if not analysis_ids:
        return {"deleted_count": 0}

    artifact_paths = session.exec(
        select(Artifact.storage_path).where(Artifact.analysis_id.in_(analysis_ids))
    ).all()
    try:
        for analysis in analyses:
            session.delete(analysis)
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        logger.exception(
            "Failed to delete completed analyses for project %s", project_id
        )
        raise HTTPException(
            status_code=500, detail="Completed analysis history could not be deleted."
        ) from None

    try:
        _remove_analysis_files(analysis_ids, list(artifact_paths))
    except OSError:
        logger.exception(
            "Completed analysis records were deleted, but some files remain "
            "for project %s",
            project_id,
        )
        raise HTTPException(
            status_code=500,
            detail="History was deleted, but some analysis files could not be removed.",
        ) from None
    return {"deleted_count": len(analysis_ids)}


@router.delete(
    "/by-project/{project_type}/{client_id}",
    responses={
        401: {"model": APIError, "description": "Authentication required"},
        404: {"model": APIError, "description": "Project not found"},
        409: {"model": APIError, "description": "Project has active analyses"},
        500: {"model": APIError, "description": "Analysis files could not be removed"},
    },
)
def delete_all_project_analyses(
    project_type: Literal["board", "media"],
    client_id: Annotated[str, PathParam(min_length=1, max_length=255)],
    session: SessionDep,
    current_user: CurrentUser,
) -> dict[str, object]:
    project_id = session.exec(
        select(Project.id).where(
            Project.owner_id == current_user.id,
            Project.type == project_type,
            Project.client_id == client_id,
        )
    ).first()
    if project_id is None:
        raise HTTPException(status_code=404, detail="Project not found")

    analyses = session.exec(
        select(Analysis)
        .join(Video, Analysis.video_id == Video.id)
        .where(Video.project_id == project_id)
    ).all()
    if any(analysis.status not in _TERMINAL_ANALYSIS_STATUSES for analysis in analyses):
        raise HTTPException(
            status_code=409,
            detail="Active analyses must finish or be cancelled before clearing history.",
        )

    analysis_ids = [analysis.id for analysis in analyses]
    video_ids = list({analysis.video_id for analysis in analyses})
    if not analysis_ids:
        return {"deleted_count": 0, "deleted_video_ids": []}

    artifact_paths = session.exec(
        select(Artifact.storage_path).where(Artifact.analysis_id.in_(analysis_ids))
    ).all()
    try:
        for analysis in analyses:
            session.delete(analysis)
        session.flush()
        orphaned_videos = session.exec(
            select(Video).where(
                Video.id.in_(video_ids),
                ~exists(select(Analysis.id).where(Analysis.video_id == Video.id)),
            )
        ).all()
        deleted_video_ids = [video.id for video in orphaned_videos]
        video_paths = [video.storage_path for video in orphaned_videos]
        for video in orphaned_videos:
            session.delete(video)
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        logger.exception("Failed to clear analyses for project %s", project_id)
        raise HTTPException(
            status_code=500, detail="Analysis history could not be deleted."
        ) from None

    try:
        _remove_analysis_files(analysis_ids, list(artifact_paths))
        _remove_uploaded_video_files(video_paths)
    except OSError:
        logger.exception(
            "Analysis records were deleted, but some files remain for project %s",
            project_id,
        )
        raise HTTPException(
            status_code=500,
            detail="History was deleted, but some analysis files could not be removed.",
        ) from None
    return {
        "deleted_count": len(analysis_ids),
        "deleted_video_ids": [str(video_id) for video_id in deleted_video_ids],
    }


@router.get(
    "/{analysis_id}",
    response_model=AnalysisPublic,
    responses={
        401: {"model": APIError, "description": "Authentication required"},
        403: {"model": APIError, "description": "Analysis belongs to another user"},
        404: {"model": APIError, "description": "Analysis not found"},
    },
)
def read_analysis(
    analysis_id: uuid.UUID, session: SessionDep, current_user: CurrentUser
) -> AnalysisPublic:
    statement = (
        select(Analysis, Video, Project)
        .join(Video, Analysis.video_id == Video.id)
        .join(Project, Video.project_id == Project.id)
        .where(Analysis.id == analysis_id)
    )
    row = session.exec(statement).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    analysis, video, project = row
    if project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Analysis belongs to another user")

    has_artifacts = (
        session.exec(
            select(Artifact.id)
            .where(Artifact.analysis_id == analysis.id)
            .limit(1)
        ).first()
        is not None
    )
    return _analysis_public(
        analysis,
        video,
        project.id,
        has_artifacts,
        _analysis_artifacts(session, analysis.id),
    )


@router.delete(
    "/{analysis_id}",
    responses={
        401: {"model": APIError, "description": "Authentication required"},
        403: {"model": APIError, "description": "Analysis belongs to another user"},
        404: {"model": APIError, "description": "Analysis not found"},
        409: {"model": APIError, "description": "Analysis is still active"},
        500: {"model": APIError, "description": "Analysis files could not be removed"},
    },
)
def delete_analysis(
    analysis_id: uuid.UUID, session: SessionDep, current_user: CurrentUser
) -> dict[str, bool]:
    statement = (
        select(Analysis, Project, Video)
        .join(Video, Analysis.video_id == Video.id)
        .join(Project, Video.project_id == Project.id)
        .where(Analysis.id == analysis_id)
    )
    row = session.exec(statement).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    analysis, project, video = row
    if project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Analysis belongs to another user")
    if analysis.status not in _TERMINAL_ANALYSIS_STATUSES:
        raise HTTPException(
            status_code=409, detail="Active analyses cannot be deleted."
        )

    artifact_paths = session.exec(
        select(Artifact.storage_path).where(Artifact.analysis_id == analysis_id)
    ).all()
    video_deleted = False
    try:
        session.delete(analysis)
        session.flush()
        has_remaining_analyses = session.exec(
            select(Analysis.id).where(Analysis.video_id == video.id).limit(1)
        ).first()
        if has_remaining_analyses is None:
            session.delete(video)
            video_deleted = True
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        logger.exception("Failed to delete analysis %s", analysis_id)
        raise HTTPException(
            status_code=500, detail="Analysis history could not be deleted."
        ) from None

    try:
        _remove_analysis_files([analysis_id], list(artifact_paths))
        if video_deleted:
            _remove_uploaded_video_files([video.storage_path])
    except OSError:
        logger.exception(
            "Analysis %s was deleted, but some output files remain", analysis_id
        )
        raise HTTPException(
            status_code=500,
            detail="History was deleted, but some analysis files could not be removed.",
        ) from None
    return {"deleted": True, "video_deleted": video_deleted}


@router.get(
    "/{analysis_id}/artifacts/{kind}",
    responses={
        401: {"model": APIError, "description": "Authentication required"},
        403: {"model": APIError, "description": "Analysis belongs to another user"},
        404: {"model": APIError, "description": "Analysis or artifact not found"},
        409: {"model": APIError, "description": "Analysis is not completed"},
    },
)
def read_analysis_artifact(
    analysis_id: uuid.UUID,
    kind: ArtifactKind,
    session: SessionDep,
    current_user: CurrentUser,
    download: Annotated[bool, Query()] = False,
) -> FileResponse:
    row = session.exec(
        select(Analysis, Project)
        .join(Video, Analysis.video_id == Video.id)
        .join(Project, Video.project_id == Project.id)
        .where(Analysis.id == analysis_id)
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    analysis, project = row
    if project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Analysis belongs to another user")
    if analysis.status != AnalysisStatus.SUCCEEDED:
        raise HTTPException(
            status_code=409, detail="Analysis has not completed successfully"
        )

    artifact = session.exec(
        select(Artifact).where(
            Artifact.analysis_id == analysis.id,
            Artifact.kind == kind,
        )
    ).first()
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")

    data_root = Path(settings.LOCAL_DATA_DIR).resolve()
    artifact_path = (data_root / artifact.storage_path).resolve()
    if not artifact_path.is_relative_to(data_root) or not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file is unavailable")

    filename = {
        ArtifactKind.TRACKED_VIDEO: "tracked.mp4",
        ArtifactKind.BROWSER_VIDEO: "browser.mp4",
        ArtifactKind.OBSERVATIONS_CSV: "observations.csv",
        ArtifactKind.SUMMARY: "summary.json",
        ArtifactKind.MATCH_SUMMARY: "match_summary.json",
        ArtifactKind.PASSES_CSV: "passes.csv",
        ArtifactKind.PASSING_NETWORK_CSV: "passing_network.csv",
        ArtifactKind.TEAM_STATISTICS_CSV: "team_statistics.csv",
        ArtifactKind.PLAYER_STATISTICS_CSV: "player_statistics.csv",
        ArtifactKind.POSSESSION_SEQUENCES_CSV: "possession_sequences.csv",
        ArtifactKind.LOG: "analysis.log",
    }.get(kind)
    return FileResponse(
        artifact_path,
        media_type=artifact.media_type,
        filename=filename,
        content_disposition_type="attachment" if download else "inline",
        headers={"Cache-Control": "private, no-store"},
    )


@router.post(
    "/{analysis_id}/cancel",
    response_model=AnalysisPublic,
    responses={
        401: {"model": APIError, "description": "Authentication required"},
        403: {"model": APIError, "description": "Analysis belongs to another user"},
        404: {"model": APIError, "description": "Analysis not found"},
    },
)
def cancel_analysis(
    analysis_id: uuid.UUID, session: SessionDep, current_user: CurrentUser
) -> AnalysisPublic:
    statement = (
        select(Analysis, Video, Project)
        .join(Video, Analysis.video_id == Video.id)
        .join(Project, Video.project_id == Project.id)
        .where(Analysis.id == analysis_id)
    )
    row = session.exec(statement).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    analysis, video, project = row
    if project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Analysis belongs to another user")

    if analysis.status in (AnalysisStatus.QUEUED, AnalysisStatus.RUNNING):
        queued_cancellation = session.exec(
            update(Analysis)
            .where(
                Analysis.id == analysis_id,
                Analysis.status == AnalysisStatus.QUEUED,
            )
            .values(
                status=AnalysisStatus.CANCELLED,
                finished_at=get_datetime_utc(),
                error_code=None,
                error_message=None,
            )
        )
        if queued_cancellation.rowcount == 0:
            session.exec(
                update(Analysis)
                .where(
                    Analysis.id == analysis_id,
                    Analysis.status == AnalysisStatus.RUNNING,
                )
                .values(status=AnalysisStatus.CANCEL_REQUESTED)
            )
        session.commit()
        session.refresh(analysis)

    has_artifacts = (
        session.exec(
            select(Artifact.id)
            .where(Artifact.analysis_id == analysis.id)
            .limit(1)
        ).first()
        is not None
    )
    return _analysis_public(
        analysis,
        video,
        project.id,
        has_artifacts,
        _analysis_artifacts(session, analysis.id),
    )
