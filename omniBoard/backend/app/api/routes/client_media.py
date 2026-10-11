import logging
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, Path, UploadFile, status
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, select

from app.api.deps import CurrentUser, SessionDep
from app.core.config import settings
from app.models import Project, Video
from app.services.video_storage import (
    InvalidVideoFile,
    StoredVideo,
    UnsupportedVideoFormat,
    UploadTooLarge,
    store_video_upload,
)

logger = logging.getLogger(__name__)
ProjectType = Literal["board", "media"]
router = APIRouter(prefix="/projects", tags=["client-media"])


def _owned_client_project(
    session: Session, user_id: UUID, project_type: ProjectType, client_id: str
) -> Project:
    project = session.exec(
        select(Project).where(
            Project.owner_id == user_id,
            Project.type == project_type,
            Project.client_id == client_id,
        )
    ).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def _video_public(video: Video, project: Project) -> dict[str, object]:
    return {
        "id": str(video.id),
        "video_id": str(video.id),
        "media_id": str(video.id),
        "project_id": project.client_id or str(project.id),
        "project_record_id": str(project.id),
        "project_type": project.type,
        "filename": video.original_filename,
        "original_filename": video.original_filename,
        "media_type": video.media_type,
        "size_bytes": video.file_size_bytes,
        "file_size_bytes": video.file_size_bytes,
        "sha256": video.sha256,
        "created_at": video.created_at.isoformat(),
    }


def _store_metadata(session: Session, project: Project, stored: StoredVideo) -> Video:
    video = Video(
        project_id=project.id,
        original_filename=stored.original_filename,
        media_type=stored.media_type,
        storage_path=stored.storage_path,
        file_size_bytes=stored.file_size_bytes,
        sha256=stored.sha256,
    )
    session.add(video)
    try:
        session.commit()
        session.refresh(video)
    except SQLAlchemyError:
        session.rollback()
        try:
            stored.remove()
        except OSError:
            logger.exception("Failed to clean up video after metadata commit failure")
        logger.exception("Failed to persist uploaded video metadata")
        raise HTTPException(
            status_code=500, detail="The uploaded video could not be registered."
        ) from None
    return video


@router.post(
    "/{project_type}/{client_id}/videos",
    status_code=status.HTTP_201_CREATED,
)
async def upload_project_video(
    project_type: ProjectType,
    client_id: Annotated[str, Path(min_length=1, max_length=255)],
    session: SessionDep,
    current_user: CurrentUser,
    video_file: Annotated[UploadFile, File()],
) -> dict[str, object]:
    project = _owned_client_project(session, current_user.id, project_type, client_id)
    try:
        stored = await store_video_upload(
            video_file, settings.LOCAL_DATA_DIR, settings.MAX_VIDEO_UPLOAD_BYTES
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
        logger.exception("Failed to store uploaded project video")
        raise HTTPException(
            status_code=500, detail="The uploaded video could not be stored."
        ) from None
    video = _store_metadata(session, project, stored)
    return _video_public(video, project)


@router.get("/{project_type}/{client_id}/videos")
def list_project_videos(
    project_type: ProjectType,
    client_id: Annotated[str, Path(min_length=1, max_length=255)],
    session: SessionDep,
    current_user: CurrentUser,
) -> dict[str, list[dict[str, object]]]:
    project = _owned_client_project(session, current_user.id, project_type, client_id)
    videos = session.exec(
        select(Video)
        .where(Video.project_id == project.id)
        .order_by(Video.created_at.desc(), Video.id.desc())
    ).all()
    return {"videos": [_video_public(video, project) for video in videos]}
