from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, delete, select, update

from app.models import (
    Analysis,
    AnalysisCreate,
    AnalysisStatus,
    Artifact,
    ArtifactCreate,
    ArtifactKind,
    Project,
    ProjectCreate,
    Video,
    VideoCreate,
)


def test_project_input_trims_name_and_rejects_blank_name() -> None:
    project = ProjectCreate(name="  Match review  ")

    assert project.name == "Match review"
    with pytest.raises(ValidationError):
        ProjectCreate(name="   ")


@pytest.mark.parametrize(
    "storage_path",
    [
        "../outside/video.mp4",
        "uploads/../outside/video.mp4",
        "C:\\outside\\video.mp4",
        "\\\\server\\share\\video.mp4",
        "/outside/video.mp4",
    ],
)
def test_video_input_rejects_paths_outside_storage_root(storage_path: str) -> None:
    with pytest.raises(ValidationError):
        VideoCreate(
            original_filename="match.mp4",
            storage_path=storage_path,
            file_size_bytes=100,
        )


def test_domain_schemas_reject_invalid_checksum_and_non_json_options() -> None:
    with pytest.raises(ValidationError):
        VideoCreate(
            original_filename="match.mp4",
            storage_path="uploads/match.mp4",
            file_size_bytes=100,
            sha256="not-a-sha256",
        )

    with pytest.raises(ValidationError):
        AnalysisCreate(video_id=uuid4(), options={"unsupported": object()})

    with pytest.raises(ValidationError):
        ArtifactCreate(
            analysis_id=uuid4(),
            kind=ArtifactKind.SUMMARY,
            storage_path="analyses/../outside/summary.json",
            media_type="application/json",
            file_size_bytes=1,
        )


def test_project_video_analysis_artifact_relationships_and_delete_rules(
    db: Session,
) -> None:
    unique = uuid4().hex
    project = Project(name=f"Fixture {unique}")
    video = Video(
        project=project,
        original_filename="match.mp4",
        storage_path=f"uploads/{unique}/source.mp4",
        file_size_bytes=4096,
        sha256="a" * 64,
        duration_seconds=600.0,
        fps=25.0,
        frame_width=1920,
        frame_height=1080,
    )
    analysis = Analysis(video=video, options={"device": "cpu", "max_frames": 100})
    artifact = Artifact(
        analysis=analysis,
        kind=ArtifactKind.OBSERVATIONS_CSV,
        storage_path=f"analyses/{unique}/observations.csv",
        media_type="text/csv",
        file_size_bytes=512,
    )
    db.add(project)
    db.commit()
    db.refresh(video)
    db.refresh(analysis)
    db.refresh(artifact)
    artifact_id = artifact.id

    assert video.project_id == project.id
    assert analysis.video_id == video.id
    assert analysis.status is AnalysisStatus.QUEUED
    assert analysis.options == {"device": "cpu", "max_frames": 100}
    assert analysis.frames_processed == 0
    assert artifact.analysis_id == analysis.id
    assert artifact.kind is ArtifactKind.OBSERVATIONS_CSV
    assert analysis.created_at.tzinfo is not None
    assert video.created_at.tzinfo is not None

    with pytest.raises(IntegrityError, match="ck_analysis_status_valid"):
        db.exec(
            update(Analysis)
            .where(Analysis.id == analysis.id)
            .values(status="not_a_status")
        )
        db.commit()
    db.rollback()

    original_updated_at = project.updated_at
    project.name = f"Updated fixture {unique}"
    db.add(project)
    db.commit()
    db.refresh(project)
    assert project.updated_at > original_updated_at

    with pytest.raises(IntegrityError):
        db.exec(delete(Video).where(Video.id == video.id))
        db.commit()
    db.rollback()

    db.exec(delete(Project).where(Project.id == project.id))
    db.commit()
    db.refresh(video)
    assert video.project_id is None

    db.exec(delete(Analysis).where(Analysis.id == analysis.id))
    db.commit()
    assert db.exec(select(Artifact).where(Artifact.id == artifact_id)).first() is None

    db.exec(delete(Video).where(Video.id == video.id))
    db.commit()
