import hashlib
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, select

from app.core.config import settings
from app.models import Analysis, AnalysisStatus, Artifact, ArtifactKind, Project, Video
from tests.utils.user import user_authentication_headers
from tests.utils.utils import random_email, random_lower_string

_MP4_BYTES = b"\x00\x00\x00\x10ftypisom" + b"\x00" * 4


def _create_project(
    client: TestClient, headers: dict[str, str], name: str = "Match"
) -> str:
    response = client.post(
        f"{settings.API_V1_STR}/projects/",
        headers=headers,
        json={"name": name},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _upload_analysis(
    client: TestClient,
    headers: dict[str, str],
    project_id: str,
    filename: str = "match.mp4",
    content: bytes = _MP4_BYTES,
):
    return client.post(
        f"{settings.API_V1_STR}/analyses/",
        headers=headers,
        data={"project_id": project_id},
        files={"video_file": (filename, content, "application/octet-stream")},
    )


def _create_second_user(
    client: TestClient,
) -> dict[str, str]:
    email = random_email()
    password = random_lower_string()
    signup = client.post(
        f"{settings.API_V1_STR}/users/signup",
        json={"email": email, "password": password},
    )
    assert signup.status_code == 200, signup.text
    return user_authentication_headers(client=client, email=email, password=password)


@pytest.fixture
def local_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(settings, "LOCAL_DATA_DIR", tmp_path)
    return tmp_path


def test_projects_require_auth_and_are_owned(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
) -> None:
    unauthenticated = client.get(f"{settings.API_V1_STR}/projects/")
    assert unauthenticated.status_code == 401

    project_id = _create_project(client, normal_user_token_headers, "  Owned  ")
    project = client.get(
        f"{settings.API_V1_STR}/projects/{project_id}",
        headers=normal_user_token_headers,
    )
    assert project.status_code == 200
    assert project.json()["name"] == "Owned"
    assert "owner_id" not in project.json()

    other_headers = _create_second_user(client)
    forbidden = client.get(
        f"{settings.API_V1_STR}/projects/{project_id}", headers=other_headers
    )
    assert forbidden.status_code == 403
    other_project_id = _create_project(client, other_headers, "Other user's project")
    own_projects = client.get(
        f"{settings.API_V1_STR}/projects/", headers=normal_user_token_headers
    )
    assert own_projects.status_code == 200
    assert other_project_id not in {
        item["id"] for item in own_projects.json()["data"]
    }

    unauthenticated_analyses = client.get(f"{settings.API_V1_STR}/analyses/")
    assert unauthenticated_analyses.status_code == 401


def test_upload_creates_queued_analysis_and_hides_paths(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    local_data_dir: Path,
) -> None:
    project_id = _create_project(client, normal_user_token_headers)
    response = _upload_analysis(
        client, normal_user_token_headers, project_id, r"..\..\match.mp4"
    )

    assert response.status_code == 201, response.text
    result = response.json()
    assert result["project_id"] == project_id
    assert result["status"] == "queued"
    assert result["video_original_filename"] == "match.mp4"
    assert result["video_size_bytes"] == len(_MP4_BYTES)
    assert result["video_sha256"] == hashlib.sha256(_MP4_BYTES).hexdigest()
    assert result["video_media_type"] == "video/mp4"
    assert result["result_available"] is False
    assert "storage_path" not in result
    assert "output_dir" not in result
    assert str(local_data_dir) not in response.text

    stored_files = list((local_data_dir / "uploads").glob("*.mp4"))
    assert len(stored_files) == 1
    assert stored_files[0].read_bytes() == _MP4_BYTES
    assert stored_files[0].name != r"..\..\match.mp4"

    analysis_id = result["id"]
    read_response = client.get(
        f"{settings.API_V1_STR}/analyses/{analysis_id}",
        headers=normal_user_token_headers,
    )
    assert read_response.status_code == 200
    assert read_response.json() == result

    listed = client.get(
        f"{settings.API_V1_STR}/analyses/", headers=normal_user_token_headers
    )
    assert listed.status_code == 200
    assert any(item["id"] == analysis_id for item in listed.json()["data"])


def test_analysis_ownership_and_project_scoped_listing(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    local_data_dir: Path,
) -> None:
    owner_headers = _create_second_user(client)
    owner_project_id = _create_project(client, owner_headers, "Private project")
    created = _upload_analysis(
        client, owner_headers, owner_project_id
    )
    analysis_id = created.json()["id"]

    denied_read = client.get(
        f"{settings.API_V1_STR}/analyses/{analysis_id}",
        headers=normal_user_token_headers,
    )
    assert denied_read.status_code == 403
    visible_to_other = client.get(
        f"{settings.API_V1_STR}/analyses/",
        headers=normal_user_token_headers,
    )
    assert visible_to_other.status_code == 200
    assert analysis_id not in {item["id"] for item in visible_to_other.json()["data"]}

    existing_uploads = set((local_data_dir / "uploads").glob("*.mp4"))
    foreign_project_id = _create_project(client, owner_headers, "Not yours")
    denied_upload = _upload_analysis(
        client, normal_user_token_headers, foreign_project_id
    )
    assert denied_upload.status_code == 403
    assert set((local_data_dir / "uploads").glob("*.mp4")) == existing_uploads


def test_client_project_analysis_history_is_persistent_and_owner_scoped(
    client: TestClient,
    db: Session,
    normal_user_token_headers: dict[str, str],
    local_data_dir: Path,
) -> None:
    client_project_id = f"cv-{uuid4()}"
    project_path = f"/api/projects/media/{client_project_id}"
    payload = {
        "id": client_project_id,
        "type": "media",
        "kind": "cv-analysis",
        "title": "Kenya vs Uganda",
    }
    saved = client.put(
        project_path,
        headers=normal_user_token_headers,
        json={"payload": payload},
    )
    assert saved.status_code == 200, saved.text
    project = db.exec(
        select(Project).where(Project.client_id == client_project_id)
    ).one()

    first = _upload_analysis(
        client, normal_user_token_headers, str(project.id), "match_video_1.mp4"
    )
    second = _upload_analysis(
        client, normal_user_token_headers, str(project.id), "match_video_2.mp4"
    )
    assert first.status_code == second.status_code == 201

    completed = db.get(Analysis, UUID(first.json()["id"]))
    assert completed is not None
    completed.status = AnalysisStatus.SUCCEEDED
    completed.frames_processed = 2
    completed.total_frames = 2
    db.add(completed)
    artifacts = {
        ArtifactKind.TRACKED_VIDEO: (
            "outputs/history/tracked.mp4",
            b"tracked video",
            "video/mp4",
        ),
        ArtifactKind.OBSERVATIONS_CSV: (
            "outputs/history/observations.csv",
            b"frame,track_id\n1,7\n",
            "text/csv",
        ),
    }
    for kind, (storage_path, content, media_type) in artifacts.items():
        artifact_path = local_data_dir / storage_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(content)
        db.add(
            Artifact(
                analysis_id=completed.id,
                kind=kind,
                storage_path=storage_path,
                media_type=media_type,
                file_size_bytes=len(content),
            )
        )
    db.commit()

    history_url = (
        f"{settings.API_V1_STR}/analyses/by-project/media/{client_project_id}"
    )
    history = client.get(history_url, headers=normal_user_token_headers)
    assert history.status_code == 200, history.text
    assert history.json()["count"] == 2
    assert {
        item["video_original_filename"] for item in history.json()["data"]
    } == {"match_video_1.mp4", "match_video_2.mp4"}
    assert {item["id"] for item in history.json()["data"]} == {
        first.json()["id"],
        second.json()["id"],
    }
    completed_entry = next(
        item for item in history.json()["data"] if item["id"] == first.json()["id"]
    )
    assert completed_entry["status"] == "succeeded"
    assert completed_entry["frames_processed"] == completed_entry["total_frames"] == 2
    assert {
        artifact["kind"] for artifact in completed_entry["artifacts"]
    } == {"tracked_video", "observations_csv"}
    for kind, (_, content, _) in artifacts.items():
        downloaded = client.get(
            f"{settings.API_V1_STR}/analyses/{completed.id}/artifacts/{kind}?download=true",
            headers=normal_user_token_headers,
        )
        assert downloaded.status_code == 200
        assert downloaded.content == content

    other_headers = _create_second_user(client)
    denied = client.get(history_url, headers=other_headers)
    assert denied.status_code == 404


@pytest.mark.parametrize(
    ("analysis_status", "frames_processed", "total_frames"),
    [
        (AnalysisStatus.QUEUED, 0, None),
        (AnalysisStatus.RUNNING, 42, 100),
        (AnalysisStatus.SUCCEEDED, 100, 100),
        (AnalysisStatus.FAILED, 42, 100),
        (AnalysisStatus.CANCELLED, 42, 100),
    ],
)
def test_analysis_status_endpoint_returns_persisted_state_and_progress(
    client: TestClient,
    db: Session,
    normal_user_token_headers: dict[str, str],
    local_data_dir: Path,
    analysis_status: AnalysisStatus,
    frames_processed: int,
    total_frames: int | None,
) -> None:
    project_id = _create_project(client, normal_user_token_headers)
    created = _upload_analysis(client, normal_user_token_headers, project_id)
    analysis_id = UUID(created.json()["id"])
    analysis = db.get(Analysis, analysis_id)
    assert analysis is not None
    analysis.status = analysis_status
    analysis.frames_processed = frames_processed
    analysis.total_frames = total_frames
    db.add(analysis)
    db.commit()

    response = client.get(
        f"{settings.API_V1_STR}/analyses/{analysis_id}",
        headers=normal_user_token_headers,
    )

    assert len(list((local_data_dir / "uploads").glob("*.mp4"))) == 1
    assert response.status_code == 200
    assert response.json()["status"] == analysis_status
    assert response.json()["frames_processed"] == frames_processed
    assert response.json()["total_frames"] == total_frames


def test_analysis_status_endpoint_requires_authentication_and_enforces_ownership(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    local_data_dir: Path,
) -> None:
    owner_headers = _create_second_user(client)
    project_id = _create_project(client, owner_headers, "Private analysis")
    created = _upload_analysis(client, owner_headers, project_id)
    analysis_id = created.json()["id"]
    assert len(list((local_data_dir / "uploads").glob("*.mp4"))) == 1

    client.cookies.clear()
    unauthenticated = client.get(f"{settings.API_V1_STR}/analyses/{analysis_id}")
    assert unauthenticated.status_code == 401

    denied = client.get(
        f"{settings.API_V1_STR}/analyses/{analysis_id}",
        headers=normal_user_token_headers,
    )
    assert denied.status_code == 403


def test_completed_analysis_exposes_summary_and_authorized_artifacts(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    local_data_dir: Path,
    db: Session,
) -> None:
    project_id = _create_project(client, normal_user_token_headers)
    created = _upload_analysis(client, normal_user_token_headers, project_id)
    analysis_id = created.json()["id"]
    premature = client.get(
        f"{settings.API_V1_STR}/analyses/{analysis_id}/artifacts/browser_video",
        headers=normal_user_token_headers,
    )
    assert premature.status_code == 409
    analysis = db.get(Analysis, UUID(analysis_id))
    assert analysis is not None
    analysis.status = AnalysisStatus.SUCCEEDED
    analysis.summary = {
        "schema_version": 1,
        "status": "succeeded",
        "artifacts": ["tracked.mp4", "observations.csv", "summary.json"],
    }

    artifact_dir = local_data_dir / "outputs" / analysis_id
    artifact_dir.mkdir(parents=True)
    video_path = artifact_dir / "tracked.mp4"
    video_path.write_bytes(_MP4_BYTES)
    browser_video_path = artifact_dir / "browser.mp4"
    browser_video_path.write_bytes(_MP4_BYTES)
    observations_path = artifact_dir / "observations.csv"
    observations_path.write_bytes(b"frame,track_id\n0,1\n")
    summary_path = artifact_dir / "summary.json"
    summary_path.write_text('{"schema_version":"1.0","status":"succeeded"}')
    passes_path = artifact_dir / "passes.csv"
    passes_path.write_text("pass_id,outcome\n1,completed\n", encoding="utf-8")
    match_summary_path = artifact_dir / "match_summary.json"
    match_summary_path.write_text(
        '{"pass_statistics":{"attempted":1}}', encoding="utf-8"
    )
    db.add_all(
        [
            Artifact(
                analysis_id=analysis.id,
                kind=ArtifactKind.TRACKED_VIDEO,
                storage_path=video_path.relative_to(local_data_dir).as_posix(),
                media_type="video/mp4",
                file_size_bytes=video_path.stat().st_size,
            ),
            Artifact(
                analysis_id=analysis.id,
                kind=ArtifactKind.BROWSER_VIDEO,
                storage_path=browser_video_path.relative_to(local_data_dir).as_posix(),
                media_type="video/mp4",
                file_size_bytes=browser_video_path.stat().st_size,
            ),
            Artifact(
                analysis_id=analysis.id,
                kind=ArtifactKind.OBSERVATIONS_CSV,
                storage_path=observations_path.relative_to(local_data_dir).as_posix(),
                media_type="text/csv",
                file_size_bytes=observations_path.stat().st_size,
            ),
            Artifact(
                analysis_id=analysis.id,
                kind=ArtifactKind.SUMMARY,
                storage_path=summary_path.relative_to(local_data_dir).as_posix(),
                media_type="application/json",
                file_size_bytes=summary_path.stat().st_size,
            ),
            Artifact(
                analysis_id=analysis.id,
                kind=ArtifactKind.PASSES_CSV,
                storage_path=passes_path.relative_to(local_data_dir).as_posix(),
                media_type="text/csv",
                file_size_bytes=passes_path.stat().st_size,
            ),
            Artifact(
                analysis_id=analysis.id,
                kind=ArtifactKind.MATCH_SUMMARY,
                storage_path=match_summary_path.relative_to(local_data_dir).as_posix(),
                media_type="application/json",
                file_size_bytes=match_summary_path.stat().st_size,
            ),
            Artifact(
                analysis_id=analysis.id,
                kind=ArtifactKind.LOG,
                storage_path=f"outputs/{analysis_id}/missing.log",
                media_type="text/plain",
                file_size_bytes=1,
            ),
        ]
    )
    db.add(analysis)
    db.commit()

    result = client.get(
        f"{settings.API_V1_STR}/analyses/{analysis_id}",
        headers=normal_user_token_headers,
    )
    assert result.status_code == 200
    assert result.json()["summary"]["schema_version"] == 1
    assert {item["kind"] for item in result.json()["artifacts"]} == {
        "tracked_video",
        "browser_video",
        "observations_csv",
        "summary",
        "passes_csv",
        "match_summary",
        "log",
    }
    assert all("storage_path" not in item for item in result.json()["artifacts"])

    video = client.get(
        f"{settings.API_V1_STR}/analyses/{analysis_id}/artifacts/tracked_video",
        headers=normal_user_token_headers,
    )
    assert video.status_code == 200
    assert video.content == _MP4_BYTES
    assert video.headers["content-type"].startswith("video/mp4")
    assert "attachment" not in video.headers.get("content-disposition", "")
    downloaded_video = client.get(
        f"{settings.API_V1_STR}/analyses/{analysis_id}/artifacts/browser_video?download=true",
        headers=normal_user_token_headers,
    )
    assert downloaded_video.status_code == 200
    assert downloaded_video.content == _MP4_BYTES
    assert downloaded_video.headers["content-type"].startswith("video/mp4")
    assert 'attachment; filename="browser.mp4"' in downloaded_video.headers[
        "content-disposition"
    ]
    ranged_video = client.get(
        f"{settings.API_V1_STR}/analyses/{analysis_id}/artifacts/tracked_video",
        headers={**normal_user_token_headers, "Range": "bytes=0-3"},
    )
    assert ranged_video.status_code == 206
    assert ranged_video.content == _MP4_BYTES[:4]
    assert ranged_video.headers["content-range"].startswith("bytes 0-3/")

    browser_video = client.get(
        f"{settings.API_V1_STR}/analyses/{analysis_id}/artifacts/browser_video",
        headers=normal_user_token_headers,
    )
    assert browser_video.status_code == 200
    assert browser_video.content == _MP4_BYTES
    assert browser_video.headers["content-type"].startswith("video/mp4")
    assert browser_video.headers["accept-ranges"] == "bytes"
    ranged_browser_video = client.get(
        f"{settings.API_V1_STR}/analyses/{analysis_id}/artifacts/browser_video",
        headers={**normal_user_token_headers, "Range": "bytes=0-3"},
    )
    assert ranged_browser_video.status_code == 206
    assert ranged_browser_video.content == _MP4_BYTES[:4]
    assert ranged_browser_video.headers["content-range"].startswith("bytes 0-3/")

    observations = client.get(
        f"{settings.API_V1_STR}/analyses/{analysis_id}/artifacts/observations_csv",
        headers=normal_user_token_headers,
    )
    assert observations.status_code == 200
    assert observations.text == "frame,track_id\n0,1\n"
    assert 'filename="observations.csv"' in observations.headers["content-disposition"]
    summary = client.get(
        f"{settings.API_V1_STR}/analyses/{analysis_id}/artifacts/summary",
        headers=normal_user_token_headers,
    )
    assert summary.status_code == 200
    assert summary.headers["content-type"].startswith("application/json")
    assert summary.json()["schema_version"] == "1.0"
    downloaded_summary = client.get(
        f"{settings.API_V1_STR}/analyses/{analysis_id}/artifacts/summary?download=true",
        headers=normal_user_token_headers,
    )
    assert downloaded_summary.status_code == 200
    assert 'attachment; filename="summary.json"' in downloaded_summary.headers[
        "content-disposition"
    ]
    downloaded_passes = client.get(
        f"{settings.API_V1_STR}/analyses/{analysis_id}/artifacts/passes_csv?download=true",
        headers=normal_user_token_headers,
    )
    assert downloaded_passes.status_code == 200
    assert downloaded_passes.text.splitlines() == [
        "pass_id,outcome",
        "1,completed",
    ]
    assert 'attachment; filename="passes.csv"' in downloaded_passes.headers[
        "content-disposition"
    ]
    downloaded_match_summary = client.get(
        f"{settings.API_V1_STR}/analyses/{analysis_id}/artifacts/match_summary?download=true",
        headers=normal_user_token_headers,
    )
    assert downloaded_match_summary.status_code == 200
    assert downloaded_match_summary.json()["pass_statistics"]["attempted"] == 1
    assert 'attachment; filename="match_summary.json"' in downloaded_match_summary.headers[
        "content-disposition"
    ]

    forbidden = client.get(
        f"{settings.API_V1_STR}/analyses/{analysis_id}/artifacts/browser_video",
        headers=_create_second_user(client),
    )
    assert forbidden.status_code == 403
    forbidden_passes = client.get(
        f"{settings.API_V1_STR}/analyses/{analysis_id}/artifacts/passes_csv",
        headers=_create_second_user(client),
    )
    assert forbidden_passes.status_code == 403
    client.cookies.clear()
    unauthenticated_artifact = client.get(
        f"{settings.API_V1_STR}/analyses/{analysis_id}/artifacts/browser_video"
    )
    assert unauthenticated_artifact.status_code == 401
    missing = client.get(
        f"{settings.API_V1_STR}/analyses/{analysis_id}/artifacts/other",
        headers=normal_user_token_headers,
    )
    assert missing.status_code == 404
    missing_file = client.get(
        f"{settings.API_V1_STR}/analyses/{analysis_id}/artifacts/log",
        headers=normal_user_token_headers,
    )
    assert missing_file.status_code == 404

    browser_artifact = db.exec(
        select(Artifact).where(
            Artifact.analysis_id == analysis.id,
            Artifact.kind == ArtifactKind.BROWSER_VIDEO,
        )
    ).one()
    browser_artifact.storage_path = "../../../outside.mp4"
    db.add(browser_artifact)
    db.commit()
    traversal = client.get(
        f"{settings.API_V1_STR}/analyses/{analysis_id}/artifacts/browser_video",
        headers=normal_user_token_headers,
    )
    assert traversal.status_code == 404


@pytest.mark.parametrize(
    ("filename", "content", "expected_status"),
    [
        ("match.txt", _MP4_BYTES, 415),
        ("match.mp4", b"not a video file", 400),
    ],
)
def test_invalid_upload_formats_are_rejected(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    local_data_dir: Path,
    filename: str,
    content: bytes,
    expected_status: int,
) -> None:
    project_id = _create_project(client, normal_user_token_headers)
    response = _upload_analysis(
        client, normal_user_token_headers, project_id, filename, content
    )

    assert response.status_code == expected_status
    temporary_directory = local_data_dir / "uploads" / ".tmp"
    assert not temporary_directory.exists() or list(temporary_directory.iterdir()) == []
    assert list((local_data_dir / "uploads").glob("*.mp4")) == []


def test_oversized_and_missing_uploads_are_rejected(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    local_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id = _create_project(client, normal_user_token_headers)
    monkeypatch.setattr(settings, "MAX_VIDEO_UPLOAD_BYTES", len(_MP4_BYTES) - 1)
    oversized = _upload_analysis(
        client, normal_user_token_headers, project_id
    )
    assert oversized.status_code == 413

    missing = client.post(
        f"{settings.API_V1_STR}/analyses/",
        headers=normal_user_token_headers,
        data={"project_id": project_id},
    )
    assert missing.status_code == 422
    assert list((local_data_dir / "uploads").glob("*.mp4")) == []


def test_unknown_and_invalid_project_or_analysis_ids(
    client: TestClient, normal_user_token_headers: dict[str, str]
) -> None:
    missing_project = _upload_analysis(
        client, normal_user_token_headers, str(uuid4())
    )
    assert missing_project.status_code == 404

    invalid_project = client.post(
        f"{settings.API_V1_STR}/analyses/",
        headers=normal_user_token_headers,
        data={"project_id": "not-a-uuid"},
        files={"video_file": ("match.mp4", _MP4_BYTES, "video/mp4")},
    )
    assert invalid_project.status_code == 422

    missing_analysis = client.get(
        f"{settings.API_V1_STR}/analyses/{uuid4()}",
        headers=normal_user_token_headers,
    )
    assert missing_analysis.status_code == 404
    invalid_analysis = client.get(
        f"{settings.API_V1_STR}/analyses/not-a-uuid",
        headers=normal_user_token_headers,
    )
    assert invalid_analysis.status_code == 422


def test_database_failure_removes_committed_upload(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    local_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id = _create_project(client, normal_user_token_headers)

    def fail_commit(_session: Session) -> None:
        raise SQLAlchemyError("simulated database failure")

    monkeypatch.setattr(Session, "commit", fail_commit)
    response = _upload_analysis(
        client, normal_user_token_headers, project_id
    )

    assert response.status_code == 500
    assert "simulated database failure" not in response.text
    assert list((local_data_dir / "uploads").glob("*.mp4")) == []


def test_cancel_analysis_requests_cancellation_for_owned_jobs(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    db: Session,
) -> None:
    project_id = _create_project(client, normal_user_token_headers)
    created = _upload_analysis(
        client, normal_user_token_headers, project_id
    )
    analysis_id = created.json()["id"]

    cancellation = client.post(
        f"{settings.API_V1_STR}/analyses/{analysis_id}/cancel",
        headers=normal_user_token_headers,
    )
    assert cancellation.status_code == 200
    assert cancellation.json()["status"] == "cancelled"
    assert cancellation.json()["finished_at"] is not None
    assert db.get(Analysis, UUID(analysis_id)).status == AnalysisStatus.CANCELLED

    other_headers = _create_second_user(client)
    denied = client.post(
        f"{settings.API_V1_STR}/analyses/{analysis_id}/cancel",
        headers=other_headers,
    )
    assert denied.status_code == 403

    missing = client.post(
        f"{settings.API_V1_STR}/analyses/{uuid4()}/cancel",
        headers=normal_user_token_headers,
    )
    assert missing.status_code == 404


def test_cancel_running_analysis_requests_worker_termination(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    db: Session,
) -> None:
    project_id = _create_project(client, normal_user_token_headers)
    created = _upload_analysis(client, normal_user_token_headers, project_id)
    analysis_id = UUID(created.json()["id"])
    analysis = db.get(Analysis, analysis_id)
    assert analysis is not None
    analysis.status = AnalysisStatus.RUNNING
    db.add(analysis)
    db.commit()

    cancellation = client.post(
        f"{settings.API_V1_STR}/analyses/{analysis_id}/cancel",
        headers=normal_user_token_headers,
    )

    assert cancellation.status_code == 200
    assert cancellation.json()["status"] == "cancel_requested"
    db.expire_all()
    assert db.get(Analysis, analysis_id).status == AnalysisStatus.CANCEL_REQUESTED


def test_delete_analysis_removes_outputs_and_unused_source_video(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    local_data_dir: Path,
    db: Session,
) -> None:
    project_id = _create_project(client, normal_user_token_headers)
    created = _upload_analysis(client, normal_user_token_headers, project_id)
    analysis_id = UUID(created.json()["id"])
    analysis = db.get(Analysis, analysis_id)
    assert analysis is not None
    video = db.get(Video, analysis.video_id)
    assert video is not None
    video_id = video.id
    source_video_path = local_data_dir / video.storage_path
    analysis.status = AnalysisStatus.SUCCEEDED

    run_dir = local_data_dir / "analyses" / str(analysis_id) / "cv-output" / "run_1"
    run_dir.mkdir(parents=True)
    tracked_video = run_dir / "tracked.mp4"
    tracked_video.write_bytes(_MP4_BYTES)
    db.add(
        Artifact(
            analysis_id=analysis_id,
            kind=ArtifactKind.TRACKED_VIDEO,
            storage_path=tracked_video.relative_to(local_data_dir).as_posix(),
            media_type="video/mp4",
            file_size_bytes=tracked_video.stat().st_size,
        )
    )
    db.add(analysis)
    db.commit()

    deleted = client.delete(
        f"{settings.API_V1_STR}/analyses/{analysis_id}",
        headers=normal_user_token_headers,
    )

    assert deleted.status_code == 200, deleted.text
    assert deleted.json() == {"deleted": True, "video_deleted": True}
    assert db.get(Analysis, analysis_id) is None
    assert db.get(Video, video_id) is None
    assert db.exec(
        select(Artifact).where(Artifact.analysis_id == analysis_id)
    ).all() == []
    assert not (local_data_dir / "analyses" / str(analysis_id)).exists()
    assert not source_video_path.exists()
    assert list((local_data_dir / "uploads").glob("*.mp4")) == []

    missing = client.delete(
        f"{settings.API_V1_STR}/analyses/{analysis_id}",
        headers=normal_user_token_headers,
    )
    assert missing.status_code == 404


def test_delete_analysis_enforces_owner_and_terminal_status(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    db: Session,
) -> None:
    owner_headers = _create_second_user(client)
    project_id = _create_project(client, owner_headers, "Private analysis")
    created = _upload_analysis(client, owner_headers, project_id)
    analysis_id = UUID(created.json()["id"])

    active = client.delete(
        f"{settings.API_V1_STR}/analyses/{analysis_id}",
        headers=owner_headers,
    )
    assert active.status_code == 409
    assert db.get(Analysis, analysis_id) is not None

    analysis = db.get(Analysis, analysis_id)
    assert analysis is not None
    analysis.status = AnalysisStatus.CANCELLED
    db.add(analysis)
    db.commit()

    denied = client.delete(
        f"{settings.API_V1_STR}/analyses/{analysis_id}",
        headers=normal_user_token_headers,
    )
    assert denied.status_code == 403
    assert db.get(Analysis, analysis_id) is not None

    client.cookies.clear()
    unauthenticated = client.delete(f"{settings.API_V1_STR}/analyses/{analysis_id}")
    assert unauthenticated.status_code == 401


def test_clear_completed_project_history_preserves_active_and_other_users(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    local_data_dir: Path,
    db: Session,
) -> None:
    client_project_id = "media-project-for-clear-history"
    project_record_id = UUID(_create_project(client, normal_user_token_headers))
    project = db.get(Project, project_record_id)
    assert project is not None
    project.type = "media"
    project.client_id = client_project_id
    db.add(project)
    db.commit()

    created_analyses = [
        _upload_analysis(client, normal_user_token_headers, str(project_record_id))
        for _ in range(3)
    ]
    analysis_ids = [UUID(response.json()["id"]) for response in created_analyses]
    statuses = (
        AnalysisStatus.SUCCEEDED,
        AnalysisStatus.INTERRUPTED,
        AnalysisStatus.RUNNING,
    )
    for analysis_id, analysis_status in zip(analysis_ids, statuses, strict=True):
        analysis = db.get(Analysis, analysis_id)
        assert analysis is not None
        analysis.status = analysis_status
        db.add(analysis)
        if analysis_status in {
            AnalysisStatus.SUCCEEDED,
            AnalysisStatus.FAILED,
            AnalysisStatus.CANCELLED,
            AnalysisStatus.INTERRUPTED,
        }:
            output_dir = local_data_dir / "analyses" / str(analysis_id)
            output_dir.mkdir(parents=True)
            (output_dir / "partial.bin").write_bytes(b"output")
    db.commit()

    other_headers = _create_second_user(client)
    other_project_id = UUID(
        _create_project(client, other_headers, "Another user's project")
    )
    other_project = db.get(Project, other_project_id)
    assert other_project is not None
    other_project.type = "media"
    other_project.client_id = client_project_id
    db.add(other_project)
    db.commit()
    other_created = _upload_analysis(
        client, other_headers, str(other_project_id)
    )
    other_analysis_id = UUID(other_created.json()["id"])
    other_analysis = db.get(Analysis, other_analysis_id)
    assert other_analysis is not None
    other_analysis.status = AnalysisStatus.SUCCEEDED
    db.add(other_analysis)
    db.commit()

    cleared = client.delete(
        f"{settings.API_V1_STR}/analyses/by-project/media/"
        f"{client_project_id}/completed",
        headers=normal_user_token_headers,
    )

    assert cleared.status_code == 200, cleared.text
    assert cleared.json() == {"deleted_count": 2}
    assert db.get(Analysis, analysis_ids[0]) is None
    assert db.get(Analysis, analysis_ids[1]) is None
    assert db.get(Analysis, analysis_ids[2]) is not None
    assert db.get(Analysis, other_analysis_id) is not None
    assert not (local_data_dir / "analyses" / str(analysis_ids[0])).exists()
    assert not (local_data_dir / "analyses" / str(analysis_ids[1])).exists()
    assert len(list((local_data_dir / "uploads").glob("*.mp4"))) == 4

    cleared_again = client.delete(
        f"{settings.API_V1_STR}/analyses/by-project/media/"
        f"{client_project_id}/completed",
        headers=normal_user_token_headers,
    )
    assert cleared_again.status_code == 200
    assert cleared_again.json() == {"deleted_count": 0}


def test_delete_analysis_keeps_source_video_when_another_analysis_uses_it(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    local_data_dir: Path,
    db: Session,
) -> None:
    project_id = _create_project(client, normal_user_token_headers)
    created = _upload_analysis(client, normal_user_token_headers, project_id)
    analysis_id = UUID(created.json()["id"])
    analysis = db.get(Analysis, analysis_id)
    assert analysis is not None
    analysis.status = AnalysisStatus.SUCCEEDED
    db.add(analysis)
    db.commit()

    video = db.get(Video, analysis.video_id)
    assert video is not None
    source_video_path = local_data_dir / video.storage_path
    another_analysis = client.post(
        f"{settings.API_V1_STR}/analyses/from-video",
        headers=normal_user_token_headers,
        json={"project_id": project_id, "video_id": str(video.id)},
    )
    assert another_analysis.status_code == 201, another_analysis.text

    deleted = client.delete(
        f"{settings.API_V1_STR}/analyses/{analysis_id}",
        headers=normal_user_token_headers,
    )

    assert deleted.status_code == 200, deleted.text
    assert deleted.json() == {"deleted": True, "video_deleted": False}
    db.expire_all()
    assert db.get(Analysis, analysis_id) is None
    assert db.get(Analysis, UUID(another_analysis.json()["id"])) is not None
    assert db.get(Video, video.id) is not None
    assert source_video_path.exists()


def test_clear_all_project_analyses_removes_owned_records_and_files(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    local_data_dir: Path,
    db: Session,
) -> None:
    client_project_id = "media-project-for-clear-all-history"
    project_record_id = UUID(_create_project(client, normal_user_token_headers))
    project = db.get(Project, project_record_id)
    assert project is not None
    project.type = "media"
    project.client_id = client_project_id
    db.add(project)
    db.commit()

    created_analyses = [
        _upload_analysis(
            client,
            normal_user_token_headers,
            project_id=str(project_record_id),
        )
        for _ in range(3)
    ]
    analysis_ids = [UUID(response.json()["id"]) for response in created_analyses]
    video_ids: list[UUID] = []
    source_paths: list[Path] = []
    statuses = (
        AnalysisStatus.SUCCEEDED,
        AnalysisStatus.FAILED,
        AnalysisStatus.CANCELLED,
    )
    for analysis_id, analysis_status in zip(analysis_ids, statuses, strict=True):
        analysis = db.get(Analysis, analysis_id)
        assert analysis is not None
        analysis.status = analysis_status
        db.add(analysis)
        video = db.get(Video, analysis.video_id)
        assert video is not None
        video_ids.append(video.id)
        source_paths.append(local_data_dir / video.storage_path)
        output_dir = local_data_dir / "analyses" / str(analysis_id)
        output_dir.mkdir(parents=True)
        (output_dir / "partial.bin").write_bytes(b"output")
    db.commit()

    other_headers = _create_second_user(client)
    other_project_id = UUID(
        _create_project(client, other_headers, "Another user's project")
    )
    other_project = db.get(Project, other_project_id)
    assert other_project is not None
    other_project.type = "media"
    other_project.client_id = client_project_id
    db.add(other_project)
    db.commit()
    other_created = _upload_analysis(
        client, other_headers, str(other_project_id)
    )
    other_analysis_id = UUID(other_created.json()["id"])
    other_analysis = db.get(Analysis, other_analysis_id)
    assert other_analysis is not None
    other_analysis.status = AnalysisStatus.SUCCEEDED
    db.add(other_analysis)
    db.commit()
    other_video = db.get(Video, other_analysis.video_id)
    assert other_video is not None
    other_source_path = local_data_dir / other_video.storage_path

    cleared = client.delete(
        f"{settings.API_V1_STR}/analyses/by-project/media/{client_project_id}",
        headers=normal_user_token_headers,
    )

    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["deleted_count"] == 3
    assert set(cleared.json()["deleted_video_ids"]) == {
        str(video_id) for video_id in video_ids
    }
    for analysis_id in analysis_ids:
        assert db.get(Analysis, analysis_id) is None
        assert not (local_data_dir / "analyses" / str(analysis_id)).exists()
    for video_id, source_path in zip(video_ids, source_paths, strict=True):
        assert db.get(Video, video_id) is None
        assert not source_path.exists()
    assert db.get(Analysis, other_analysis_id) is not None
    assert db.get(Video, other_video.id) is not None
    assert other_source_path.exists()


def test_clear_all_project_analyses_refuses_when_a_job_is_active(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    local_data_dir: Path,
    db: Session,
) -> None:
    client_project_id = "media-project-with-active-analysis"
    project_record_id = UUID(_create_project(client, normal_user_token_headers))
    project = db.get(Project, project_record_id)
    assert project is not None
    project.type = "media"
    project.client_id = client_project_id
    db.add(project)
    db.commit()

    created = _upload_analysis(
        client, normal_user_token_headers, str(project_record_id)
    )
    analysis_id = UUID(created.json()["id"])
    analysis = db.get(Analysis, analysis_id)
    assert analysis is not None
    analysis.status = AnalysisStatus.RUNNING
    db.add(analysis)
    db.commit()
    video = db.get(Video, analysis.video_id)
    assert video is not None
    source_path = local_data_dir / video.storage_path

    cleared = client.delete(
        f"{settings.API_V1_STR}/analyses/by-project/media/{client_project_id}",
        headers=normal_user_token_headers,
    )

    assert cleared.status_code == 409
    assert db.get(Analysis, analysis_id) is not None
    assert db.get(Video, video.id) is not None
    assert source_path.exists()


def test_phase3_migration_keeps_legacy_project_unowned(
    client: TestClient, normal_user_token_headers: dict[str, str], db: Session
) -> None:
    legacy_project = Project(name="Legacy unowned project")
    db.add(legacy_project)
    db.commit()
    db.refresh(legacy_project)

    listed = client.get(
        f"{settings.API_V1_STR}/projects/", headers=normal_user_token_headers
    )
    assert listed.status_code == 200
    assert legacy_project.id not in {
        item["id"] for item in listed.json()["data"]
    }
    assert db.get(Project, legacy_project.id).owner_id is None
