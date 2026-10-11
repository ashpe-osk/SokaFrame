from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.models import Analysis, AnalysisStatus
from app.worker.manager import FakeProcessor, LocalJobManager
from tests.utils.user import user_authentication_headers
from tests.utils.utils import random_email, random_lower_string

CLIENT_PROJECTS_PREFIX = "/api/projects"
ANALYSES_URL = f"{settings.API_V1_STR}/analyses/from-video"
MP4_BYTES = b"\x00\x00\x00\x10ftypisom" + b"\x00" * 4


def _user_headers(client: TestClient) -> dict[str, str]:
    email = random_email()
    password = random_lower_string()
    signup = client.post(
        f"{settings.API_V1_STR}/users/signup",
        json={"email": email, "password": password},
    )
    assert signup.status_code == 200, signup.text
    return user_authentication_headers(client=client, email=email, password=password)


def _create_project(client: TestClient, headers: dict[str, str]) -> str:
    client_id = str(uuid4())
    response = client.put(
        f"{CLIENT_PROJECTS_PREFIX}/media/{client_id}",
        headers=headers,
        json={
            "payload": {
                "id": client_id,
                "type": "media",
                "title": "Analysis dispatch test",
            }
        },
    )
    assert response.status_code == 200, response.text
    return client_id


def _upload_video(
    client: TestClient, headers: dict[str, str], client_project_id: str
) -> dict[str, str]:
    response = client.post(
        f"{CLIENT_PROJECTS_PREFIX}/media/{client_project_id}/videos",
        headers=headers,
        files={"video_file": ("match.mp4", MP4_BYTES, "video/mp4")},
    )
    assert response.status_code == 201, response.text
    return response.json()


@pytest.fixture
def local_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(settings, "LOCAL_DATA_DIR", tmp_path)
    return tmp_path


def test_uploaded_video_creates_queued_analysis_claimable_by_worker(
    client: TestClient,
    db: Session,
    local_data_dir: Path,
) -> None:
    headers = _user_headers(client)
    client_project_id = _create_project(client, headers)
    video = _upload_video(client, headers, client_project_id)
    project_record_id = UUID(video["project_record_id"])

    response = client.post(
        ANALYSES_URL,
        headers=headers,
        json={"project_id": video["project_record_id"], "video_id": video["video_id"]},
    )

    assert response.status_code == 201, response.text
    result = response.json()
    analysis_id = UUID(result["id"])
    assert result["project_id"] == str(project_record_id)
    assert result["video_id"] == video["video_id"]
    assert result["status"] == AnalysisStatus.QUEUED

    analysis = db.get(Analysis, analysis_id)
    assert analysis is not None
    assert analysis.video_id == UUID(video["video_id"])
    assert analysis.status == AnalysisStatus.QUEUED

    # Put this job first so the manager claims this API-created row even when
    # the shared test database contains queued rows from other API tests.
    analysis.created_at = datetime(2000, 1, 1, tzinfo=UTC)
    db.add(analysis)
    db.commit()
    manager = LocalJobManager(
        processor=FakeProcessor(),
        database_engine=db.get_bind(),
        data_dir=local_data_dir,
    )
    job = manager.claim_next()
    assert job is not None
    assert job.id == analysis_id
    assert job.video_id == UUID(video["video_id"])
    assert db.get(Analysis, analysis_id).status == AnalysisStatus.RUNNING


def test_video_analysis_requires_authentication(
    client: TestClient,
) -> None:
    headers = _user_headers(client)
    client_project_id = _create_project(client, headers)
    video = _upload_video(client, headers, client_project_id)
    client.cookies.clear()

    response = client.post(
        ANALYSES_URL,
        json={"project_id": video["project_record_id"], "video_id": video["video_id"]},
    )
    assert response.status_code == 401


def test_video_analysis_enforces_project_and_video_ownership(
    client: TestClient,
) -> None:
    owner_headers = _user_headers(client)
    owner_project_id = _create_project(client, owner_headers)
    video = _upload_video(client, owner_headers, owner_project_id)

    other_headers = _user_headers(client)
    other_project_id = _create_project(client, other_headers)
    other_video = _upload_video(client, other_headers, other_project_id)

    foreign_project = client.post(
        ANALYSES_URL,
        headers=other_headers,
        json={
            "project_id": video["project_record_id"],
            "video_id": video["video_id"],
        },
    )
    assert foreign_project.status_code == 403

    mismatched_video = client.post(
        ANALYSES_URL,
        headers=other_headers,
        json={
            "project_id": other_video["project_record_id"],
            "video_id": video["video_id"],
        },
    )
    assert mismatched_video.status_code == 404
    assert other_video["project_id"] != video["project_id"]


def test_repeated_video_analysis_submissions_create_distinct_jobs(
    client: TestClient,
) -> None:
    headers = _user_headers(client)
    client_project_id = _create_project(client, headers)
    video = _upload_video(client, headers, client_project_id)
    request_body = {
        "project_id": video["project_record_id"],
        "video_id": video["video_id"],
    }

    first = client.post(ANALYSES_URL, headers=headers, json=request_body)
    second = client.post(ANALYSES_URL, headers=headers, json=request_body)

    assert first.status_code == second.status_code == 201
    assert first.json()["id"] != second.json()["id"]
