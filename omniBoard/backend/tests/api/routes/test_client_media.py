from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.models import Project, Video
from tests.utils.user import user_authentication_headers
from tests.utils.utils import random_email, random_lower_string

CLIENT_PROJECTS_PREFIX = "/api/projects"
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


def _create_client_project(
    client: TestClient, headers: dict[str, str], project_type: str = "media"
) -> str:
    client_id = str(uuid4())
    response = client.put(
        f"{CLIENT_PROJECTS_PREFIX}/{project_type}/{client_id}",
        headers=headers,
        json={
            "payload": {
                "schemaVersion": 1,
                "id": client_id,
                "type": project_type,
                "title": "Football analysis workspace",
            }
        },
    )
    assert response.status_code == 200, response.text
    return client_id


def _upload(
    client: TestClient,
    headers: dict[str, str] | None,
    project_id: str,
    filename: str = "match.mp4",
    content: bytes = MP4_BYTES,
):
    return client.post(
        f"{CLIENT_PROJECTS_PREFIX}/media/{project_id}/videos",
        headers=headers,
        files={"video_file": (filename, content, "application/octet-stream")},
    )


@pytest.fixture
def local_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(settings, "LOCAL_DATA_DIR", tmp_path)
    return tmp_path


def test_authenticated_video_upload_persists_metadata_and_file(
    client: TestClient,
    db: Session,
    local_data_dir: Path,
) -> None:
    headers = _user_headers(client)
    project_id = _create_client_project(client, headers)

    response = _upload(client, headers, project_id, r"..\matches\final.mp4")
    assert response.status_code == 201, response.text
    metadata = response.json()
    assert metadata["id"] == metadata["video_id"] == metadata["media_id"]
    assert metadata["project_id"] == project_id
    assert metadata["project_type"] == "media"
    assert metadata["filename"] == "final.mp4"
    assert metadata["original_filename"] == "final.mp4"
    assert metadata["media_type"] == "video/mp4"
    assert metadata["size_bytes"] == metadata["file_size_bytes"] == len(MP4_BYTES)
    assert metadata["sha256"]

    video = db.get(Video, UUID(metadata["id"]))
    assert video is not None
    assert video.project_id is not None
    project = db.get(Project, video.project_id)
    assert project is not None
    assert project.client_id == project_id
    file_path = local_data_dir / video.storage_path
    assert file_path.exists()
    assert file_path.read_bytes() == MP4_BYTES

    listed = client.get(
        f"{CLIENT_PROJECTS_PREFIX}/media/{project_id}/videos", headers=headers
    )
    assert listed.status_code == 200
    assert listed.json()["videos"] == [metadata]


def test_video_upload_requires_authentication(
    client: TestClient,
    local_data_dir: Path,
) -> None:
    headers = _user_headers(client)
    project_id = _create_client_project(client, headers)
    client.cookies.clear()

    response = _upload(client, None, project_id)
    assert response.status_code == 401
    assert list((local_data_dir / "uploads").glob("*.mp4")) == []


def test_video_upload_enforces_project_ownership(
    client: TestClient,
    local_data_dir: Path,
) -> None:
    owner_headers = _user_headers(client)
    project_id = _create_client_project(client, owner_headers)
    other_headers = _user_headers(client)

    response = _upload(client, other_headers, project_id)
    assert response.status_code == 404
    assert list((local_data_dir / "uploads").glob("*.mp4")) == []

    owner_videos = client.get(
        f"{CLIENT_PROJECTS_PREFIX}/media/{project_id}/videos",
        headers=owner_headers,
    )
    assert owner_videos.status_code == 200
    assert owner_videos.json()["videos"] == []
    assert (
        client.get(
            f"{CLIENT_PROJECTS_PREFIX}/media/{project_id}/videos",
            headers=other_headers,
        ).status_code
        == 404
    )


@pytest.mark.parametrize(
    ("filename", "content", "expected_status"),
    [
        ("match.exe", MP4_BYTES, 415),
        ("match.mp4", b"not a supported video header", 400),
    ],
)
def test_video_upload_rejects_unsupported_or_invalid_files(
    client: TestClient,
    local_data_dir: Path,
    filename: str,
    content: bytes,
    expected_status: int,
) -> None:
    headers = _user_headers(client)
    project_id = _create_client_project(client, headers)

    response = _upload(client, headers, project_id, filename, content)
    assert response.status_code == expected_status
    temporary_dir = local_data_dir / "uploads" / ".tmp"
    assert not temporary_dir.exists() or list(temporary_dir.iterdir()) == []
    assert list((local_data_dir / "uploads").glob("*.mp4")) == []


def test_video_upload_enforces_configured_size_limit(
    client: TestClient,
    local_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = _user_headers(client)
    project_id = _create_client_project(client, headers)
    monkeypatch.setattr(settings, "MAX_VIDEO_UPLOAD_BYTES", len(MP4_BYTES) - 1)

    response = _upload(client, headers, project_id)
    assert response.status_code == 413
    temporary_dir = local_data_dir / "uploads" / ".tmp"
    assert not temporary_dir.exists() or list(temporary_dir.iterdir()) == []
    assert list((local_data_dir / "uploads").glob("*.mp4")) == []
