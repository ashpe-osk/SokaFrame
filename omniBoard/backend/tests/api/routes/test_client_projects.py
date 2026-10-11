from uuid import uuid4

from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.cookie_auth import CSRF_COOKIE_NAME, CSRF_HEADER_NAME
from tests.utils.user import user_authentication_headers
from tests.utils.utils import random_email, random_lower_string

PROJECTS_PREFIX = "/api/projects"


def _create_user_headers(client: TestClient) -> dict[str, str]:
    email = random_email()
    password = random_lower_string()
    signup = client.post(
        f"{settings.API_V1_STR}/users/signup",
        json={"email": email, "password": password},
    )
    assert signup.status_code == 200, signup.text
    return user_authentication_headers(client=client, email=email, password=password)


def _payload(
    project_id: str, project_type: str = "board", title: str = "Test project"
) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "id": project_id,
        "type": project_type,
        "title": title,
        "createdAt": "2026-10-07T00:00:00.000Z",
        "updatedAt": "2026-10-07T00:00:00.000Z",
        "settings": {"pitchColor": "#318447"},
        "boardState": {
            "tokens": [{"team": "home", "number": 9, "x": "25%", "y": "40%"}]
        },
        "mediaState": None,
        "analysisState": None,
    }


def test_client_project_crud_preserves_payload_and_type(client: TestClient) -> None:
    headers = _create_user_headers(client)
    project_id = f"legacy-{uuid4()}"
    payload = _payload(project_id)
    path = f"{PROJECTS_PREFIX}/board/{project_id}"

    create = client.put(path, headers=headers, json={"payload": payload})
    assert create.status_code == 200, create.text
    assert create.json() == {"payload": payload, "kind": "tactical"}

    project_list = client.get(f"{PROJECTS_PREFIX}/board", headers=headers)
    assert project_list.status_code == 200
    item = next(
        project
        for project in project_list.json()["projects"]
        if project["id"] == project_id
    )
    assert item["type"] == "board"
    assert item["kind"] == "tactical"
    assert item["title"] == payload["title"]
    assert "updatedAt" in item

    read = client.get(path, headers=headers)
    assert read.status_code == 200
    assert read.json() == {"payload": payload, "kind": "tactical"}

    updated_payload = _payload(project_id, title="Saved title")
    updated_payload["boardState"] = {"tokens": [], "scenes": ["second"]}
    update = client.put(path, headers=headers, json={"payload": updated_payload})
    assert update.status_code == 200
    assert update.json() == {"payload": updated_payload, "kind": "tactical"}
    assert client.get(path, headers=headers).json() == {
        "payload": updated_payload,
        "kind": "tactical",
    }

    delete = client.delete(path, headers=headers)
    assert delete.status_code == 200
    assert delete.json() == {"ok": True}
    assert client.get(path, headers=headers).status_code == 404
    assert all(
        project["id"] != project_id
        for project in client.get(f"{PROJECTS_PREFIX}/board", headers=headers).json()[
            "projects"
        ]
    )


def test_client_projects_are_authenticated_and_owner_scoped(
    client: TestClient,
) -> None:
    client.cookies.clear()
    project_id = str(uuid4())
    path = f"{PROJECTS_PREFIX}/board/{project_id}"
    unauthenticated_read = client.get(path)
    assert unauthenticated_read.status_code == 401
    unauthenticated_write = client.put(path, json={"payload": _payload(project_id)})
    assert unauthenticated_write.status_code == 401
    unauthenticated_delete = client.delete(path)
    assert unauthenticated_delete.status_code == 401

    owner_headers = _create_user_headers(client)
    owner_payload = _payload(project_id, title="Owner snapshot")
    assert (
        client.put(
            path, headers=owner_headers, json={"payload": owner_payload}
        ).status_code
        == 200
    )

    other_headers = _create_user_headers(client)
    assert client.get(path, headers=other_headers).status_code == 404

    other_payload = _payload(project_id, title="Other user's snapshot")
    other_write = client.put(
        path, headers=other_headers, json={"payload": other_payload}
    )
    assert other_write.status_code == 404
    assert client.get(path, headers=owner_headers).json() == {
        "payload": owner_payload,
        "kind": "tactical",
    }

    assert client.delete(path, headers=other_headers).status_code == 404
    assert client.get(path, headers=owner_headers).json() == {
        "payload": owner_payload,
        "kind": "tactical",
    }
    assert client.delete(path, headers=owner_headers).status_code == 200


def test_client_project_types_are_separate_and_validated(
    client: TestClient,
) -> None:
    headers = _create_user_headers(client)
    project_id = "default"
    board_payload = _payload(project_id, "board", "Board default")
    media_payload = _payload(project_id, "media", "Media default")
    board_path = f"{PROJECTS_PREFIX}/board/{project_id}"
    media_path = f"{PROJECTS_PREFIX}/media/{project_id}"

    assert (
        client.put(
            board_path, headers=headers, json={"payload": board_payload}
        ).status_code
        == 200
    )
    assert (
        client.put(
            media_path, headers=headers, json={"payload": media_payload}
        ).status_code
        == 200
    )
    assert client.get(board_path, headers=headers).json() == {
        "payload": board_payload,
        "kind": "tactical",
    }
    assert client.get(media_path, headers=headers).json() == {
        "payload": media_payload,
        "kind": "tactical",
    }
    assert (
        client.get(f"{PROJECTS_PREFIX}/board", headers=headers).json()["projects"][0][
            "type"
        ]
        == "board"
    )
    assert (
        client.get(f"{PROJECTS_PREFIX}/media", headers=headers).json()["projects"][0][
            "type"
        ]
        == "media"
    )

    mismatched = client.put(
        board_path, headers=headers, json={"payload": media_payload}
    )
    assert mismatched.status_code == 422
    assert client.get(f"{PROJECTS_PREFIX}/invalid", headers=headers).status_code == 422


def test_client_project_kind_is_persisted_and_immutable(client: TestClient) -> None:
    headers = _create_user_headers(client)
    cv_id = f"cv-{uuid4()}"
    cv_path = f"{PROJECTS_PREFIX}/media/{cv_id}"
    cv_payload = _payload(cv_id, "media", "CV project")
    cv_payload["kind"] = "cv-analysis"

    created = client.put(cv_path, headers=headers, json={"payload": cv_payload})
    assert created.status_code == 200, created.text
    assert created.json()["kind"] == "cv-analysis"
    listed = client.get(f"{PROJECTS_PREFIX}/media", headers=headers)
    cv_project = next(
        project for project in listed.json()["projects"] if project["id"] == cv_id
    )
    assert cv_project["kind"] == "cv-analysis"
    assert client.get(cv_path, headers=headers).json()["kind"] == "cv-analysis"

    changed_payload = {**cv_payload, "kind": "tactical"}
    rejected_change = client.put(
        cv_path, headers=headers, json={"payload": changed_payload}
    )
    assert rejected_change.status_code == 422
    assert client.get(cv_path, headers=headers).json()["kind"] == "cv-analysis"

    tactical_id = f"tactical-{uuid4()}"
    tactical_payload = _payload(tactical_id, "media", "Legacy default")
    tactical = client.put(
        f"{PROJECTS_PREFIX}/media/{tactical_id}",
        headers=headers,
        json={"payload": tactical_payload},
    )
    assert tactical.status_code == 200
    assert tactical.json()["kind"] == "tactical"

    invalid_payload = {**tactical_payload, "kind": "unknown"}
    invalid = client.put(
        f"{PROJECTS_PREFIX}/media/{tactical_id}",
        headers=headers,
        json={"payload": invalid_payload},
    )
    assert invalid.status_code == 422


def test_project_create_kind_defaults_and_validation(client: TestClient) -> None:
    headers = _create_user_headers(client)
    projects_path = f"{settings.API_V1_STR}/projects/"

    cv = client.post(
        projects_path,
        headers=headers,
        json={"name": "CV API project", "kind": "cv-analysis"},
    )
    assert cv.status_code == 201, cv.text
    cv_record = cv.json()
    assert cv_record["kind"] == "cv-analysis"
    listed = client.get(projects_path, headers=headers)
    assert (
        next(item for item in listed.json()["data"] if item["id"] == cv_record["id"])[
            "kind"
        ]
        == "cv-analysis"
    )

    tactical = client.post(
        projects_path,
        headers=headers,
        json={"name": "Default tactical API project"},
    )
    assert tactical.status_code == 201, tactical.text
    assert tactical.json()["kind"] == "tactical"

    invalid = client.post(
        projects_path,
        headers=headers,
        json={"name": "Invalid kind", "kind": "unknown"},
    )
    assert invalid.status_code == 422


def test_client_project_cookie_auth_and_csrf(client: TestClient) -> None:
    client.cookies.clear()
    csrf_response = client.get("/api/auth/csrf")
    assert csrf_response.status_code == 200
    csrf = csrf_response.json()["csrf"]
    assert CSRF_COOKIE_NAME in client.cookies

    email = random_email()
    password = random_lower_string()
    register = client.post(
        "/api/auth/register",
        json={"email": email, "password": password},
        headers={CSRF_HEADER_NAME: csrf},
    )
    assert register.status_code == 201, register.text

    login = client.post(
        "/api/auth/login",
        json={"email": email, "password": password},
        headers={CSRF_HEADER_NAME: csrf},
    )
    assert login.status_code == 200, login.text

    project_id = "default"
    path = f"{PROJECTS_PREFIX}/board/{project_id}"
    payload = _payload(project_id)
    forbidden = client.put(path, json={"payload": payload})
    assert forbidden.status_code == 403

    save = client.put(
        path,
        json={"payload": payload},
        headers={CSRF_HEADER_NAME: csrf},
    )
    assert save.status_code == 200, save.text
    assert client.get(path).json() == {"payload": payload, "kind": "tactical"}
