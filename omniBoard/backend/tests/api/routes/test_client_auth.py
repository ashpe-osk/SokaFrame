from fastapi.testclient import TestClient
from sqlmodel import Session

from app import crud
from app.core.config import settings
from app.core.cookie_auth import AUTH_COOKIE_NAME, CSRF_COOKIE_NAME, CSRF_HEADER_NAME
from app.models import UserCreate
from tests.utils.utils import random_email, random_lower_string

CLIENT_AUTH_PREFIX = "/api/auth"


def get_csrf(client: TestClient) -> str:
    response = client.get(f"{CLIENT_AUTH_PREFIX}/csrf")
    assert response.status_code == 200
    assert CSRF_COOKIE_NAME in client.cookies
    return response.json()["csrf"]


def test_client_auth_register_requires_login_before_session_and_logout(
    client: TestClient,
) -> None:
    csrf = get_csrf(client)

    existing_email = random_email()
    existing_password = random_lower_string()
    existing_register = client.post(
        f"{CLIENT_AUTH_PREFIX}/register",
        json={"email": existing_email, "password": existing_password},
        headers={CSRF_HEADER_NAME: csrf},
    )
    assert existing_register.status_code == 201, existing_register.text
    existing_login = client.post(
        f"{CLIENT_AUTH_PREFIX}/login",
        json={"email": existing_email, "password": existing_password},
        headers={CSRF_HEADER_NAME: csrf},
    )
    assert existing_login.status_code == 200, existing_login.text
    assert AUTH_COOKIE_NAME in client.cookies

    email = random_email()
    password = random_lower_string()
    response = client.post(
        f"{CLIENT_AUTH_PREFIX}/register",
        json={"email": email, "password": password},
        headers={CSRF_HEADER_NAME: csrf},
    )

    assert response.status_code == 201, response.text
    user = response.json()
    assert user["email"] == email
    assert user["access"] == {"board": True, "media": True}
    assert AUTH_COOKIE_NAME not in client.cookies
    assert f"{AUTH_COOKIE_NAME}=" in response.headers["set-cookie"]
    assert "hashed_password" not in user
    assert client.get(f"{CLIENT_AUTH_PREFIX}/me").status_code == 401

    login = client.post(
        f"{CLIENT_AUTH_PREFIX}/login",
        json={"email": email, "password": password},
        headers={CSRF_HEADER_NAME: csrf},
    )
    assert login.status_code == 200, login.text
    assert AUTH_COOKIE_NAME in client.cookies
    assert "httponly" in login.headers["set-cookie"].lower()

    me = client.get(f"{CLIENT_AUTH_PREFIX}/me")
    assert me.status_code == 200
    assert me.json()["id"] == user["id"]

    logout = client.post(
        f"{CLIENT_AUTH_PREFIX}/logout",
        headers={CSRF_HEADER_NAME: csrf},
    )
    assert logout.status_code == 200
    assert AUTH_COOKIE_NAME not in client.cookies
    assert client.get(f"{CLIENT_AUTH_PREFIX}/me").status_code == 401


def test_client_auth_login_uses_existing_password_authentication(
    client: TestClient, db: Session
) -> None:
    email = random_email()
    password = random_lower_string()
    user = crud.create_user(
        session=db,
        user_create=UserCreate(email=email, password=password),
    )
    csrf = get_csrf(client)
    response = client.post(
        f"{CLIENT_AUTH_PREFIX}/login",
        json={"email": email, "password": password},
        headers={CSRF_HEADER_NAME: csrf},
    )

    assert response.status_code == 200, response.text
    assert response.json()["id"] == str(user.id)
    assert AUTH_COOKIE_NAME in client.cookies
    assert client.get(f"{CLIENT_AUTH_PREFIX}/me").status_code == 200


def test_client_auth_rejects_missing_or_invalid_csrf(client: TestClient) -> None:
    missing = client.post(
        f"{CLIENT_AUTH_PREFIX}/login",
        json={"email": "nobody@example.com", "password": "not-a-password"},
    )
    assert missing.status_code == 403
    assert missing.json()["code"] == "csrf_failed"

    csrf = get_csrf(client)
    invalid = client.post(
        f"{CLIENT_AUTH_PREFIX}/login",
        json={"email": "nobody@example.com", "password": "not-a-password"},
        headers={CSRF_HEADER_NAME: csrf + "tampered"},
    )
    assert invalid.status_code == 403
    assert invalid.json()["code"] == "csrf_failed"


def test_cookie_auth_requires_csrf_for_protected_mutations(
    client: TestClient, db: Session
) -> None:
    email = random_email()
    password = random_lower_string()
    user = crud.create_user(
        session=db,
        user_create=UserCreate(email=email, password=password),
    )
    csrf = get_csrf(client)
    login = client.post(
        f"{CLIENT_AUTH_PREFIX}/login",
        json={"email": email, "password": password},
        headers={CSRF_HEADER_NAME: csrf},
    )
    assert login.status_code == 200

    api_path = f"{settings.API_V1_STR}/items/"
    no_csrf = client.post(api_path, json={"title": "cookie protected"})
    assert no_csrf.status_code == 403

    with_csrf = client.post(
        api_path,
        json={"title": "cookie protected"},
        headers={CSRF_HEADER_NAME: csrf},
    )
    assert with_csrf.status_code == 200
    assert with_csrf.json()["owner_id"] == str(user.id)


def test_client_auth_cors_allows_local_frontends_with_credentials(
    client: TestClient,
) -> None:
    for origin in (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
    ):
        response = client.options(
            f"{CLIENT_AUTH_PREFIX}/login",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,x-omni-csrf",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin
        assert response.headers["access-control-allow-credentials"] == "true"


def test_client_auth_me_rejects_unauthenticated_requests(client: TestClient) -> None:
    client.cookies.clear()
    response = client.get(f"{CLIENT_AUTH_PREFIX}/me")
    assert response.status_code == 401
