from datetime import timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from app import crud
from app.api.deps import CurrentUser, SessionDep
from app.core import security
from app.core.config import settings
from app.core.cookie_auth import (
    AUTH_COOKIE_NAME,
    CSRF_COOKIE_NAME,
    create_csrf_token,
    validate_csrf,
)
from app.models import User, UserCreate, UserPublic, UserRegister

router = APIRouter(prefix="/auth", tags=["client-auth"])


def _public_user(user: User) -> dict[str, Any]:
    payload = UserPublic.model_validate(user).model_dump(mode="json")
    payload["access"] = {"board": True, "media": True}
    return payload


def _set_session_cookie(response: Response, request: Request, user: User) -> None:
    max_age = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    token = security.create_access_token(
        user.id,
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=token,
        max_age=max_age,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
        path="/",
    )


def _csrf_error_response() -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={"error": "CSRF validation failed", "code": "csrf_failed"},
    )


def _check_csrf(request: Request) -> None:
    validate_csrf(request, settings.SECRET_KEY)


@router.get("/csrf")
def get_csrf_token(request: Request, response: Response) -> dict[str, str]:
    token = create_csrf_token(settings.SECRET_KEY)
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=token,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
        path="/",
    )
    return {"csrf": token}


@router.post("/register")
def register(request: Request, session: SessionDep, user_in: UserRegister) -> Response:
    try:
        _check_csrf(request)
    except HTTPException:
        return _csrf_error_response()

    if crud.get_user_by_email(session=session, email=str(user_in.email)):
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "error": "The user with this email already exists.",
                "code": "email_exists",
            },
        )

    user = crud.create_user(
        session=session,
        user_create=UserCreate.model_validate(user_in),
    )
    response = JSONResponse(
        status_code=status.HTTP_201_CREATED, content=_public_user(user)
    )
    response.delete_cookie(
        key=AUTH_COOKIE_NAME,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
        path="/",
    )
    return response


@router.post("/login")
def login(
    request: Request, session: SessionDep, credentials: dict[str, Any]
) -> Response:
    try:
        _check_csrf(request)
    except HTTPException:
        return _csrf_error_response()

    email = credentials.get("email")
    password = credentials.get("password")
    if not isinstance(email, str) or not isinstance(password, str):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Email and password are required.",
                "code": "invalid_request",
            },
        )
    user = crud.authenticate(session=session, email=email, password=password)
    if user is None:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": "Incorrect email or password.",
                "code": "invalid_credentials",
            },
        )
    if not user.is_active:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"error": "Inactive user.", "code": "inactive_user"},
        )

    response = JSONResponse(content=_public_user(user))
    _set_session_cookie(response, request, user)
    return response


@router.get("/me")
def read_current_user(current_user: CurrentUser) -> dict[str, Any]:
    return _public_user(current_user)


@router.post("/logout")
def logout(request: Request) -> Response:
    try:
        _check_csrf(request)
    except HTTPException:
        return _csrf_error_response()
    response = JSONResponse(content={"message": "Logged out"})
    response.delete_cookie(
        key=AUTH_COOKIE_NAME,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
        path="/",
    )
    return response
