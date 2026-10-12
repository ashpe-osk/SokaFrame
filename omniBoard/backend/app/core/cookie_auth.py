import hashlib
import hmac
import secrets

from fastapi import HTTPException, Request, status

AUTH_COOKIE_NAME = "omniboard_session"
MEDIA_PLAYBACK_COOKIE_NAME = "omniboard_media_playback"
CSRF_COOKIE_NAME = "omniboard_csrf"
CSRF_HEADER_NAME = "X-Omni-CSRF"


def create_csrf_token(secret_key: str) -> str:
    nonce = secrets.token_urlsafe(32)
    signature = hmac.new(
        secret_key.encode("utf-8"), nonce.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"{nonce}.{signature}"


def validate_csrf(request: Request, secret_key: str) -> None:
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME, "")
    header_token = request.headers.get(CSRF_HEADER_NAME, "")
    if not cookie_token or not hmac.compare_digest(cookie_token, header_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF validation failed",
        )

    try:
        nonce, signature = cookie_token.rsplit(".", 1)
    except ValueError:
        nonce, signature = "", ""
    expected = hmac.new(
        secret_key.encode("utf-8"), nonce.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    if not nonce or not hmac.compare_digest(signature, expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF validation failed",
        )
