import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy.engine import URL
from sqlmodel import Session, delete

_TEST_DATABASE_DIRECTORY = tempfile.TemporaryDirectory(prefix="omniboard-sports-tests-")
_TEST_DATABASE_PATH = Path(_TEST_DATABASE_DIRECTORY.name) / "test.db"
os.environ["DATABASE_URL"] = URL.create(
    "sqlite", database=_TEST_DATABASE_PATH.as_posix()
).render_as_string(hide_password=False)
os.environ["FASTAPI_ENV"] = "development"

from app.core.config import settings  # noqa: E402
from app.core.db import engine, init_db  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Item, User  # noqa: E402
from tests.utils.user import authentication_token_from_email  # noqa: E402
from tests.utils.utils import get_superuser_token_headers  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def db() -> Generator[Session]:
    database_path = Path(engine.url.database or "").resolve()
    test_directory = Path(_TEST_DATABASE_DIRECTORY.name).resolve()
    if engine.dialect.name != "sqlite" or not database_path.is_relative_to(
        test_directory
    ):
        raise RuntimeError(
            "Backend tests must use the isolated temporary SQLite database."
        )

    command.upgrade(Config("alembic.ini"), "head")
    with Session(engine) as session:
        init_db(session)
        yield session
        statement = delete(Item)
        session.execute(statement)
        statement = delete(User)
        session.execute(statement)
        session.commit()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    del session, exitstatus
    engine.dispose()
    _TEST_DATABASE_DIRECTORY.cleanup()


@pytest.fixture(scope="module")
def client() -> Generator[TestClient]:
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def superuser_token_headers(client: TestClient) -> dict[str, str]:
    return get_superuser_token_headers(client)


@pytest.fixture(scope="module")
def normal_user_token_headers(client: TestClient, db: Session) -> dict[str, str]:
    return authentication_token_from_email(
        client=client, email=settings.EMAIL_TEST_USER, db=db
    )
