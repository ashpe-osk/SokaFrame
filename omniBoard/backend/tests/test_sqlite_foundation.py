import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy import inspect
from sqlmodel import Session, delete, select

from app.core.db import engine
from app.models import Item, User


@pytest.mark.usefixtures("db")
def test_migrations_create_template_schema() -> None:
    table_names = set(inspect(engine).get_table_names())

    assert {"alembic_version", "item", "user"} <= table_names


@pytest.mark.usefixtures("db")
def test_sqlite_foreign_keys_are_enabled() -> None:
    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1


def test_uuid_and_utc_timestamps_round_trip(db: Session) -> None:
    user = User(
        email="sqlite-foundation@example.com",
        hashed_password="test-password-hash",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    stored_user = db.exec(select(User).where(User.id == user.id)).one()

    assert isinstance(stored_user.id, uuid.UUID)
    assert isinstance(stored_user.created_at, datetime)
    assert stored_user.created_at is not None
    assert stored_user.created_at.tzinfo is not None
    assert stored_user.created_at.utcoffset() == timedelta(0)

    db.delete(stored_user)
    db.commit()


def test_deleting_user_cascades_to_items(db: Session) -> None:
    user = User(
        email="sqlite-cascade@example.com",
        hashed_password="test-password-hash",
    )
    item = Item(title="Cascade check", owner=user)
    db.add(user)
    db.add(item)
    db.commit()
    user_id = user.id
    item_id = item.id

    db.exec(delete(User).where(User.id == user_id))
    db.commit()

    assert db.exec(select(Item).where(Item.id == item_id)).first() is None
