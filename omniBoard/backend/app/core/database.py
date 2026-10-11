from sqlite3 import Connection as SQLiteConnection

from sqlalchemy import event
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.pool import ConnectionPoolEntry
from sqlmodel import create_engine


def create_database_engine(database_url: str) -> Engine:
    url = make_url(database_url)
    is_sqlite = url.get_backend_name() == "sqlite"
    connect_args = {"check_same_thread": False} if is_sqlite else {}
    engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)

    if is_sqlite:

        @event.listens_for(engine, "connect")
        def enable_sqlite_foreign_keys(
            dbapi_connection: SQLiteConnection,
            _connection_record: ConnectionPoolEntry,
        ) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
            finally:
                cursor.close()

    return engine
