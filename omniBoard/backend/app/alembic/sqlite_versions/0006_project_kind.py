"""Add a durable kind to client projects.

Revision ID: 0006_project_kind
Revises: 0005_client_project_snapshots
"""

from alembic import op

revision = "0006_project_kind"
down_revision = "0005_client_project_snapshots"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    columns = {
        row[1] for row in connection.exec_driver_sql("PRAGMA table_info(project)")
    }
    if "kind" not in columns:
        connection.exec_driver_sql(
            "ALTER TABLE project ADD COLUMN kind TEXT NOT NULL DEFAULT 'tactical'"
        )


def downgrade() -> None:
    raise RuntimeError(
        "The project kind migration is forward-only to preserve stored kind values."
    )
