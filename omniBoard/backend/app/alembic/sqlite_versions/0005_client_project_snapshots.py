"""Persist client project types, identifiers, and snapshots.

Revision ID: 0005_client_project_snapshots
Revises: 0004_browser_video
"""

import sqlalchemy as sa
from alembic import op

revision = "0005_client_project_snapshots"
down_revision = "0004_browser_video"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("project", recreate="always") as batch_op:
        batch_op.add_column(
            sa.Column("type", sa.String(length=16), nullable=False, server_default="board")
        )
        batch_op.add_column(sa.Column("client_id", sa.String(length=255), nullable=True))
        batch_op.add_column(
            sa.Column(
                "payload",
                sa.JSON(),
                nullable=False,
                server_default=sa.text("'{}'"),
            )
        )
        batch_op.create_unique_constraint(
            "uq_project_owner_type_client_id",
            ["owner_id", "type", "client_id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("project", recreate="always") as batch_op:
        batch_op.drop_constraint("uq_project_owner_type_client_id", type_="unique")
        batch_op.drop_column("payload")
        batch_op.drop_column("client_id")
        batch_op.drop_column("type")
