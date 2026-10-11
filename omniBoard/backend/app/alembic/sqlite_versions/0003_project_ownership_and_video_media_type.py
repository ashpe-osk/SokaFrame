"""Add optional project ownership and detected video media type.

Revision ID: 0003_project_ownership
Revises: 0002_football_domain
"""

import sqlalchemy as sa
from alembic import op

revision = "0003_project_ownership"
down_revision = "0002_football_domain"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("project", recreate="always") as batch_op:
        batch_op.add_column(sa.Column("owner_id", sa.Uuid(as_uuid=True), nullable=True))
        batch_op.create_foreign_key(
            "fk_project_owner_id_user",
            "user",
            ["owner_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index("ix_project_owner_id", ["owner_id"], unique=False)

    with op.batch_alter_table("video", recreate="always") as batch_op:
        batch_op.add_column(sa.Column("media_type", sa.String(length=127), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("video", recreate="always") as batch_op:
        batch_op.drop_column("media_type")

    with op.batch_alter_table("project", recreate="always") as batch_op:
        batch_op.drop_index("ix_project_owner_id")
        batch_op.drop_constraint("fk_project_owner_id_user", type_="foreignkey")
        batch_op.drop_column("owner_id")
