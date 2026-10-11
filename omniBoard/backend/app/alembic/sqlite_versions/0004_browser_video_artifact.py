"""Allow an application-generated browser-compatible video artifact.

Revision ID: 0004_browser_video
Revises: 0003_project_ownership
"""

from alembic import op

revision = "0004_browser_video"
down_revision = "0003_project_ownership"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("artifact", recreate="always") as batch_op:
        batch_op.drop_constraint("ck_artifact_kind_valid", type_="check")
        batch_op.create_check_constraint(
            "ck_artifact_kind_valid",
            "kind IN ('tracked_video', 'browser_video', 'observations_csv', "
            "'summary', 'log', 'other')",
        )


def downgrade() -> None:
    op.execute("DELETE FROM artifact WHERE kind = 'browser_video'")
    with op.batch_alter_table("artifact", recreate="always") as batch_op:
        batch_op.drop_constraint("ck_artifact_kind_valid", type_="check")
        batch_op.create_check_constraint(
            "ck_artifact_kind_valid",
            "kind IN ('tracked_video', 'observations_csv', 'summary', 'log', 'other')",
        )
