"""Allow downloadable football analytics artifacts.

Revision ID: 24c7a5d18f02
Revises: fe56fa70289e
"""

from alembic import op

revision = "24c7a5d18f02"
down_revision = "fe56fa70289e"
branch_labels = None
depends_on = None

_ARTIFACT_KINDS = (
    "'tracked_video', 'browser_video', 'observations_csv', 'summary', "
    "'match_summary', 'passes_csv', 'passing_network_csv', "
    "'team_statistics_csv', 'player_statistics_csv', "
    "'possession_sequences_csv', 'log', 'other'"
)


def upgrade() -> None:
    with op.batch_alter_table("artifact", recreate="always") as batch_op:
        batch_op.drop_constraint("ck_artifact_kind_valid", type_="check")
        batch_op.create_check_constraint(
            "ck_artifact_kind_valid", f"kind IN ({_ARTIFACT_KINDS})"
        )


def downgrade() -> None:
    raise RuntimeError(
        "The analytics artifact migration is forward-only to preserve "
        "downloadable analysis outputs."
    )