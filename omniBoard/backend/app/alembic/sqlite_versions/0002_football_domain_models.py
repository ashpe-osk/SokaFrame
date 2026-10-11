"""Add local football analysis domain metadata.

Revision ID: 0002_football_domain
Revises: 0001_sqlite_template
"""

import sqlalchemy as sa
from alembic import op

revision = "0002_football_domain"
down_revision = "0001_sqlite_template"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "project",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("description", sa.String(length=2000), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "length(trim(name)) > 0", name="ck_project_name_nonempty"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "video",
        sa.Column("project_id", sa.Uuid(as_uuid=True), nullable=True),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("storage_path", sa.String(length=1024), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("fps", sa.Float(), nullable=True),
        sa.Column("frame_width", sa.Integer(), nullable=True),
        sa.Column("frame_height", sa.Integer(), nullable=True),
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "file_size_bytes >= 0", name="ck_video_file_size_nonnegative"
        ),
        sa.CheckConstraint(
            "sha256 IS NULL OR length(sha256) = 64",
            name="ck_video_sha256_length",
        ),
        sa.CheckConstraint(
            "duration_seconds IS NULL OR duration_seconds >= 0",
            name="ck_video_duration_nonnegative",
        ),
        sa.CheckConstraint(
            "fps IS NULL OR fps > 0", name="ck_video_fps_positive"
        ),
        sa.CheckConstraint(
            "frame_width IS NULL OR frame_width > 0",
            name="ck_video_width_positive",
        ),
        sa.CheckConstraint(
            "frame_height IS NULL OR frame_height > 0",
            name="ck_video_height_positive",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"], ["project.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("storage_path"),
    )
    op.create_index("ix_video_project_id", "video", ["project_id"], unique=False)
    op.create_table(
        "analysis",
        sa.Column("video_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("options", sa.JSON(), nullable=False),
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("frames_processed", sa.Integer(), nullable=False),
        sa.Column("total_frames", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("error_message", sa.String(length=2000), nullable=True),
        sa.Column("output_dir", sa.String(length=1024), nullable=True),
        sa.Column("summary", sa.JSON(), nullable=True),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'cancel_requested', 'succeeded', "
            "'failed', 'cancelled', 'interrupted')",
            name="ck_analysis_status_valid",
        ),
        sa.CheckConstraint(
            "frames_processed >= 0",
            name="ck_analysis_frames_processed_nonnegative",
        ),
        sa.CheckConstraint(
            "total_frames IS NULL OR total_frames >= 0",
            name="ck_analysis_total_frames_nonnegative",
        ),
        sa.CheckConstraint(
            "total_frames IS NULL OR frames_processed <= total_frames",
            name="ck_analysis_progress_within_total",
        ),
        sa.ForeignKeyConstraint(["video_id"], ["video.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_analysis_video_id", "analysis", ["video_id"], unique=False)
    op.create_table(
        "artifact",
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("storage_path", sa.String(length=1024), nullable=False),
        sa.Column("media_type", sa.String(length=127), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("analysis_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "kind IN ('tracked_video', 'observations_csv', 'summary', 'log', 'other')",
            name="ck_artifact_kind_valid",
        ),
        sa.CheckConstraint(
            "sha256 IS NULL OR length(sha256) = 64",
            name="ck_artifact_sha256_length",
        ),
        sa.CheckConstraint(
            "file_size_bytes >= 0", name="ck_artifact_file_size_nonnegative"
        ),
        sa.ForeignKeyConstraint(
            ["analysis_id"], ["analysis.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("storage_path"),
    )
    op.create_index(
        "ix_artifact_analysis_id", "artifact", ["analysis_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_artifact_analysis_id", table_name="artifact")
    op.drop_table("artifact")
    op.drop_index("ix_analysis_video_id", table_name="analysis")
    op.drop_table("analysis")
    op.drop_index("ix_video_project_id", table_name="video")
    op.drop_table("video")
    op.drop_table("project")
