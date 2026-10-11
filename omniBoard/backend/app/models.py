import uuid
from datetime import UTC, datetime
from enum import Enum, StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal
from uuid import UUID

from pydantic import EmailStr, JsonValue, field_validator
from sqlalchemy import JSON, CheckConstraint, DateTime, Text, UniqueConstraint
from sqlalchemy import Enum as SAEnum
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.types import TypeDecorator
from sqlmodel import Field, Relationship, SQLModel


class UTCDateTime(TypeDecorator[datetime]):
    impl = DateTime
    cache_ok = True

    def process_bind_param(
        self, value: datetime | None, _dialect: Dialect
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(
        self, value: datetime | None, _dialect: Dialect
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


def get_datetime_utc() -> datetime:
    return datetime.now(UTC)


class AnalysisStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class ArtifactKind(StrEnum):
    TRACKED_VIDEO = "tracked_video"
    BROWSER_VIDEO = "browser_video"
    OBSERVATIONS_CSV = "observations_csv"
    SUMMARY = "summary"
    MATCH_SUMMARY = "match_summary"
    PASSES_CSV = "passes_csv"
    PASSING_NETWORK_CSV = "passing_network_csv"
    TEAM_STATISTICS_CSV = "team_statistics_csv"
    PLAYER_STATISTICS_CSV = "player_statistics_csv"
    POSSESSION_SEQUENCES_CSV = "possession_sequences_csv"
    LOG = "log"
    OTHER = "other"


def _enum_values(enum_type: type[Enum]) -> list[str]:
    return [member.value for member in enum_type]


def _relative_storage_path(value: str) -> str:
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        not value
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in posix_path.parts
        or ".." in windows_path.parts
    ):
        raise ValueError("Storage paths must be relative and stay within the data root")
    return value


# Shared properties
class UserBase(SQLModel):
    email: EmailStr = Field(unique=True, index=True, max_length=255)
    is_active: bool = True
    is_superuser: bool = False
    full_name: str | None = Field(default=None, max_length=255)


# Properties to receive via API on creation
class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=128)


class UserRegister(SQLModel):
    email: EmailStr = Field(max_length=255)
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = Field(default=None, max_length=255)


# Properties to receive via API on update, all are optional
class UserUpdate(SQLModel):
    email: EmailStr | None = Field(default=None, max_length=255)
    is_active: bool | None = None
    is_superuser: bool | None = None
    full_name: str | None = Field(default=None, max_length=255)
    password: str | None = Field(default=None, min_length=8, max_length=128)


class UserUpdateMe(SQLModel):
    full_name: str | None = Field(default=None, max_length=255)
    email: EmailStr | None = Field(default=None, max_length=255)


class UpdatePassword(SQLModel):
    current_password: str = Field(min_length=8, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


# Database model, database table inferred from class name
class User(UserBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    hashed_password: str
    created_at: datetime | None = Field(
        default_factory=get_datetime_utc,
        sa_type=UTCDateTime(),
    )
    items: list[Item] = Relationship(back_populates="owner", cascade_delete=True)
    projects: list[Project] = Relationship(back_populates="owner")


# Properties to return via API, id is always required
class UserPublic(UserBase):
    id: uuid.UUID
    created_at: datetime | None = None


class UsersPublic(SQLModel):
    data: list[UserPublic]
    count: int


# Shared properties
class ItemBase(SQLModel):
    title: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=255)


# Properties to receive on item creation
class ItemCreate(ItemBase):
    pass


# Properties to receive on item update
class ItemUpdate(SQLModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=255)


# Database model, database table inferred from class name
class Item(ItemBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime | None = Field(
        default_factory=get_datetime_utc,
        sa_type=UTCDateTime(),
    )
    owner_id: uuid.UUID = Field(
        foreign_key="user.id", nullable=False, ondelete="CASCADE"
    )
    owner: User | None = Relationship(back_populates="items")


# Properties to return via API, id is always required
class ItemPublic(ItemBase):
    id: uuid.UUID
    owner_id: uuid.UUID
    created_at: datetime | None = None


class ItemsPublic(SQLModel):
    data: list[ItemPublic]
    count: int


class ProjectBase(SQLModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Project name must not be blank")
        return value


ProjectKind = Literal["tactical", "cv-analysis"]


class Project(ProjectBase, table=True):
    __table_args__ = (
        CheckConstraint("length(trim(name)) > 0", name="ck_project_name_nonempty"),
        UniqueConstraint(
            "owner_id",
            "type",
            "client_id",
            name="uq_project_owner_type_client_id",
        ),
    )

    id: UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    owner_id: UUID | None = Field(
        default=None,
        foreign_key="user.id",
        ondelete="SET NULL",
        index=True,
    )
    type: str = Field(default="board", max_length=16)
    kind: str = Field(default="tactical", max_length=16, sa_type=Text)
    client_id: str | None = Field(default=None, max_length=255)
    payload: dict[str, JsonValue] = Field(default_factory=dict, sa_type=JSON)
    created_at: datetime = Field(default_factory=get_datetime_utc, sa_type=UTCDateTime())
    updated_at: datetime = Field(
        default_factory=get_datetime_utc,
        sa_type=UTCDateTime(),
        sa_column_kwargs={"onupdate": get_datetime_utc},
    )
    owner: User | None = Relationship(back_populates="projects")
    videos: list[Video] = Relationship(back_populates="project")


class ProjectCreate(ProjectBase):
    kind: ProjectKind = "tactical"


class ProjectUpdate(SQLModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Project name must not be blank")
        return value


class ProjectPublic(ProjectBase):
    id: UUID
    kind: ProjectKind
    created_at: datetime
    updated_at: datetime


class ProjectsPublic(SQLModel):
    data: list[ProjectPublic]
    count: int


class VideoBase(SQLModel):
    project_id: UUID | None = None
    original_filename: str = Field(min_length=1, max_length=255)
    media_type: str | None = Field(default=None, max_length=127)
    storage_path: str = Field(min_length=1, max_length=1024)
    file_size_bytes: int = Field(ge=0)
    sha256: str | None = Field(default=None, min_length=64, max_length=64)
    duration_seconds: float | None = Field(default=None, ge=0)
    fps: float | None = Field(default=None, gt=0)
    frame_width: int | None = Field(default=None, gt=0)
    frame_height: int | None = Field(default=None, gt=0)

    @field_validator("storage_path")
    @classmethod
    def validate_storage_path(cls, value: str) -> str:
        return _relative_storage_path(value)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str | None) -> str | None:
        if value is not None and any(char not in "0123456789abcdefABCDEF" for char in value):
            raise ValueError("SHA-256 checksums must be hexadecimal")
        return value


class Video(VideoBase, table=True):
    __table_args__ = (
        CheckConstraint("file_size_bytes >= 0", name="ck_video_file_size_nonnegative"),
        CheckConstraint(
            "sha256 IS NULL OR length(sha256) = 64",
            name="ck_video_sha256_length",
        ),
        CheckConstraint(
            "duration_seconds IS NULL OR duration_seconds >= 0",
            name="ck_video_duration_nonnegative",
        ),
        CheckConstraint("fps IS NULL OR fps > 0", name="ck_video_fps_positive"),
        CheckConstraint(
            "frame_width IS NULL OR frame_width > 0", name="ck_video_width_positive"
        ),
        CheckConstraint(
            "frame_height IS NULL OR frame_height > 0",
            name="ck_video_height_positive",
        ),
    )

    id: UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    project_id: UUID | None = Field(
        default=None, foreign_key="project.id", ondelete="SET NULL", index=True
    )
    storage_path: str = Field(unique=True, max_length=1024)
    created_at: datetime = Field(default_factory=get_datetime_utc, sa_type=UTCDateTime())
    project: Project | None = Relationship(back_populates="videos")
    analyses: list[Analysis] = Relationship(back_populates="video")


class VideoCreate(VideoBase):
    pass


class VideoPublic(SQLModel):
    id: UUID
    project_id: UUID | None
    original_filename: str
    media_type: str | None
    file_size_bytes: int
    sha256: str | None
    duration_seconds: float | None
    fps: float | None
    frame_width: int | None
    frame_height: int | None
    created_at: datetime


class VideosPublic(SQLModel):
    data: list[VideoPublic]
    count: int


class AnalysisBase(SQLModel):
    video_id: UUID
    options: dict[str, JsonValue] = Field(default_factory=dict, sa_type=JSON)


class Analysis(AnalysisBase, table=True):
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'cancel_requested', 'succeeded', "
            "'failed', 'cancelled', 'interrupted')",
            name="ck_analysis_status_valid",
        ),
        CheckConstraint(
            "frames_processed >= 0", name="ck_analysis_frames_processed_nonnegative"
        ),
        CheckConstraint(
            "total_frames IS NULL OR total_frames >= 0",
            name="ck_analysis_total_frames_nonnegative",
        ),
        CheckConstraint(
            "total_frames IS NULL OR frames_processed <= total_frames",
            name="ck_analysis_progress_within_total",
        ),
    )

    id: UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    video_id: UUID = Field(
        foreign_key="video.id", nullable=False, ondelete="RESTRICT", index=True
    )
    status: AnalysisStatus = Field(
        default=AnalysisStatus.QUEUED,
        sa_type=SAEnum(
            AnalysisStatus,
            name="analysis_status",
            native_enum=False,
            create_constraint=False,
            length=32,
            values_callable=_enum_values,
        ),
    )
    frames_processed: int = Field(default=0, ge=0)
    total_frames: int | None = Field(default=None, ge=0)
    created_at: datetime = Field(default_factory=get_datetime_utc, sa_type=UTCDateTime())
    started_at: datetime | None = Field(default=None, sa_type=UTCDateTime())
    finished_at: datetime | None = Field(default=None, sa_type=UTCDateTime())
    error_code: str | None = Field(default=None, max_length=100)
    error_message: str | None = Field(default=None, max_length=2000)
    output_dir: str | None = Field(default=None, max_length=1024)
    summary: dict[str, JsonValue] | None = Field(default=None, sa_type=JSON)
    video: Video = Relationship(back_populates="analyses")
    artifacts: list[Artifact] = Relationship(back_populates="analysis", cascade_delete=True)


class AnalysisCreate(AnalysisBase):
    pass


class AnalysisArtifactPublic(SQLModel):
    id: UUID
    kind: ArtifactKind
    media_type: str
    file_size_bytes: int
    sha256: str | None


class AnalysisPublic(SQLModel):
    id: UUID
    project_id: UUID
    video_id: UUID
    options: dict[str, JsonValue]
    status: AnalysisStatus
    frames_processed: int
    total_frames: int | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error_code: str | None
    result_available: bool
    video_original_filename: str
    video_media_type: str | None
    video_size_bytes: int
    video_sha256: str | None
    summary: dict[str, JsonValue] | None = None
    artifacts: list[AnalysisArtifactPublic] = Field(default_factory=list)


class AnalysesPublic(SQLModel):
    data: list[AnalysisPublic]
    count: int


class ArtifactBase(SQLModel):
    kind: ArtifactKind
    storage_path: str = Field(min_length=1, max_length=1024)
    media_type: str = Field(min_length=1, max_length=127)
    file_size_bytes: int = Field(ge=0)
    sha256: str | None = Field(default=None, min_length=64, max_length=64)

    @field_validator("storage_path")
    @classmethod
    def validate_storage_path(cls, value: str) -> str:
        return _relative_storage_path(value)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str | None) -> str | None:
        if value is not None and any(char not in "0123456789abcdefABCDEF" for char in value):
            raise ValueError("SHA-256 checksums must be hexadecimal")
        return value


class Artifact(ArtifactBase, table=True):
    __table_args__ = (
        CheckConstraint(
            "kind IN ('tracked_video', 'browser_video', 'observations_csv', "
            "'summary', 'match_summary', 'passes_csv', "
            "'passing_network_csv', 'team_statistics_csv', "
            "'player_statistics_csv', 'possession_sequences_csv', "
            "'log', 'other')",
            name="ck_artifact_kind_valid",
        ),
        CheckConstraint(
            "sha256 IS NULL OR length(sha256) = 64",
            name="ck_artifact_sha256_length",
        ),
        CheckConstraint(
            "file_size_bytes >= 0", name="ck_artifact_file_size_nonnegative"
        ),
    )

    id: UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    kind: ArtifactKind = Field(
        sa_type=SAEnum(
            ArtifactKind,
            name="artifact_kind",
            native_enum=False,
            create_constraint=False,
            length=32,
            values_callable=_enum_values,
        )
    )
    analysis_id: UUID = Field(
        foreign_key="analysis.id", nullable=False, ondelete="CASCADE", index=True
    )
    storage_path: str = Field(unique=True, max_length=1024)
    created_at: datetime = Field(default_factory=get_datetime_utc, sa_type=UTCDateTime())
    analysis: Analysis = Relationship(back_populates="artifacts")


class ArtifactCreate(ArtifactBase):
    analysis_id: UUID


class ArtifactPublic(ArtifactBase):
    id: UUID
    analysis_id: UUID
    created_at: datetime


class ArtifactsPublic(SQLModel):
    data: list[ArtifactPublic]
    count: int


# Generic message
class Message(SQLModel):
    message: str


class APIError(SQLModel):
    detail: str


# JSON payload containing access token
class Token(SQLModel):
    access_token: str
    token_type: str = "bearer"


# Contents of JWT token
class TokenPayload(SQLModel):
    sub: str | None = None


class NewPassword(SQLModel):
    token: str
    new_password: str = Field(min_length=8, max_length=128)
