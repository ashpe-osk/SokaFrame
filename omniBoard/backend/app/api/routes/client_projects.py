from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Path, status
from pydantic import JsonValue
from sqlmodel import Session, select

from app.api.deps import CurrentUser, SessionDep
from app.models import Project

ProjectType = Literal["board", "media"]
ProjectKind = Literal["tactical", "cv-analysis"]
ProjectPayload = dict[str, JsonValue]
ClientProjectResponse = dict[str, ProjectPayload | ProjectKind]

router = APIRouter(prefix="/projects", tags=["client-projects"])


def _owned_project(
    session: Session, user_id: UUID, project_type: ProjectType, client_id: str
) -> Project | None:
    return session.exec(
        select(Project).where(
            Project.owner_id == user_id,
            Project.type == project_type,
            Project.client_id == client_id,
        )
    ).first()


def _project_belongs_to_another_user(
    session: Session, user_id: UUID, project_type: ProjectType, client_id: str
) -> bool:
    if client_id == "default":
        return False
    other_project_id = session.exec(
        select(Project.id).where(
            Project.type == project_type,
            Project.client_id == client_id,
            Project.owner_id != user_id,
        )
    ).first()
    return other_project_id is not None


def _project_summary(project: Project) -> dict[str, str]:
    summary = {
        "id": project.client_id or str(project.id),
        "type": project.type,
        "kind": project.kind,
        "title": project.name,
        "createdAt": project.created_at.isoformat(),
        "updatedAt": project.updated_at.isoformat(),
    }
    workspace = project.payload.get("workspace")
    if isinstance(workspace, str):
        summary["workspace"] = workspace
    return summary


def _validate_payload(
    payload: ProjectPayload, project_type: ProjectType, client_id: str
) -> tuple[str, ProjectKind]:
    payload_id = payload.get("id")
    payload_type = payload.get("type")
    title = payload.get("title")
    if payload_id is not None and payload_id != client_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Payload project ID must match the URL",
        )
    if payload_type is not None and payload_type != project_type:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Payload project type must match the URL",
        )
    kind = payload.get("kind", "tactical")
    if not isinstance(kind, str) or kind not in ("tactical", "cv-analysis"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Project kind must be 'tactical' or 'cv-analysis'",
        )
    if not isinstance(title, str) or not title.strip() or len(title.strip()) > 120:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Payload title must contain 1 to 120 characters",
        )
    return title.strip(), kind


@router.get("/{project_type}")
def list_client_projects(
    project_type: ProjectType, session: SessionDep, current_user: CurrentUser
) -> dict[str, list[dict[str, str]]]:
    projects = session.exec(
        select(Project)
        .where(
            Project.owner_id == current_user.id,
            Project.type == project_type,
            Project.client_id.is_not(None),
        )
        .order_by(Project.updated_at.desc(), Project.id.desc())
    ).all()
    return {"projects": [_project_summary(project) for project in projects]}


@router.get("/{project_type}/{client_id}")
def read_client_project(
    project_type: ProjectType,
    client_id: Annotated[str, Path(min_length=1, max_length=255)],
    session: SessionDep,
    current_user: CurrentUser,
) -> ClientProjectResponse:
    project = _owned_project(session, current_user.id, project_type, client_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"payload": project.payload, "kind": project.kind}


@router.put("/{project_type}/{client_id}")
def upsert_client_project(
    project_type: ProjectType,
    client_id: Annotated[str, Path(min_length=1, max_length=255)],
    project_in: dict[str, ProjectPayload],
    session: SessionDep,
    current_user: CurrentUser,
) -> ClientProjectResponse:
    payload = project_in.get("payload")
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="A project payload object is required",
        )
    title, requested_kind = _validate_payload(payload, project_type, client_id)
    project = _owned_project(session, current_user.id, project_type, client_id)
    if project is None:
        if _project_belongs_to_another_user(
            session, current_user.id, project_type, client_id
        ):
            raise HTTPException(status_code=404, detail="Project not found")
        project = Project(
            owner_id=current_user.id,
            type=project_type,
            client_id=client_id,
            name=title,
            payload=payload,
            kind=requested_kind,
        )
        session.add(project)
    else:
        if "kind" in payload and requested_kind != project.kind:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Project kind cannot be changed after creation",
            )
        project.name = title
        project.payload = payload
    session.commit()
    session.refresh(project)
    return {"payload": project.payload, "kind": project.kind}


@router.delete("/{project_type}/{client_id}")
def delete_client_project(
    project_type: ProjectType,
    client_id: Annotated[str, Path(min_length=1, max_length=255)],
    session: SessionDep,
    current_user: CurrentUser,
) -> dict[str, bool]:
    project = _owned_project(session, current_user.id, project_type, client_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    session.delete(project)
    session.commit()
    return {"ok": True}
