import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from sqlmodel import func, select

from app.api.deps import CurrentUser, SessionDep
from app.models import APIError, Project, ProjectCreate, ProjectPublic, ProjectsPublic

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post(
    "/",
    response_model=ProjectPublic,
    status_code=201,
    responses={401: {"model": APIError, "description": "Authentication required"}},
)
def create_project(
    session: SessionDep, current_user: CurrentUser, project_in: ProjectCreate
) -> Project:
    project = Project.model_validate(
        project_in, update={"owner_id": current_user.id}
    )
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


@router.get(
    "/",
    response_model=ProjectsPublic,
    responses={401: {"model": APIError, "description": "Authentication required"}},
)
def read_projects(
    session: SessionDep,
    current_user: CurrentUser,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
) -> ProjectsPublic:
    base_statement = select(Project).where(Project.owner_id == current_user.id)
    count = session.exec(
        select(func.count()).select_from(base_statement.subquery())
    ).one()
    projects = session.exec(
        base_statement.order_by(Project.created_at.desc(), Project.id.desc())
        .offset(skip)
        .limit(limit)
    ).all()
    return ProjectsPublic(
        data=[ProjectPublic.model_validate(project) for project in projects],
        count=count,
    )


@router.get(
    "/{project_id}",
    response_model=ProjectPublic,
    responses={
        401: {"model": APIError, "description": "Authentication required"},
        403: {"model": APIError, "description": "Project belongs to another user"},
        404: {"model": APIError, "description": "Project not found"},
    },
)
def read_project(
    project_id: uuid.UUID, session: SessionDep, current_user: CurrentUser
) -> Project:
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Project belongs to another user")
    return project
