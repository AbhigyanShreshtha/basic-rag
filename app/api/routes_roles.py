from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_container
from app.core.container import ServiceContainer
from app.core.schemas import RoleSummary, RolesListResponse, RolesReloadResponse


router = APIRouter(tags=["roles"])


def _summaries(container: ServiceContainer) -> list[RoleSummary]:
    return [RoleSummary.model_validate(role.model_dump()) for role in container.role_service.list_roles()]


@router.get("/roles", response_model=RolesListResponse)
def list_roles(container: ServiceContainer = Depends(get_container)) -> RolesListResponse:
    return RolesListResponse(roles=_summaries(container))


@router.post("/roles/reload", response_model=RolesReloadResponse)
def reload_roles(container: ServiceContainer = Depends(get_container)) -> RolesReloadResponse:
    reloaded = container.role_service.load_roles()
    return RolesReloadResponse(
        roles=[RoleSummary.model_validate(role.model_dump()) for role in reloaded],
        reloaded_count=len(reloaded),
    )
