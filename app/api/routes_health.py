from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_container
from app.core.container import ServiceContainer
from app.core.schemas import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(container: ServiceContainer = Depends(get_container)) -> HealthResponse:
    ollama_available = await container.ollama_client.check_health()
    return HealthResponse(
        status="ok" if ollama_available else "degraded",
        app_env=container.settings.app_env,
        ollama_available=ollama_available,
        ollama_base_url=container.settings.ollama_base_url,
    )
