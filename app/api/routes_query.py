from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.dependencies import get_container
from app.core.container import ServiceContainer
from app.core.schemas import QueryJsonRequest, QueryRequest, QueryResponse


router = APIRouter(tags=["query"])


def parse_query_form(
    question: Annotated[str, Form(...)],
    role_name: Annotated[str | None, Form()] = None,
    retrieval_mode: Annotated[str, Form()] = "local_only",
    use_citations: Annotated[bool, Form()] = True,
    top_k: Annotated[int | None, Form()] = None,
    session_id: Annotated[str | None, Form()] = None,
    chat_model: Annotated[str | None, Form()] = None,
    use_thinking: Annotated[bool | None, Form()] = None,
    debug: Annotated[bool, Form()] = False,
) -> QueryRequest:
    return QueryRequest(
        question=question,
        role_name=role_name,
        retrieval_mode=retrieval_mode,
        use_citations=use_citations,
        top_k=top_k,
        session_id=session_id,
        chat_model=chat_model,
        use_thinking=use_thinking,
        debug=debug,
    )


@router.post("/query", response_model=QueryResponse)
async def query(
    payload: Annotated[QueryRequest, Depends(parse_query_form)],
    images: list[UploadFile] | None = File(default=None),
    container: ServiceContainer = Depends(get_container),
) -> QueryResponse:
    result = await container.rag_service.answer_query(
        request=payload,
        image_files=images or [],
    )
    return QueryResponse.model_validate(result.model_dump())


@router.post("/query/json", response_model=QueryResponse)
async def query_json(
    payload: QueryJsonRequest,
    container: ServiceContainer = Depends(get_container),
) -> QueryResponse:
    result = await container.rag_service.answer_query(
        request=QueryRequest.model_validate(payload.model_dump(exclude={"image_base64"})),
        image_base64=payload.image_base64,
    )
    return QueryResponse.model_validate(result.model_dump())
