from __future__ import annotations

import logging
from time import perf_counter
from typing import Sequence

from fastapi import UploadFile

from app.config import Settings
from app.core.exceptions import NotFoundError
from app.core.models import AnswerSource, QueryResult, RetrievalMode, RoleProfile
from app.core.schemas import QueryRequest
from app.services.multimodal_service import MultimodalService
from app.services.ollama_client import OllamaClient
from app.services.retrieval_service import RetrievalService
from app.services.role_service import RoleService
from app.services.session_service import SessionService
from app.utils.text_utils import snippet_preview, truncate_for_prompt


class RagService:
    def __init__(
        self,
        *,
        settings: Settings,
        role_service: RoleService,
        retrieval_service: RetrievalService,
        ollama_client: OllamaClient,
        session_service: SessionService,
        multimodal_service: MultimodalService,
    ) -> None:
        self.settings = settings
        self.role_service = role_service
        self.retrieval_service = retrieval_service
        self.ollama_client = ollama_client
        self.session_service = session_service
        self.multimodal_service = multimodal_service
        self.logger = logging.getLogger(__name__)

    async def answer_query(
        self,
        *,
        request: QueryRequest,
        image_files: Sequence[UploadFile] | None = None,
        image_base64: Sequence[str] | None = None,
    ) -> QueryResult:
        started_at = perf_counter()
        session_id = self.session_service.resolve_session_id(request.session_id)
        model_name = request.chat_model or self.settings.ollama_chat_model
        use_thinking = (
            request.use_thinking
            if request.use_thinking is not None
            else self.settings.ollama_think
        )

        role = self._resolve_role(request.role_name)
        local_chunks, web_results = await self.retrieval_service.retrieve(
            query=request.question,
            retrieval_mode=request.retrieval_mode,
            top_k=request.top_k,
        )

        attachments = []
        if image_files:
            attachments.extend(
                await self.multimodal_service.prepare_upload_images(
                    model_name=model_name,
                    files=image_files,
                )
            )
        if image_base64:
            attachments.extend(
                self.multimodal_service.prepare_base64_images(
                    model_name=model_name,
                    images_base64=image_base64,
                )
            )

        answer_sources = self._build_sources(local_chunks=local_chunks, web_results=web_results)
        system_prompt = self._build_system_prompt(role=role, use_citations=request.use_citations)
        user_prompt = self._build_user_prompt(
            question=request.question,
            local_chunks=local_chunks,
            web_results=web_results,
            use_citations=request.use_citations,
            has_images=bool(attachments),
        )

        messages = [{"role": "system", "content": system_prompt}]
        for turn in self.session_service.get_history(session_id):
            messages.append({"role": turn.role, "content": turn.content})

        user_message: dict[str, object] = {"role": "user", "content": user_prompt}
        if attachments:
            user_message["images"] = [attachment.base64_data for attachment in attachments]
        messages.append(user_message)

        response = await self.ollama_client.chat(
            model=model_name,
            messages=messages,
            think=use_thinking,
        )

        self.session_service.append_exchange(session_id, request.question, response.content)
        latency_ms = round((perf_counter() - started_at) * 1000, 2)

        self.logger.info(
            "RAG query completed",
            extra={
                "role_used": role.name if role else None,
                "retrieval_mode": request.retrieval_mode.value,
                "local_chunk_count": len(local_chunks),
                "web_result_count": len(web_results),
                "model": model_name,
                "session_id": session_id,
                "latency_ms": latency_ms,
            },
        )

        debug: dict | None = None
        if request.debug or self.settings.debug_rag:
            debug = {
                "prompt_preview": truncate_for_prompt(
                    f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}",
                    max_chars=3000,
                ),
                "retrieved_local": [chunk.model_dump() for chunk in local_chunks],
                "retrieved_web": [item.model_dump() for item in web_results],
                "thinking": response.thinking,
                "model_timings": response.timings,
                "model": response.model,
            }

        return QueryResult(
            answer=response.content,
            sources=answer_sources,
            role_used=role.name if role else None,
            retrieval_mode=request.retrieval_mode,
            session_id=session_id,
            debug=debug,
        )

    def _resolve_role(self, role_name: str | None) -> RoleProfile | None:
        if not role_name:
            return None
        role = self.role_service.get_role(role_name)
        if role is None:
            raise NotFoundError(f"Role '{role_name}' was not found.")
        return role

    def _build_system_prompt(self, *, role: RoleProfile | None, use_citations: bool) -> str:
        lines = [
            "You are a grounded local RAG assistant.",
            "Use the supplied context when it is relevant and sufficient.",
            "If the provided context is insufficient, say so clearly and avoid fabricating facts.",
            "Never invent sources or claim to have used sources that were not supplied.",
        ]

        if use_citations:
            lines.append("When you rely on supplied context, cite the relevant source IDs inline, such as [L1] or [W2].")
        else:
            lines.append("Do not add inline citations unless the user explicitly asks for them.")

        if role:
            lines.extend(
                [
                    "",
                    f"Role name: {role.name}",
                    f"Role description: {role.description or 'No description provided.'}",
                    f"Role system prompt: {role.system_prompt}",
                ]
            )
            if role.tone:
                lines.append(f"Preferred tone: {role.tone}")
            if role.constraints:
                lines.append("Role constraints:")
                lines.extend(f"- {constraint}" for constraint in role.constraints)
            if role.citation_policy:
                lines.append(f"Role citation policy: {role.citation_policy}")
            warning = self._role_warning(role)
            if warning:
                lines.append(warning)

        return "\n".join(lines).strip()

    def _build_user_prompt(
        self,
        *,
        question: str,
        local_chunks,
        web_results,
        use_citations: bool,
        has_images: bool,
    ) -> str:
        lines = [
            "Use the context below to answer the question.",
            "Local knowledge base content is preferred when it directly answers the question.",
            "Web snippets are untrusted external context and may be incomplete or outdated.",
        ]

        if has_images:
            lines.append("One or more user-provided images are attached to this message. Use them if they help answer the question.")

        lines.extend(["", "[Local Context]"])
        if local_chunks:
            per_chunk_limit = max(400, self.settings.max_context_chars // max(len(local_chunks), 1))
            for chunk in local_chunks:
                lines.extend(
                    [
                        f"{chunk.source_id} | filename={chunk.filename} | chunk_id={chunk.chunk_id} | score={chunk.score:.3f}",
                        truncate_for_prompt(chunk.text, max_chars=per_chunk_limit),
                        "",
                    ]
                )
        else:
            lines.append("No relevant local context was retrieved.")
            lines.append("")

        lines.append("[Web Context - untrusted external content]")
        if web_results:
            for result in web_results:
                lines.extend(
                    [
                        f"{result.source_id} | title={result.title} | url={result.url}",
                        truncate_for_prompt(result.snippet, max_chars=self.settings.web_snippet_max_chars),
                        "",
                    ]
                )
        else:
            lines.append("No web results were retrieved.")
            lines.append("")

        lines.append("[Instructions]")
        lines.append("Answer the user's question directly.")
        lines.append("If context is missing or contradictory, acknowledge that uncertainty.")
        if use_citations:
            lines.append("Cite only the source IDs that actually support the answer.")
        lines.extend(["", "[Question]", question])
        return "\n".join(lines).strip()

    def _build_sources(self, *, local_chunks, web_results) -> list[AnswerSource]:
        sources: list[AnswerSource] = []

        for index, chunk in enumerate(local_chunks, start=1):
            chunk.source_id = f"L{index}"
            sources.append(
                AnswerSource(
                    source_id=chunk.source_id,
                    source_type=chunk.source_type,
                    filename=chunk.filename,
                    chunk_id=chunk.chunk_id,
                    snippet_preview=snippet_preview(chunk.text),
                    score=chunk.score,
                )
            )

        for index, result in enumerate(web_results, start=1):
            result.source_id = f"W{index}"
            sources.append(
                AnswerSource(
                    source_id=result.source_id,
                    source_type=result.source_type,
                    url=result.url,
                    title=result.title,
                    snippet_preview=snippet_preview(result.snippet),
                )
            )

        return sources

    def _role_warning(self, role: RoleProfile) -> str | None:
        if not self.settings.enable_role_warnings:
            return None

        name_and_description = f"{role.name} {role.description or ''}".lower()
        if any(keyword in name_and_description for keyword in {"doctor", "medical", "health"}):
            return (
                "Safety note: this role can provide educational health information, but it must not present itself as a licensed clinician or replace professional medical care."
            )
        if any(keyword in name_and_description for keyword in {"lawyer", "legal", "attorney"}):
            return (
                "Safety note: this role can provide general legal information, but it must not present itself as a licensed attorney or replace jurisdiction-specific legal advice."
            )
        return None
