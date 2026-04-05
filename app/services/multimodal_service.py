from __future__ import annotations

import base64
from typing import Sequence

from fastapi import UploadFile

from app.config import Settings
from app.core.exceptions import BadRequestError
from app.core.models import MediaAttachment, ModelCapability
from app.utils.file_utils import (
    ensure_allowed_extension,
    guess_content_type,
    sanitize_filename,
    validate_file_size,
)


class MultimodalService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def get_model_capability(self, model_name: str) -> ModelCapability:
        normalized = model_name.strip().lower()
        base_name = normalized.split(":", maxsplit=1)[0]

        for configured_name, capability in self.settings.model_capabilities.items():
            configured = configured_name.strip().lower()
            if normalized == configured or base_name == configured or normalized.startswith(f"{configured}:"):
                try:
                    return ModelCapability(capability)
                except ValueError:
                    break

        return ModelCapability.text_only

    def validate_model_supports_images(self, model_name: str) -> None:
        capability = self.get_model_capability(model_name)
        if capability != ModelCapability.text_image:
            raise BadRequestError(
                f"Model '{model_name}' is configured as text-only and cannot accept image input."
            )

    async def prepare_upload_images(
        self,
        *,
        model_name: str,
        files: Sequence[UploadFile],
    ) -> list[MediaAttachment]:
        if not files:
            return []

        self.validate_model_supports_images(model_name)
        attachments: list[MediaAttachment] = []
        for file in files:
            safe_name = sanitize_filename(file.filename or "image")
            ensure_allowed_extension(safe_name, self.settings.supported_image_extensions)
            payload = await file.read()
            validate_file_size(
                len(payload),
                self.settings.max_image_size_bytes,
                kind=f"Image '{safe_name}'",
            )
            attachments.append(
                MediaAttachment(
                    filename=safe_name,
                    media_type=file.content_type or guess_content_type(safe_name),
                    base64_data=base64.b64encode(payload).decode("utf-8"),
                )
            )
        return attachments

    def prepare_base64_images(self, *, model_name: str, images_base64: Sequence[str]) -> list[MediaAttachment]:
        if not images_base64:
            return []

        self.validate_model_supports_images(model_name)
        attachments: list[MediaAttachment] = []
        for index, encoded in enumerate(images_base64, start=1):
            if not encoded.strip():
                raise BadRequestError("Image payload cannot be empty.")
            attachments.append(
                MediaAttachment(
                    filename=f"image_{index}.png",
                    media_type="image/png",
                    base64_data=encoded.strip(),
                )
            )
        return attachments
