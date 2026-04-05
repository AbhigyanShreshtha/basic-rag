from __future__ import annotations

from typing import Any


class AppError(Exception):
    def __init__(
        self,
        message: str,
        *,
        error_code: str = "app_error",
        status_code: int = 400,
        details: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details


class BadRequestError(AppError):
    def __init__(self, message: str, *, details: Any | None = None) -> None:
        super().__init__(
            message,
            error_code="bad_request",
            status_code=400,
            details=details,
        )


class NotFoundError(AppError):
    def __init__(self, message: str, *, details: Any | None = None) -> None:
        super().__init__(
            message,
            error_code="not_found",
            status_code=404,
            details=details,
        )


class UnsupportedMediaTypeError(AppError):
    def __init__(self, message: str, *, details: Any | None = None) -> None:
        super().__init__(
            message,
            error_code="unsupported_media_type",
            status_code=415,
            details=details,
        )


class ExternalServiceError(AppError):
    def __init__(self, message: str, *, details: Any | None = None) -> None:
        super().__init__(
            message,
            error_code="external_service_error",
            status_code=502,
            details=details,
        )


class FeatureUnavailableError(AppError):
    def __init__(self, message: str, *, details: Any | None = None) -> None:
        super().__init__(
            message,
            error_code="feature_unavailable",
            status_code=503,
            details=details,
        )


class ConfigurationError(AppError):
    def __init__(self, message: str, *, details: Any | None = None) -> None:
        super().__init__(
            message,
            error_code="configuration_error",
            status_code=500,
            details=details,
        )
