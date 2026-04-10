"""Custom application exception hierarchy."""

from __future__ import annotations


class AppException(Exception):
    """Base class for all application-level errors.

    All subclasses are caught by the global exception handler and converted
    to a standardised JSON error response.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "APP_ERROR",
        status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code


class NotFoundError(AppException):
    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(message, error_code="NOT_FOUND", status_code=404)


class ConflictError(AppException):
    def __init__(self, message: str = "Resource already exists") -> None:
        super().__init__(message, error_code="CONFLICT", status_code=409)


class AuthenticationError(AppException):
    def __init__(self, message: str = "Authentication failed") -> None:
        super().__init__(message, error_code="AUTHENTICATION_ERROR", status_code=401)


class AuthorizationError(AppException):
    def __init__(self, message: str = "Insufficient permissions") -> None:
        super().__init__(message, error_code="AUTHORIZATION_ERROR", status_code=403)


class ValidationError(AppException):
    def __init__(self, message: str = "Validation error") -> None:
        super().__init__(message, error_code="VALIDATION_ERROR", status_code=422)
