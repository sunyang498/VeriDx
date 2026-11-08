"""统一错误响应与错误码（设计文档 §8.3：业务错误不得返回 200）。"""

from __future__ import annotations

from enum import StrEnum

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ErrorCode(StrEnum):
    INVALID_INPUT = "invalid_input"
    NOT_FOUND = "not_found"
    DUPLICATE_ATTEMPT = "duplicate_attempt"
    UNPARSABLE_STEM = "unparsable_stem"
    RATE_LIMITED = "rate_limited"
    QUEUE_FULL = "queue_full"
    PROVIDER_ERROR = "provider_error"
    INTERNAL_ERROR = "internal_error"
    # P0 桩专用：P2.1 / P4.1 实现对应端点后从枚举中移除
    NOT_IMPLEMENTED = "not_implemented"


HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.INVALID_INPUT: 400,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.DUPLICATE_ATTEMPT: 409,
    ErrorCode.UNPARSABLE_STEM: 422,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.QUEUE_FULL: 429,
    ErrorCode.PROVIDER_ERROR: 502,
    ErrorCode.INTERNAL_ERROR: 500,
    ErrorCode.NOT_IMPLEMENTED: 501,
}


class ErrorResponse(BaseModel):
    code: ErrorCode
    message: str
    run_id: str | None = None


class ApiError(Exception):
    def __init__(self, code: ErrorCode, message: str, *, run_id: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.run_id = run_id

    @property
    def status_code(self) -> int:
        return HTTP_STATUS[self.code]

    def response(self) -> JSONResponse:
        body = ErrorResponse(code=self.code, message=self.message, run_id=self.run_id)
        return JSONResponse(status_code=self.status_code, content=body.model_dump(mode="json"))


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return exc.response()

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        body = ErrorResponse(code=ErrorCode.INVALID_INPUT, message=_format_validation(exc))
        return JSONResponse(
            status_code=HTTP_STATUS[ErrorCode.INVALID_INPUT], content=body.model_dump(mode="json")
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        # 未捕获异常也必须走统一结构（带 run_id 便于回放，见 §12.3）
        body = ErrorResponse(code=ErrorCode.INTERNAL_ERROR, message=f"{type(exc).__name__}: {exc}")
        return JSONResponse(
            status_code=HTTP_STATUS[ErrorCode.INTERNAL_ERROR], content=body.model_dump(mode="json")
        )


def _format_validation(exc: RequestValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(x) for x in error.get("loc", ()))
        parts.append(f"{loc}: {error.get('msg', 'invalid')}")
    return "; ".join(parts) or "invalid input"


def not_implemented(owner: str) -> ApiError:
    return ApiError(ErrorCode.NOT_IMPLEMENTED, f"P0 桩：该端点将在 {owner} 实现")
