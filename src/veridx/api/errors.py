"""统一错误响应与错误码（设计文档 §8.3：业务错误不得返回 200）。

两条 P0.6 纪律：

* **响应里出现的每个字段都必须有人在写它。** `run_id` 只在控制流跑起来之后才有；
  500 通常发生在控制流之前，所以 500 的定位靠 `request_id`（中间件生成），而不是 `run_id`。
* **500 不把异常串返给客户端**（§11.4 脱敏）。固定文案 + `request_id` 给用户；
  完整堆栈只进日志。
"""

from __future__ import annotations

import logging
from enum import StrEnum

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("veridx.api")

INTERNAL_ERROR_MESSAGE = "服务器内部错误。请把 request_id 一并反馈，便于定位。"


class ErrorCode(StrEnum):
    INVALID_INPUT = "invalid_input"
    NOT_FOUND = "not_found"
    DUPLICATE_ATTEMPT = "duplicate_attempt"
    IDEMPOTENCY_KEY_CONFLICT = "idempotency_key_conflict"
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
    ErrorCode.IDEMPOTENCY_KEY_CONFLICT: 409,
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
    request_id: str | None = None
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

    def response(self, request: Request | None = None) -> JSONResponse:
        body = ErrorResponse(
            code=self.code,
            message=self.message,
            request_id=_request_id(request),
            run_id=self.run_id,
        )
        return JSONResponse(status_code=self.status_code, content=body.model_dump(mode="json"))


def _request_id(request: Request | None) -> str | None:
    return getattr(request.state, "request_id", None) if request is not None else None


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return exc.response(request)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        body = ErrorResponse(
            code=ErrorCode.INVALID_INPUT,
            message=_format_validation(exc),
            request_id=_request_id(request),
        )
        return JSONResponse(
            status_code=HTTP_STATUS[ErrorCode.INVALID_INPUT], content=body.model_dump(mode="json")
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        request_id = _request_id(request)
        logger.exception(
            "unhandled error on %s %s [request_id=%s]", request.method, request.url.path, request_id
        )
        body = ErrorResponse(
            code=ErrorCode.INTERNAL_ERROR, message=INTERNAL_ERROR_MESSAGE, request_id=request_id
        )
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
