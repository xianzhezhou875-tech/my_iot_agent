import logging
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# 必须在 import LangChain / main_graph 之前加载，LangSmith 才读得到环境变量
load_dotenv()

from logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

from main_graph import app as agent_graph

# LLM / 网络瞬断：仅对这些类型做 tenacity 重试（最多 3 次）
_retryable_types: tuple[type[BaseException], ...] = (
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.TimeoutException,
    TimeoutError,
)
try:
    from openai import APIConnectionError, APITimeoutError

    _retryable_types = _retryable_types + (APIConnectionError, APITimeoutError)
except ImportError:
    pass


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type(_retryable_types),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _invoke_agent_graph_with_retry(
    input_state: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_config = config or {"recursion_limit": 30}
    return agent_graph.invoke(input_state, config=run_config)


app = FastAPI(title="IoT Agent API")

from database_worker import init_db
from rag_worker import init_rag

init_db()
init_rag()
logger.info("FastAPI 应用已加载，数据库与 RAG 初始化完成")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": "请求参数不合法，请检查输入",
            "detail": exc.errors(),
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    detail = exc.detail
    message = detail if isinstance(detail, str) else "请求无法完成"
    return JSONResponse(status_code=exc.status_code, content={"error": message})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("未处理异常 path=%s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "Agent 服务暂时不可用，请稍后重试"},
    )


class ChatRequest(BaseModel):
    user_input: str
    session_id: str = "default"


@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    logger.debug(
        "收到 /chat 请求 session_id=%s len(user_input)=%s",
        request.session_id,
        len(request.user_input or ""),
    )
    input_state = {
        "messages": [HumanMessage(content=request.user_input)],
        "rewrite_count": 0,
    }
    try:
        result = _invoke_agent_graph_with_retry(input_state)
    except _retryable_types as e:  # type: ignore[misc]
        logger.warning("Agent 在重试后仍失败（多为超时/连接）: %s", e)
        return JSONResponse(
            status_code=504,
            content={"error": "Agent 处理超时，请重试"},
        )

    final_reply = result["messages"][-1].content
    text = final_reply if isinstance(final_reply, str) else str(final_reply)
    logger.debug("Agent 回复长度: %s", len(text))
    return {"reply": text}
