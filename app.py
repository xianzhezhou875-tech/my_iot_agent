"""
FastAPI 接入层 — IoT Agent 对外 HTTP API。

端点：
  POST /chat         — 用户对话入口，内部调度 LangGraph 多智能体编排图。
  POST /api/approve  — HITL 解冻路由，人工确认后恢复中断的 Agent 图执行。

特性：
  - tenacity 指数退避重试（网络瞬断 / API 超时自动恢复）
  - 全局异常拦截器（Pydantic 校验错误、HTTP 异常、未预期异常）
  - HITL（Human-in-the-Loop）中断感知：GraphInterrupt → status: "interrupted"
  - 启动时自动初始化数据库 & RAG 知识库
"""

import logging as _logging_mod
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
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

# 必须在 import LangChain / main_graph 之前加载 .env，LangSmith 才读得到环境变量
load_dotenv()

from logging_config import configure_logging, logger

configure_logging()

from main_graph import app as agent_graph

# ── 可重试异常类型 ──────────────────────────────────────────────
# 仅对网络层瞬断做 tenacity 重试（最多 4 次，指数退避 2s→4s→8s→16s）
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
    before_sleep=before_sleep_log(logger, _logging_mod.WARNING),
    reraise=True,
)
def _invoke_agent_graph_with_retry(
    input_state: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    带指数退避的 Agent 图调用。

    指数退避 (Exponential Backoff /ˌekspəˈnenʃəl ˈbækɒf/)：
    第 1 次重试等 2s → 第 2 次等 4s → 第 3 次等 8s → 第 4 次等 16s（上限 20s）

    这种"越等越久"的策略可以避免在服务恢复瞬间所有请求同时涌入（Thundering Herd 效应）。
    """
    run_config = config or {}
    return agent_graph.invoke(input_state, config=run_config)


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type(_retryable_types),
    before_sleep=before_sleep_log(logger, _logging_mod.WARNING),
    reraise=True,
)
def _resume_agent_graph_with_retry(
    command: Command,
    config: dict[str, Any],
) -> dict[str, Any]:
    """带指数退避的 Agent 图恢复调用。与 _invoke_agent_graph_with_retry 对称。"""
    return agent_graph.invoke(command, config=config)


# ── FastAPI 应用 ───────────────────────────────────────────────

app = FastAPI(title="IoT Agent API")

from database_worker import init_db
from rag_worker import init_rag

init_db()
init_rag()
logger.info("FastAPI 应用已启动，数据库 & RAG 知识库初始化完成")


# ── 全局异常拦截器 ─────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Pydantic 请求校验失败 → 422。"""
    logger.warning("请求参数校验失败 — path=%s errors=%s", request.url.path, exc.errors())
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
    """Starlette HTTP 异常（404 / 405 等）→ 透传状态码。"""
    detail = exc.detail
    message = detail if isinstance(detail, str) else "请求无法完成"
    logger.warning("HTTP 异常 — path=%s status=%s detail=%s", request.url.path, exc.status_code, message)
    return JSONResponse(status_code=exc.status_code, content={"error": message})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """兜底全局异常 → 500，记录完整堆栈。"""
    logger.exception("未处理异常 — path=%s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "Agent 服务暂时不可用，请稍后重试"},
    )


# ── 请求模型 & 路由 ────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_input: str
    session_id: str = "default"


@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    """
    对话入口 — 用户输入 → Agent 图 → AI 回复。

    重试策略：网络/超时类异常自动重试 4 次；
    重试耗尽仍失败 → 504 Gateway Timeout；
    其他异常 → 500 全局兜底。
    """
    logger.info(
        "收到 /chat 请求 — session_id=%s input_len=%s",
        request.session_id,
        len(request.user_input or ""),
    )
    logger.debug("用户输入原文: %s", request.user_input[:200] if request.user_input else "(空)")

    input_state = {
        "messages": [HumanMessage(content=request.user_input)],
        "rewrite_count": 0,
    }
    # ── HITL 会话绑定：session_id → thread_id ──
    # checkpointer 靠此 ID 存取快照；/chat 和 /api/approve 必须用同一个 ID。
    config = {"configurable": {"thread_id": request.session_id}}

    try:
        result = _invoke_agent_graph_with_retry(input_state, config)
    except GraphInterrupt:
        # ── HITL 冻结响应：图已挂起，等待人工确认 ──
        logger.info(
            "[HITL] Agent 挂起 — session_id=%s（前端需调用 /api/approve 解冻）",
            request.session_id,
        )
        return JSONResponse(
            status_code=200,
            content={
                "status": "interrupted",
                "session_id": request.session_id,
                "message": "Agent 需要人工确认，请调用 POST /api/approve 继续。",
            },
        )
    except _retryable_types as e:
        logger.warning("Agent 在 4 次重试后仍失败（网络/超时）: %s", e)
        return JSONResponse(
            status_code=504,
            content={"error": "Agent 处理超时，请重试"},
        )

    final_reply = result["messages"][-1].content
    text = final_reply if isinstance(final_reply, str) else str(final_reply)
    logger.info("Agent 回复完成 — session_id=%s reply_len=%s", request.session_id, len(text))
    return {
        "status": "completed",
        "session_id": request.session_id,
        "reply": text,
    }


# ── HITL 解冻模型 & 路由 ────────────────────────────────────────


class ApproveRequest(BaseModel):
    """HITL 人工确认请求。

    action 取值：
      - "continue"  → 确认通过，继续执行
      - "skip"      → 跳过当前步骤
      - "override"  → 覆盖路由决策（需配合 override_target）
    """
    session_id: str
    action: str = "continue"
    override_target: str | None = None


def _build_resume_payload(action: str, override_target: str | None) -> dict:
    """根据前端动作构建 interrupt() 的返回值（即 resume 载荷）。"""
    if action == "override" and override_target in ("device_agent", "manual_agent"):
        return {"override": override_target}
    elif action == "skip":
        return {"action": "skip"}
    else:
        return {"action": "continue"}


@app.post("/api/approve")
async def approve_agent_action(request: ApproveRequest):
    """
    HITL 解冻路由 — 人工确认/拒绝后恢复 Agent 图执行。

    前端在收到 /chat 返回 status="interrupted" 后，
    引导用户点击「确认/跳过/覆盖」，然后调用本接口解冻状态机。
    """
    logger.info(
        "[HITL] 收到解冻请求 — session_id=%s action=%s override=%s",
        request.session_id, request.action, request.override_target,
    )

    resume_payload = _build_resume_payload(request.action, request.override_target)
    config = {"configurable": {"thread_id": request.session_id}}
    command = Command(resume=resume_payload)

    try:
        result = _resume_agent_graph_with_retry(command, config)
    except GraphInterrupt:
        # 可能还有下一道门禁，继续返回 interrupted
        logger.info(
            "[HITL] Agent 再次挂起 — session_id=%s（仍有门禁待确认）",
            request.session_id,
        )
        return JSONResponse(
            status_code=200,
            content={
                "status": "interrupted",
                "session_id": request.session_id,
                "message": "仍有门禁需要确认，请继续调用 /api/approve。",
            },
        )
    except _retryable_types as e:
        logger.warning("[HITL] 恢复调用在 4 次重试后仍失败: %s", e)
        return JSONResponse(
            status_code=504,
            content={"error": "Agent 处理超时，请重试"},
        )

    final_reply = result["messages"][-1].content
    text = final_reply if isinstance(final_reply, str) else str(final_reply)
    logger.info(
        "[HITL] 解冻完成 — session_id=%s reply_len=%d",
        request.session_id, len(text),
    )
    return {
        "status": "completed",
        "session_id": request.session_id,
        "reply": text,
    }
