import logging
import re
from typing import Annotated, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from typing_extensions import TypedDict

from database_worker import query_user_device_tool
from llm_factory import (
    AUDITOR_PROMPT,
    DEVICE_AGENT_SYSTEM,
    MANUAL_AGENT_SYSTEM,
    SUPERVISOR_PROMPT,
    create_deepseek_brain,
)
from rag_worker import query_repair_manual_tool

logger = logging.getLogger(__name__)

MAX_REWRITE_ATTEMPTS = 2


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    rewrite_count: int


_brain = create_deepseek_brain()
_device_model = _brain.bind_tools([query_user_device_tool])
_manual_model = _brain.bind_tools([query_repair_manual_tool])
_device_tools = ToolNode([query_user_device_tool])
_manual_tools = ToolNode([query_repair_manual_tool])


def _last_user_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def _last_tool_result_text(messages: list) -> Optional[str]:
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return None


def _audit_passed(raw: str) -> bool:
    """解析审计员输出，避免「FAIL。」等格式导致永远进 rewriter。"""
    text = raw.strip().upper()
    if not text:
        return True
    if "PASS" in text and "FAIL" not in text:
        return True
    if "FAIL" in text:
        return False
    return True


def _last_assistant_reply_text(messages: list) -> str:
    last = messages[-1]
    if isinstance(last, AIMessage):
        content = last.content
        return content if isinstance(content, str) else str(content)
    return ""


def _heuristic_route(user_text: str) -> Literal["device_agent", "manual_agent"]:
    """DeepSeek 路由失败时的关键词兜底。"""
    text = user_text.lower()
    manual_hints = (
        "修理",
        "维修",
        "手册",
        "故障",
        "传感器",
        "正常吗",
        "温度",
        "原理",
        "怎么修",
        "向量",
    )
    device_hints = ("设备", "拥有", "名下", "清单", "哪些", "绑定", "小明")
    manual_score = sum(1 for k in manual_hints if k in text)
    device_score = sum(1 for k in device_hints if k in text)
    if device_score > manual_score:
        return "device_agent"
    if manual_score > device_score:
        return "manual_agent"
    return "device_agent"


def _parse_supervisor_target(raw: str) -> Optional[Literal["device_agent", "manual_agent"]]:
    text = raw.strip().lower()
    if re.search(r"\bdevice_agent\b", text) or "device" in text and "manual" not in text:
        return "device_agent"
    if re.search(r"\bmanual_agent\b", text) or "manual" in text:
        return "manual_agent"
    if "设备" in raw and "手册" not in raw:
        return "device_agent"
    if any(k in raw for k in ("修理", "手册", "故障", "传感器")):
        return "manual_agent"
    return None


def _agent_messages(state: AgentState, system_prompt: str) -> list:
    """专家节点：系统 Prompt + 对话历史（不含其它专家的 SystemMessage）。"""
    history = [
        m
        for m in state["messages"]
        if not isinstance(m, SystemMessage)
    ]
    return [SystemMessage(content=system_prompt), *history]


def supervisor_node(state: AgentState) -> Command:
    """经理：只决策，不调用业务工具。

    注意：DeepSeek 不支持 OpenAI 的 json_schema structured output，
    故用纯文本回复 + 解析，失败时用关键词兜底。
    """
    user_text = _last_user_text(state["messages"])
    prompt = (
        f"{SUPERVISOR_PROMPT}\n\n【用户问题】\n{user_text}\n\n"
        "请只回复 exactly 一个词组，不要解释：\n"
        "- device_agent\n"
        "- manual_agent"
    )
    try:
        res = _brain.invoke(prompt)
        raw = res.content if isinstance(res.content, str) else str(res.content)
        target = _parse_supervisor_target(raw)
        if target is None:
            logger.warning("Supervisor 无法解析 LLM 回复 %r，使用关键词兜底", raw)
            target = _heuristic_route(user_text)
        else:
            logger.debug("Supervisor LLM 回复: %r -> %s", raw, target)
    except Exception as e:
        logger.warning("Supervisor LLM 调用失败 (%s)，使用关键词兜底", e)
        target = _heuristic_route(user_text)

    logger.info("Supervisor 路由 -> %s", target)
    return Command(goto=target)


def device_agent_node(state: AgentState):
    response = _device_model.invoke(_agent_messages(state, DEVICE_AGENT_SYSTEM))
    return {"messages": [response]}


def manual_agent_node(state: AgentState):
    response = _manual_model.invoke(_agent_messages(state, MANUAL_AGENT_SYSTEM))
    return {"messages": [response]}


def route_after_device(state: AgentState) -> str:
    if state["messages"][-1].tool_calls:
        return "device_tools"
    # 设备/SQL 结果不走 RAG 质检，直接结束
    return END


def route_after_manual(state: AgentState) -> str:
    if state["messages"][-1].tool_calls:
        return "manual_tools"
    return "auditor"


def node_audit_answer(state: AgentState) -> Command:
    rewrites = state.get("rewrite_count", 0)
    if rewrites >= MAX_REWRITE_ATTEMPTS:
        logger.warning("已达最大重写次数 %s，强制结束", MAX_REWRITE_ATTEMPTS)
        return Command(goto=END)

    messages = state["messages"]
    tool_ref = _last_tool_result_text(messages)
    if not tool_ref:
        logger.debug("审计跳过：messages 中无工具结果")
        return Command(goto=END)

    final_answer = _last_assistant_reply_text(messages)
    if not final_answer.strip():
        logger.debug("审计跳过：无助手最终文本")
        return Command(goto=END)

    brain = create_deepseek_brain()
    prompt = AUDITOR_PROMPT.format(rag_info=tool_ref, final_answer=final_answer)
    res = brain.invoke(prompt)
    raw_audit = res.content if isinstance(res.content, str) else str(res.content)
    logger.debug("审计原始输出: %s", raw_audit)
    if _audit_passed(raw_audit):
        return Command(goto=END)

    logger.info("审计未通过 (rewrite_count=%s)，进入 rewriter", rewrites)
    return Command(goto="rewriter")


def node_rewrite(state: AgentState):
    rewrites = state.get("rewrite_count", 0) + 1
    tool_ref = _last_tool_result_text(state["messages"])
    if tool_ref:
        prompt = (
            "由于审计失败，请严格基于下面【工具返回的参考资料】重新组织回答。"
            "可以引用用户问题中的数值做对比，但不要编造资料以外的技术参数。"
            "直接回答用户问题：\n\n"
            f"【工具返回的参考资料】\n{tool_ref}"
        )
    else:
        prompt = "由于审计失败，请根据对话里已有工具返回内容重新组织回答，确保严谨。"

    rewrite_msg = HumanMessage(content=prompt)
    new_response = _manual_model.invoke(
        _agent_messages(state, MANUAL_AGENT_SYSTEM) + [rewrite_msg]
    )
    return {"messages": [new_response], "rewrite_count": rewrites}


workflow = StateGraph(AgentState)

workflow.add_node(
    "supervisor",
    supervisor_node,
    destinations=("device_agent", "manual_agent"),
)
workflow.add_node("device_agent", device_agent_node)
workflow.add_node("manual_agent", manual_agent_node)
workflow.add_node("device_tools", _device_tools)
workflow.add_node("manual_tools", _manual_tools)
workflow.add_node(
    "auditor",
    node_audit_answer,
    destinations=(END, "rewriter"),
)
workflow.add_node("rewriter", node_rewrite)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "device_agent",
    route_after_device,
    {"device_tools": "device_tools", END: END},
)
workflow.add_conditional_edges(
    "manual_agent",
    route_after_manual,
    {"manual_tools": "manual_tools", "auditor": "auditor"},
)

workflow.add_edge("device_tools", "device_agent")
workflow.add_edge("manual_tools", "manual_agent")
workflow.add_edge("rewriter", "auditor")

app = workflow.compile()
# 调用时可传 config={"recursion_limit": 30}，防止异常图结构死循环


if __name__ == "__main__":
    from logging_config import configure_logging

    configure_logging()
    from database_worker import init_db
    from rag_worker import init_rag

    init_db()
    init_rag()

    msg = HumanMessage(
        content=(
            "传感器 A 是 100 度正常吗？,少于50字回答我，"
            "并且把你在向量库中查询到的内容告诉我不管匹不匹配"
        )
    )
    result = app.invoke({"messages": [msg], "rewrite_count": 0})
    last = result["messages"][-1].content
    logger.info("AI 回复: %s", last if isinstance(last, str) else str(last))
