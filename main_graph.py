import logging
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.types import Command
from typing import Annotated
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

from database_worker import query_user_device_tool
from llm_factory import create_deepseek_brain, AUDITOR_PROMPT
from rag_worker import query_repair_manual_tool

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    # 工具结果与最终回答均通过 ToolNode / LLM 写入 messages，不在此重复维护
    messages: Annotated[list, add_messages]


# 1. 建立工具集（确保 query_* 已加 @tool）
tools = [query_user_device_tool, query_repair_manual_tool]
tool_node = ToolNode(tools)
model_with_tools = create_deepseek_brain().bind_tools(tools)


def _last_tool_result_text(messages: list) -> Optional[str]:
    """从 messages 中取时间顺序上最后一次工具返回（ToolNode 写入的 ToolMessage）。"""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return None


def _last_assistant_reply_text(messages: list) -> str:
    """进入审计前，最后一条应为不带 tool_calls 的 AIMessage。"""
    last = messages[-1]
    if isinstance(last, AIMessage):
        content = last.content
        return content if isinstance(content, str) else str(content)
    return ""


def call_model(state: AgentState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def node_audit_answer(state: AgentState) -> Command:
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
    audit = res.content.strip().upper()
    logger.debug("审计结果: %s", audit)
    if audit == "PASS":
        return Command(goto=END)
    logger.info("审计未通过，进入 rewriter")
    return Command(goto="rewriter")


def node_rewrite(state: AgentState):
    tool_ref = _last_tool_result_text(state["messages"])
    if tool_ref:
        prompt = (
            "由于审计失败，请严格基于下面【工具返回的参考资料】重新组织回答，"
            "不要编造其中没有的事实，直接回答用户问题：\n\n"
            f"【工具返回的参考资料】\n{tool_ref}"
        )
    else:
        prompt = "由于审计失败，请根据对话里已有工具返回内容重新组织回答，确保严谨。"

    rewrite_msg = HumanMessage(content=prompt)
    new_response = model_with_tools.invoke(state["messages"] + [rewrite_msg])
    return {"messages": [new_response]}


def check_if_done(state: AgentState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return "auditor"


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node(
    "auditor",
    node_audit_answer,
    destinations=(END, "rewriter"),
)
workflow.add_node("rewriter", node_rewrite)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    check_if_done,
    {"tools": "tools", "auditor": "auditor"},
)

workflow.add_edge("tools", "agent")
workflow.add_edge("rewriter", "auditor")

app = workflow.compile()


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
    result = app.invoke({"messages": [msg]})
    last = result["messages"][-1].content
    logger.info("AI 回复: %s", last if isinstance(last, str) else str(last))
