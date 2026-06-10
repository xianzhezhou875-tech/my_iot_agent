"""
LangGraph 多智能体编排图 — IoT 运维 Agent 核心调度引擎。

节点拓扑：
  supervisor → device_agent ⇄ device_tools → END
  supervisor → manual_agent ⇄ manual_tools → auditor ⇄ rewriter → END
"""

import os
import re
import sqlite3
import operator
from typing import Literal, Optional

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, interrupt
from typing_extensions import Annotated, TypedDict

from database_worker import query_user_device_tool
from llm_factory import (
    AUDITOR_PROMPT,
    DEVICE_AGENT_SYSTEM,
    MANUAL_AGENT_SYSTEM,
    SUPERVISOR_PROMPT,
    create_deepseek_brain,
)
from rag_worker import query_repair_manual_tool
from logging_config import logger

# ── 常量 ───────────────────────────────────────────────────────
MAX_REWRITE_ATTEMPTS = 2


# ── 状态定义 ───────────────────────────────────────────────────

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    rewrite_count: int


# ── 模型 & 工具绑定 ────────────────────────────────────────────

_brain = create_deepseek_brain()
_device_model = _brain.bind_tools([query_user_device_tool])
_manual_model = _brain.bind_tools([query_repair_manual_tool])
_device_tools = ToolNode([query_user_device_tool])
_manual_tools = ToolNode([query_repair_manual_tool])


# ── 工具函数 ───────────────────────────────────────────────────

def _last_user_text(messages: list) -> str:
    """从消息列表尾部向前取第一条 HumanMessage 的文本内容。"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def _last_tool_result_text(messages: list) -> Optional[str]:
    """从消息列表尾部向前取第一条 ToolMessage 的文本内容。"""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return None


def _last_assistant_reply_text(messages: list) -> str:
    """获取最后一条 AIMessage 的文本内容。"""
    last = messages[-1]
    if isinstance(last, AIMessage):
        content = last.content
        return content if isinstance(content, str) else str(content)
    return ""


def _agent_messages(state: MessagesState, system_prompt: str) -> list:
    """
    构建专家节点的消息列表：
    系统 Prompt 打头 + 对话历史（剔除其他专家的 SystemMessage 避免角色冲突）。
    """
    history = [
        m for m in state["messages"]
        if not isinstance(m, SystemMessage)
    ]
    return [SystemMessage(content=system_prompt), *history]


def _safe_llm_invoke(model, messages: list, node_name: str) -> AIMessage:
    """
    防御性 LLM 调用包装器。

    所有 Agent 节点统一走此函数，捕获网络/API 异常后返回一条友好的
    AIMessage 降级回复，而非让异常直接炸穿 LangGraph 图。

    Args:
        model:   LangChain ChatModel 实例
        messages: 消息列表
        node_name: 调用方节点名（用于日志上下文）

    Returns:
        AIMessage — 正常时为 LLM 回复，异常时为降级提示。
    """
    try:
        return model.invoke(messages)
    except Exception:
        logger.exception("%s LLM 调用失败，返回降级回复", node_name)
        return AIMessage(
            content="抱歉，AI 服务暂时不可用，请稍后重试。"
        )


# ── Supervisor 路由 ────────────────────────────────────────────

def _parse_supervisor_target(raw: str) -> Optional[Literal["device_agent", "manual_agent"]]:
    """
    解析 Supervisor 的纯文本回复，提取路由目标。
    兼容多种 LLM 输出格式：纯标签、带解释的短句、中文关键词。
    """
    text = raw.strip().lower()
    # 精确匹配
    if re.search(r"\bdevice_agent\b", text) or ("device" in text and "manual" not in text):
        return "device_agent"
    if re.search(r"\bmanual_agent\b", text) or "manual" in text:
        return "manual_agent"
    # 中文兜底
    if "设备" in raw and "手册" not in raw:
        return "device_agent"
    if any(k in raw for k in ("修理", "手册", "故障", "传感器")):
        return "manual_agent"
    return None


def _heuristic_route(user_text: str) -> Literal["device_agent", "manual_agent"]:
    """
    LLM 路由失败时的关键词兜底策略。
    统计用户输入中的设备类/手册类关键词命中数，多数胜出。
    """
    text = user_text.lower()
    manual_hints = (
        "修理", "维修", "手册", "故障", "传感器",
        "正常吗", "温度", "原理", "怎么修", "向量",
    )
    device_hints = ("设备", "拥有", "名下", "清单", "哪些", "绑定", "小明")
    manual_score = sum(1 for k in manual_hints if k in text)
    device_score = sum(1 for k in device_hints if k in text)

    target = "device_agent" if device_score >= manual_score else "manual_agent"
    logger.info("关键词兜底路由 → %s (device=%d manual=%d)", target, device_score, manual_score)
    return target


def supervisor_node(state: MessagesState) -> Command:
    """
    经理节点 — 分析用户意图，按需分诊（dispatch /dɪˈspætʃ/）到设备专家或手册专家。

    注意：DeepSeek 不支持 OpenAI structured output（json_schema），
    故用纯文本回复 + 正则解析 + 关键词兜底。
    """
    user_text = _last_user_text(state["messages"])
    prompt = (
        f"{SUPERVISOR_PROMPT}\n\n"
        f"【用户问题】\n{user_text}\n\n"
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
            logger.debug("Supervisor LLM 回复: %r → %s", raw, target)
    except Exception:
        logger.exception("Supervisor LLM 调用异常，使用关键词兜底")
        target = _heuristic_route(user_text)

    logger.info("Supervisor 路由 → %s", target)

    # ── HITL 门禁：高危路由需人工确认 ──
    # interrupt() 暂停图执行，将当前 State 写入 checkpointer；
    # 外部通过 Command(resume=<value>) 恢复，返回值即 resume 传入的值。
    human_decision = interrupt({
        "gate": "supervisor_routing",
        "question": user_text,
        "proposed_target": target,
        "message": (
            f"Supervisor 建议路由到 {target}。"
            "回复 'y' 确认，或输入 'device_agent'/'manual_agent' 覆盖。"
        ),
    })

    # 处理人工覆盖
    if isinstance(human_decision, dict):
        override = human_decision.get("override")
        if override in ("device_agent", "manual_agent"):
            logger.info(
                "人工覆盖路由: %s → %s",
                target, override,
            )
            target = override
    elif isinstance(human_decision, str):
        decision_lower = human_decision.strip().lower()
        if decision_lower in ("device_agent", "manual_agent"):
            logger.info("人工覆盖路由(文本): %s", decision_lower)
            target = decision_lower

    logger.info("Supervisor 最终路由（经 HITL 确认）→ %s", target)
    return Command(goto=target)


# ── 设备专家 ────────────────────────────────────────────────────

def device_agent_node(state: MessagesState) -> dict:
    """设备专家 — 查询用户名下 IoT 设备归属（含 HITL 门禁）。"""
    user_question = _last_user_text(state["messages"])

    # ── HITL 门禁：设备查询前人工确认 ──
    approval = interrupt({
        "gate": "device_query_confirm",
        "question": user_question,
        "message": "即将查询设备数据库。回复 'y' 继续，或 'skip' 跳过。",
    })

    if isinstance(approval, dict) and approval.get("action") == "skip":
        logger.info("人工跳过设备查询: %s", user_question[:80])
        return {
            "messages": [AIMessage(
                content="设备查询已被人工跳过，请告知需要什么帮助。"
            )]
        }

    logger.info("设备查询 HITL 确认通过，继续执行")
    messages = _agent_messages(state, DEVICE_AGENT_SYSTEM)
    response = _safe_llm_invoke(_device_model, messages, "device_agent")
    return {"messages": [response]}


def route_after_device(state: MessagesState) -> str:
    """设备专家后路由：有 tool_call → 执行工具，否则 → 结束。"""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "device_tools"
    return END


# ── 手册专家 ────────────────────────────────────────────────────

def manual_agent_node(state: MessagesState) -> dict:
    """手册专家 — 检索维修手册 & 技术原理（含 HITL 门禁）。"""
    user_question = _last_user_text(state["messages"])

    # ── HITL 门禁：RAG 检索前人工确认 ──
    approval = interrupt({
        "gate": "manual_query_confirm",
        "question": user_question,
        "message": "即将检索维修手册知识库。回复 'y' 继续，或 'skip' 跳过。",
    })

    if isinstance(approval, dict) and approval.get("action") == "skip":
        logger.info("人工跳过手册检索: %s", user_question[:80])
        return {
            "messages": [AIMessage(
                content="手册检索已被人工跳过，请告知需要什么帮助。"
            )]
        }

    logger.info("手册检索 HITL 确认通过，继续执行")
    messages = _agent_messages(state, MANUAL_AGENT_SYSTEM)
    response = _safe_llm_invoke(_manual_model, messages, "manual_agent")
    return {"messages": [response]}


def route_after_manual(state: MessagesState) -> str:
    """手册专家后路由：有 tool_call → 执行工具 → auditor 质检。"""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "manual_tools"
    return "auditor"


# ── 审计 & 重写 ─────────────────────────────────────────────────

def _audit_passed(raw: str) -> bool:
    """
    解析审计员输出。
    严格模式：仅在明确包含 PASS 且不含 FAIL 时返回 True。
    避免 LLM 输出 "FAIL。" / "FAIL（原因）" 等变体漏判。
    """
    text = raw.strip().upper()
    if not text:
        return True  # 空输出宽松处理，放行
    if "PASS" in text and "FAIL" not in text:
        return True
    if "FAIL" in text:
        return False
    # 既无 PASS 也无 FAIL → 容错放行
    logger.debug("审计结果模糊，容错放行: %r", raw[:100])
    return True


def node_audit_answer(state: MessagesState) -> Command:
    """
    审计节点 — 核对 AI 最终回答是否忠于 RAG 参考资料。

    审计未通过 → 进入 rewriter 节点重写；
    审计通过 / 已达最大重试次数 / 无工具结果可审 → 直接结束。
    """
    rewrites = state.get("rewrite_count", 0)
    if rewrites >= MAX_REWRITE_ATTEMPTS:
        logger.warning("已达最大重写次数 %s，强制结束审计循环", MAX_REWRITE_ATTEMPTS)
        return Command(goto=END)

    tool_ref = _last_tool_result_text(state["messages"])
    if not tool_ref:
        logger.debug("审计跳过 — 消息中无工具结果可供核对")
        return Command(goto=END)

    final_answer = _last_assistant_reply_text(state["messages"])
    if not final_answer.strip():
        logger.debug("审计跳过 — 无助手最终回复文本")
        return Command(goto=END)

    # 调用审计 LLM
    try:
        brain = create_deepseek_brain()
        prompt = AUDITOR_PROMPT.format(rag_info=tool_ref, final_answer=final_answer)
        res = brain.invoke(prompt)
        raw_audit = res.content if isinstance(res.content, str) else str(res.content)
        logger.debug("审计原始输出: %s", raw_audit)
    except Exception:
        logger.exception("审计 LLM 调用异常，容错放行直接结束")
        return Command(goto=END)

    if _audit_passed(raw_audit):
        logger.info("审计 PASS，回答可信")
        return Command(goto=END)

    logger.info("审计 FAIL (rewrite_count=%s)，进入重写节点", rewrites)
    return Command(goto="rewriter")


def node_rewrite(state: MessagesState) -> dict:
    """
    重写节点 — 审计未通过时，强制基于 RAG 原始资料重新生成回答。

    逻辑：
      - 将工具返回的原始资料作为 HumanMessage 注入对话，
      - 要求 LLM 严格基于资料重新回答，杜绝幻觉（Hallucination /həˌluːsɪˈneɪʃən/）。
    """
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
    messages = _agent_messages(state, MANUAL_AGENT_SYSTEM) + [rewrite_msg]
    response = _safe_llm_invoke(_manual_model, messages, "rewriter")

    logger.info("重写完成 (第 %s 次)", rewrites)
    return {"messages": [response], "rewrite_count": rewrites}


# ── 图构建 ──────────────────────────────────────────────────────

workflow = StateGraph(MessagesState)

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

# ── HITL 持久化快照底座（SqliteSaver + WAL） ──
# 快照存储在项目目录下 data/checkpoints/，随项目迁移。
# WAL 模式 + busy_timeout 全面防御高并发锁死。
_CP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "checkpoints")
os.makedirs(_CP_DIR, exist_ok=True)
logger.info("HITL 快照目录已就绪: %s", _CP_DIR)

_conn = sqlite3.connect(
    os.path.join(_CP_DIR, "checkpoint.db"),
    timeout=10.0,             # 数据库锁等待上限（秒）
    check_same_thread=False,  # FastAPI 多 worker 线程安全
)
_conn.execute("PRAGMA journal_mode=WAL;")
_conn.execute("PRAGMA busy_timeout=5000;")   # 5 秒忙等待（毫秒）
logger.info("SqliteSaver 已连接 — WAL 模式 activated | busy_timeout=5000ms")

_checkpointer = SqliteSaver(_conn)
app = workflow.compile(checkpointer=_checkpointer)
# interrupt() 机制天然防止死循环，recursion_limit 不再需要手动传


# ── 本地调试入口 ───────────────────────────────────────────────

if __name__ == "__main__":
    """
    HITL 双段式状态机本地调试入口。

    双段式的含义：
      - 「冻结段」→ interrupt() 触发，GraphInterrupt 异常抛出，State 写入 checkpointer。
      - 「复活段」→ Command(resume=<value>) 重新 invoke，从冻结点恢复执行。

    每条 HITL 门禁 = 一次冻结 + 一次复活，多个门禁 = 多轮循环。
    本查询 "传感器 100 度正常吗" 预期路径：
      supervisor(冻结→复活) → manual_agent(冻结→复活) → tools → auditor → END
    """
    from logging_config import configure_logging
    configure_logging()

    from database_worker import init_db
    from rag_worker import init_rag
    from langgraph.errors import GraphInterrupt

    init_db()
    init_rag()

    # ── 调试会话配置 ──
    # checkpointer 靠 thread_id 区分不同会话快照，多轮 resume 必须用同一个 ID。
    config = {"configurable": {"thread_id": "debug-hitl-001"}}

    msg = HumanMessage(
        content=(
            "传感器 A 是 100 度正常吗？,少于50字回答我，"
            "并且把你在向量库中查询到的内容告诉我不管匹不匹配"
        )
    )

    logger.info("=" * 55)
    logger.info("HITL 多段式状态机调试启动 | thread_id=%s", config["configurable"]["thread_id"])
    logger.info("查询: %s", msg.content[:70])
    logger.info("=" * 55)

    # ═════════════════════════════════════════════════════════════
    # 阶段-1：首次 invoke → 预期冻结在 supervisor 路由门禁
    # ═════════════════════════════════════════════════════════════
    logger.info("[阶段-1] 首次 invoke，预期冻结在 supervisor 路由门禁...")

    try:
        result = app.invoke(
            {"messages": [msg], "rewrite_count": 0},
            config=config,
        )
        # 防御：某些 LangGraph 版本不抛 GraphInterrupt，而是静默写入返回值
        interrupted = isinstance(result, dict) and "__interrupt__" in result
        if interrupted:
            logger.info("[阶段-1] ✅ 静默挂起 — __interrupt__ 字段已检测到")
        else:
            logger.info("[阶段-1] ⚠️ 未触发任何门禁，图直接跑完（是否没装 checkpointer？）")
            last_content = result["messages"][-1].content
            logger.info("最终回复: %s", last_content[:200] if isinstance(last_content, str) else str(last_content)[:200])
    except GraphInterrupt:
        logger.info("[阶段-1] ✅ GraphInterrupt 捕获 — supervisor 路由门禁已挂起")

        # ═════════════════════════════════════════════════════════
        # 阶段-2：确认路由 → 复活 → 预期冻结在 expert 执行门禁
        # ═════════════════════════════════════════════════════════
        logger.info("[阶段-2] Command(resume={'action': 'continue'}) 复活中...")

        try:
            result = app.invoke(
                Command(resume={"action": "continue"}),
                config=config,
            )
            interrupted2 = isinstance(result, dict) and "__interrupt__" in result
            if interrupted2:
                logger.info("[阶段-2] ✅ 静默挂起 — __interrupt__ 字段检测到")
            else:
                logger.info("[阶段-2] ⚠️ 未触发二次门禁，图直接完成")
                last_content = result["messages"][-1].content
                logger.info("最终回复: %s", last_content[:200] if isinstance(last_content, str) else str(last_content)[:200])
        except GraphInterrupt:
            logger.info("[阶段-2] ✅ GraphInterrupt 捕获 — expert 执行门禁已挂起")

            # ═════════════════════════════════════════════════════
            # 阶段-3：确认 expert 执行 → 最终复活 → 冲向终局
            # ═════════════════════════════════════════════════════
            logger.info("[阶段-3] Command(resume={'action': 'continue'}) 最终复活...")

            try:
                result = app.invoke(
                    Command(resume={"action": "continue"}),
                    config=config,
                )
                logger.info("[阶段-3] ✅ 图执行完成，无更多门禁中断")

                final_msg = result["messages"][-1]
                content = final_msg.content
                content_str = content if isinstance(content, str) else str(content)

                logger.info("=" * 55)
                logger.info("AI 最终回复:\n%s", content_str)
                logger.info("=" * 55)
                logger.info(
                    "统计 | 消息总数=%d | 重写次数=%d",
                    len(result["messages"]),
                    result.get("rewrite_count", 0),
                )
            except GraphInterrupt:
                logger.info("[阶段-3] ⚠️ 仍有未预期的门禁挂起 — 请检查是否新增了 interrupt()")
            except Exception:
                logger.exception("[阶段-3] 非预期异常")
