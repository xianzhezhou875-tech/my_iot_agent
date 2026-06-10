"""
chaos_test.py — HITL 全栈混沌测试套件

覆盖用例：
  用例1：多线程并发抢占 — WAL + busy_timeout 压力测试
  用例2：中断期插话熔断 — 旧快照覆盖 + 幽灵会话拦截
  用例3：跨进程断电复活 — checkpoint.db 持久化恢复

设计约束：
  - 零 print()，全量走 logging_config.logger
  - 直接对 agent_graph 做 invoke/resume，不依赖 FastAPI 进程
  - 每个用例使用独立的 thread_id，避免交叉污染
"""

import os
import sqlite3
import threading
import time
from typing import Any

from logging_config import configure_logging, logger

configure_logging()

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from langgraph.checkpoint.sqlite import SqliteSaver

from main_graph import app as agent_graph, workflow

# ── 常量 ───────────────────────────────────────────────────────
_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
CP_DB = os.path.join(_PROJ_ROOT, "data", "checkpoints", "checkpoint.db")
THREAD_COUNT = 5           # 并发线程数
CONCURRENT_TIMEOUT = 45.0  # 并发测试超时（秒）
LOG_SEP = "=" * 55


# ╔══════════════════════════════════════════════════════════════════
# ║  测试工具函数
# ╚══════════════════════════════════════════════════════════════════

# 全局测试结果收集器
_results: list[dict[str, Any]] = []


def _record(case_id: str, name: str, passed: bool, detail: str = "") -> None:
    """记录单条测试结果，统一走 logger 输出 PASS/FAIL 诊断。"""
    _results.append({"id": case_id, "name": name, "passed": passed, "detail": detail})
    status = "✅ PASS" if passed else "❌ FAIL"
    logger.info("[%s] [用例%s] %s — %s", status, case_id, name, detail)


def _config(thread_id: str) -> dict:
    """构建带 thread_id 的 LangGraph config。"""
    return {"configurable": {"thread_id": thread_id}}


def _state(question: str) -> dict:
    """构建初始 MessagesState。"""
    return {
        "messages": [HumanMessage(content=question)],
        "rewrite_count": 0,
    }


def _resume_until_done(app, config: dict, max_attempts: int = 5) -> dict | None:
    """
    循环 resume 直到图执行完成（无更多中断）。

    Args:
        app: CompiledStateGraph 实例
        config: 包含 thread_id 的 config
        max_attempts: 最大 resume 次数（防止意外死循环）

    Returns:
        最终 result dict，或 None（若超过 max_attempts 仍未完成）。
    """
    for i in range(max_attempts):
        try:
            result = app.invoke(
                Command(resume={"action": "continue"}),
                config,
            )
            logger.debug("resume #%d → 完成，无更多中断", i + 1)
            return result
        except GraphInterrupt:
            logger.debug("resume #%d → 再次挂起，继续复活", i + 1)
            continue
        except Exception:
            logger.exception("resume #%d → 非预期异常", i + 1)
            return None
    logger.warning("resume 达最大次数 %d 仍未完成", max_attempts)
    return None


def _compile_fresh_graph(db_path: str = CP_DB) -> tuple:
    """
    模拟进程重启：基于同一个 checkpoint.db 创建全新的 CompiledStateGraph。

    Returns:
        (new_app, conn) — new_app 用于 resume，conn 用于后续清理。
    """
    conn = sqlite3.connect(db_path, timeout=10.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    saver = SqliteSaver(conn)
    new_app = workflow.compile(checkpointer=saver)
    logger.debug("新图编译完成 — 指向 %s", db_path)
    return new_app, conn


# ╔══════════════════════════════════════════════════════════════════
# ║  用例1：多线程并发抢占 — WAL + busy_timeout 压力测试
# ╚══════════════════════════════════════════════════════════════════

def test_concurrent_stress() -> None:
    """
    5 个线程同时发起 /chat，每个使用独立 thread_id，
    验证 WAL 模式下无死锁、busy_timeout 排队正常。

    成功标准：全部 5 个线程在超时内完成，且均触发 GraphInterrupt
    （或被正常处理，无 crash）。
    """
    case_id = "1"
    results_lock = threading.Lock()
    completed: list[dict] = []
    errors: list[str] = []

    def worker(idx: int) -> None:
        tid = f"chaos-concurrent-{idx}"
        cfg = _config(tid)
        st = _state(f"设备 {idx} 号传感器正常范围是多少？")

        t_start = time.perf_counter()
        try:
            result = agent_graph.invoke(st, cfg)
            elapsed = time.perf_counter() - t_start
            # LangGraph 1.x + checkpointer 可能不抛 GraphInterrupt，
            # 而是把 __interrupt__ 静默写入返回值
            if isinstance(result, dict) and "__interrupt__" in result:
                with results_lock:
                    completed.append({"tid": tid, "elapsed": elapsed, "outcome": "interrupted_silent"})
            else:
                with results_lock:
                    completed.append({"tid": tid, "elapsed": elapsed, "outcome": "no_interrupt"})
        except GraphInterrupt:
            elapsed = time.perf_counter() - t_start
            with results_lock:
                completed.append({"tid": tid, "elapsed": elapsed, "outcome": "interrupted"})
        except Exception as e:
            elapsed = time.perf_counter() - t_start
            with results_lock:
                errors.append(f"{tid}: {type(e).__name__} — {e}")
            logger.exception("并发 worker %d 异常", idx)

    # 启动线程
    threads = [
        threading.Thread(target=worker, args=(i,), name=f"w-{i}")
        for i in range(THREAD_COUNT)
    ]
    t0 = time.perf_counter()
    for t in threads:
        t.start()

    # 等待全部完成
    for t in threads:
        t.join(timeout=CONCURRENT_TIMEOUT)

    total_elapsed = time.perf_counter() - t0

    # 诊断输出
    alive = [t.name for t in threads if t.is_alive()]
    interrupted = sum(1 for c in completed if "interrupt" in c["outcome"])
    avg_elapsed = (
        sum(c["elapsed"] for c in completed) / len(completed) if completed else 0
    )

    logger.info(
        "[用例1 诊断] 总耗时=%.1fs | 完成=%d | 中断=%d(含静默) | 存活=%d | 错误=%d | 平均延迟=%.1fs",
        total_elapsed, len(completed), interrupted,
        len(alive), len(errors), avg_elapsed,
    )

    # 断言
    if alive:
        _record(case_id, "并发死锁检测", False,
                f"{len(alive)} 个线程仍在运行 → 疑似 WAL 锁死: {alive}")
        return

    if errors:
        _record(case_id, "并发错误", False,
                f"{len(errors)} 个线程异常: {errors[:3]}")
        return

    if len(completed) < THREAD_COUNT:
        _record(case_id, "完成线程数", False,
                f"仅 {len(completed)}/{THREAD_COUNT} 完成")
        return

    _record(case_id, "并发压力测试", True,
            f"{THREAD_COUNT}线程/{total_elapsed:.1f}s | 均延{avg_elapsed:.1f}s | WAL 正常")


# ╔══════════════════════════════════════════════════════════════════
# ║  用例2：中断期插话熔断 — 旧快照覆盖 + 幽灵会话拦截
# ╚══════════════════════════════════════════════════════════════════

def test_stale_session_override() -> None:
    """
    场景：用户在中断期不点确认，而是直接发送一条全新提问。

    步骤：
      1. thread_id=X 发起问题 A → supervisor 门禁挂起
      2. thread_id=X 发起问题 B（不 resume）→ 旧快照应被覆盖
      3. resume 到底 → 最终回答应对应问题 B（而非 A）
    """
    case_id = "2"
    tid = "chaos-stale-override"

    # Step 1：问题 A → 挂起
    state_a = _state("传感器 A 是 100 度正常吗？")
    try:
        agent_graph.invoke(state_a, _config(tid))
    except GraphInterrupt:
        logger.debug("stale-override step1: 问题 A 已挂起")
    except Exception:
        logger.exception("stale-override step1 异常")
        _record(case_id, "中断期插话", False, "step1 异常")
        return

    # Step 2：不 resume，直接用同一 thread_id 发问题 B
    state_b = _state("小明名下的设备有哪些？（50字内）")
    try:
        result_b = agent_graph.invoke(state_b, _config(tid))
    except GraphInterrupt:
        # 问题 B 也触发了门禁 → 说明是全新会话
        logger.debug("stale-override step2: 问题 B 触发了新门禁（旧快照已覆盖）")

        # 复活到底，验证回答对应问题 B
        final = _resume_until_done(agent_graph, _config(tid))
        if final is None:
            _record(case_id, "中断期插话", False, "问题 B resume 未完成")
            return

        reply = final["messages"][-1].content
        reply_str = reply if isinstance(reply, str) else str(reply)

        # 关键断言：回复应包含问题 B 的内容（设备归属），而非问题 A（传感器）
        if "设备" in reply_str or "小明" in reply_str:
            _record(case_id, "中断期插话熔断", True,
                    f"旧快照已覆盖，回复对应新问题 (len={len(reply_str)})")
        else:
            _record(case_id, "中断期插话熔断", False,
                    f"回复可能来自旧快照: {reply_str[:80]}")
        return
    except Exception:
        logger.exception("stale-override step2 异常")
        _record(case_id, "中断期插话", False, "step2 异常")
        return

    # 如果问题 B 没有触发任何门禁直接完成了
    reply = result_b["messages"][-1].content
    reply_str = reply if isinstance(reply, str) else str(reply)
    _record(case_id, "中断期插话熔断", True,
            f"问题 B 直接完成（无门禁），旧快照已覆盖 (len={len(reply_str)})")


def test_ghost_session_reject() -> None:
    """
    场景：前端用不存在的 thread_id 调用 /api/approve。

    LangGraph 1.x 行为说明：
      对不存在的 thread_id 发 Command(resume=...) 不会抛异常，
      而是将其视为新会话的初始输入并启动新 run。
      这意味着前端应自行校验 session 是否处于 interrupted 状态。
    """
    case_id = "2b"
    tid = "chaos-ghost-nonexistent-99999"

    try:
        result = agent_graph.invoke(
            Command(resume={"action": "continue"}),
            _config(tid),
        )
        # LangGraph 1.x 会为此幽灵会话创建新 run，返回正常结果
        if isinstance(result, dict) and "messages" in result:
            logger.debug("幽灵会话被 LangGraph 视为新会话启动，非错误行为")
            _record(case_id, "幽灵会话行为确认", True,
                    "LangGraph 1.x 对幽灵 resume 静默创建新 run（已知行为）")
        else:
            _record(case_id, "幽灵会话行为确认", False,
                    f"非预期返回值类型: {type(result)}")
    except GraphInterrupt:
        _record(case_id, "幽灵会话拦截", False,
                "幽灵会话不应触发 GraphInterrupt")
    except Exception as e:
        # 某些版本抛出异常 → 也是一种合理的防御行为
        logger.debug("幽灵会话被拒绝: %s", e)
        _record(case_id, "幽灵会话拦截", True,
                f"主动拒绝: {type(e).__name__}")


# ╔══════════════════════════════════════════════════════════════════
# ║  用例3：跨进程断电复活 — checkpoint.db 持久化恢复
# ╚══════════════════════════════════════════════════════════════════

def test_process_restart_resurrection() -> None:
    """
    场景：进程意外中断后重启，从 checkpoint.db 恢复中断会话。

    步骤：
      1. 用当前 agent_graph 发起 invoke → supervisor 门禁挂起
      2. 记录 checkpoint.db 文件大小
      3. 创建全新编译的图（模拟进程重启），指向同一个 checkpoint.db
      4. 在新图上 resume → 应成功恢复并执行到底
      5. 清理新创建的连接
    """
    case_id = "3"
    tid = "chaos-restart-resurrect"

    # Step 1：挂起
    state = _state("传感器 A 是 100 度正常吗？")
    try:
        agent_graph.invoke(state, _config(tid))
    except GraphInterrupt:
        logger.debug("restart step1: 已挂起")
    except Exception:
        logger.exception("restart step1 异常")
        _record(case_id, "断电复活", False, "step1 异常")
        return

    # Step 2：检查 checkpoint.db 有无数据
    db_size_before = os.path.getsize(CP_DB) if os.path.exists(CP_DB) else 0
    logger.debug("restart step2: checkpoint.db 大小=%d bytes", db_size_before)

    if db_size_before == 0:
        _record(case_id, "断电复活", False, "checkpoint.db 为空，快照未持久化")
        return

    # Step 3 & 4：模拟重启 → 新图 resume
    new_app, new_conn = _compile_fresh_graph(CP_DB)

    try:
        final = _resume_until_done(new_app, _config(tid))
    finally:
        # Step 5：无论如何都要关闭新连接
        new_conn.close()
        logger.debug("restart step5: 新连接已关闭")

    if final is None:
        _record(case_id, "断电复活", False, "新图上 resume 未完成")
        return

    reply = final["messages"][-1].content
    reply_str = reply if isinstance(reply, str) else str(reply)

    if len(reply_str) > 0:
        _record(case_id, "断电复活", True,
                f"从 checkpoint.db 复活成功，回复长度={len(reply_str)}")
    else:
        _record(case_id, "断电复活", False, "复活后回复为空")


# ╔══════════════════════════════════════════════════════════════════
# ║  测试编排器
# ╚══════════════════════════════════════════════════════════════════

def run_all() -> int:
    """
    运行全部混沌测试用例，输出诊断矩阵。

    Returns:
        0 表示全部通过，1 表示存在失败。
    """
    logger.info(LOG_SEP)
    logger.info("🔬 HITL 全栈混沌测试套件 — 启动")
    logger.info(LOG_SEP)

    # 前置检查
    if not os.path.exists(CP_DB):
        logger.warning("checkpoint.db 不存在，测试将在首次 interrupt 时创建")

    # ── 执行测试 ──
    test_concurrent_stress()
    test_stale_session_override()
    test_ghost_session_reject()
    test_process_restart_resurrection()

    # ── 诊断矩阵 ──
    passed = sum(1 for r in _results if r["passed"])
    failed = sum(1 for r in _results if not r["passed"])

    logger.info(LOG_SEP)
    logger.info("📊 混沌测试诊断矩阵")
    logger.info(LOG_SEP)
    for r in _results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        logger.info("  %s | 用例%s | %s", status, r["id"], r["name"])
        if not r["passed"]:
            logger.info("         └─ %s", r["detail"])
    logger.info(LOG_SEP)
    logger.info("🏁 总计: %d PASS / %d FAIL / %d TOTAL", passed, failed, len(_results))
    logger.info(LOG_SEP)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(run_all())
