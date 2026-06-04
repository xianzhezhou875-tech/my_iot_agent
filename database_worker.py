"""
数据库工作器 — SQLite 设备归属查询。

对外暴露：
  - init_db()                → 建表 & 插入示例数据（幂等）
  - query_user_device_tool   → LangChain Tool，供 Device Agent 调用
"""

import sqlite3
from pathlib import Path

from langchain_core.tools import tool

from logging_config import logger

# 数据库文件固定在项目根目录，避免因 os.getcwd() 变化而找不到文件
_DB_DIR = Path(__file__).resolve().parent
_DB_PATH = str(_DB_DIR / "my_ai_data.db")


def _get_conn() -> sqlite3.Connection:
    """创建数据库连接（调用方负责关闭）。"""
    return sqlite3.connect(_DB_PATH)


def init_db() -> None:
    """
    初始化数据库：建表 + 插入示例数据。
    幂等安全 — 表已存在则跳过 CREATE，数据已存在则 IGNORE。
    """
    conn = None
    try:
        conn = _get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Owners (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Devices (
                owner_id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_name TEXT
            )
        """)

        cursor.execute(
            "INSERT OR IGNORE INTO Owners (id, name) VALUES (1, '小明')"
        )
        cursor.execute(
            "INSERT OR IGNORE INTO Devices (owner_id, device_name) VALUES (1, '风扇')"
        )

        conn.commit()
        logger.info("SQLite 数据库已就绪: %s", _DB_PATH)

    except sqlite3.Error:
        logger.exception("SQLite 初始化失败，路径: %s", _DB_PATH)
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        raise
    finally:
        if conn:
            conn.close()


@tool
def query_user_device_tool(name: str) -> str:
    """
    当需要查询特定用户名下拥有的 IoT 设备名称时，调用此工具。

    Args:
        name: 用户名（如 "小明"）

    Returns:
        设备归属信息，或错误提示。
    """
    logger.debug("query_user_device_tool 被调用: name=%r", name)

    conn = None
    try:
        conn = _get_conn()
        cursor = conn.cursor()

        query = """
            SELECT Owners.name, Devices.device_name
            FROM Owners
            JOIN Devices ON Owners.id = Devices.owner_id
            WHERE Owners.name = ?
        """
        cursor.execute(query, (name,))
        result = cursor.fetchone()

        if result:
            reply = f"{result[0]} 拥有的设备是 {result[1]}"
            logger.info("设备查询命中 — %s", reply)
            return reply

        logger.info("设备查询无结果 — name=%r", name)
        return f"未找到 {name} 的相关设备信息"

    except sqlite3.Error:
        logger.exception("设备查询 SQL 异常 — name=%r", name)
        return "数据库查询暂时不可用，请稍后重试。"
    finally:
        if conn:
            conn.close()
