import logging
import sqlite3
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def init_db():
    conn = sqlite3.connect("my_ai_data.db")
    cursor = conn.cursor()
    # 逻辑：连连看的地基
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
    # 逻辑：为了防止空库跑不通，顺手塞点测试数据（这里你可以用你之前的 INSERT 逻辑）
    cursor.execute("INSERT OR IGNORE INTO Owners (id, name) VALUES (1, '小明')")
    cursor.execute("INSERT OR IGNORE INTO Devices (owner_id, device_name) VALUES (1, '风扇')")

    conn.commit()
    conn.close()
    logger.info("SQLite 已就绪: my_ai_data.db")


@tool
def query_user_device_tool(name: str):
    """当需要查询特定用户名下拥有的 IoT 设备名称时，调用此工具。"""
    logger.debug("query_user_device_tool: name=%r", name)
    conn = sqlite3.connect("my_ai_data.db")
    cursor = conn.cursor()

    query = """
            SELECT Owners.name, Devices.device_name
            FROM Owners
                     JOIN Devices ON Owners.id = Devices.owner_id
            WHERE Owners.name = ? \
            """
    cursor.execute(query, (name,))
    result = cursor.fetchone()
    conn.close()

    if result:
        return f"{result[0]} 拥有的设备是 {result[1]}"
    return "未找到相关设备信息"