"""
全局日志工厂 — 整个项目统一入口。
用法：
    from logging_config import logger          # 顶层模块级通用 logger
    from logging_config import get_logger       # 子模块获取命名 logger
    from logging_config import configure_logging # 入口显式调用一次
"""

import logging
import logging.handlers
import os
import sys

# ── 常量 ──────────────────────────────────────────────
LOG_DIR = r"D:/my_agent_logs"
LOG_FILENAME = "runtime.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILENAME)
BACKUP_COUNT = 30  # 保留最近 30 个轮转切片


def _ensure_log_dir() -> None:
    """静默创建日志目录，不存在则递归创建（防御性）。"""
    os.makedirs(LOG_DIR, exist_ok=True)


# ── 占位符格式化器设计 ────────────────────────────────
#
#   %(asctime)s        → 2026-06-04 14:32:01,234   时间戳（精确到毫秒）
#   %(levelname)-8s    → "INFO    "                 级别，左对齐，固定 8 字符宽度
#   %(name)s           → "app.routes.agent"          logger 层级名，一眼定位模块
#   %(filename)s       → "agent_router.py"          源文件名
#   %(lineno)d         → 142                        行号
#   %(funcName)s       → "invoke_agent"             函数名
#   %(message)s        → 用户写的日志正文
#
#   【设计思路】：
#   - 文件日志 = 法医级全量字段（毫秒时间、文件行号、函数名），出问题时 grep 秒定位。
#   - 控制台日志 = 精简版，去掉文件名/行号/函数名，避免刷屏干扰日常开发。

_CONSOLE_FMT = logging.Formatter(
    fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_FILE_FMT = logging.Formatter(
    fmt="%(asctime)s [%(levelname)-8s] %(name)s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ── 模块级通用 logger ─────────────────────────────────
# 外部直接 `from logging_config import logger` 即可使用，无需重复创建。
_configured = False
logger = logging.getLogger("iot_agent")


def configure_logging() -> None:
    """
    配置根 logger + 文件轮转 + 控制台。
    入口（FastAPI / Streamlit / main_graph.py）在启动时各调用一次，幂等安全。

    环境变量 LOG_LEVEL: DEBUG / INFO / WARNING / ERROR（默认 INFO）。
    """
    global _configured
    if _configured:
        return

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    # 1. 保证日志目录存在
    _ensure_log_dir()

    # 2. 构建根 logger
    root = logging.getLogger()
    root.setLevel(level)

    # ── 3. 文件 Handler：TimedRotatingFileHandler ─────
    #    when="D"             → 每天午夜轮转
    #    interval=1           → 每 1 天切一次
    #    backupCount=30       → 只保留最近 30 个 .log 切片
    #    encoding="utf-8"     → 避免 Windows 中文乱码
    #    delay=True            → 延迟创建文件，不写日志时不占句柄
    #
    #    轮转后的文件命名：runtime.log.2026-06-03
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=LOG_PATH,
        when="D",
        interval=1,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setLevel(logging.DEBUG)          # 文件存全量日志
    file_handler.setFormatter(_FILE_FMT)
    root.addHandler(file_handler)

    # ── 4. 控制台 Handler ────────────────────────────
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)               # 控制台跟环境变量走
    console_handler.setFormatter(_CONSOLE_FMT)
    root.addHandler(console_handler)

    # ── 5. 遏制第三方库日志噪音 ─────────────────────
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    # 6. 标记已配置，保证幂等
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    按模块名获取子 logger，继承根配置。
    用法：logger = get_logger(__name__)
    """
    if not _configured:
        configure_logging()
    return logging.getLogger(name)
