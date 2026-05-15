"""统一日志配置：各入口（uvicorn / streamlit / python main_graph.py）启动时调用 configure_logging()。"""

import logging
import os
import sys


def configure_logging() -> None:
    """
    配置根 logger 与控制台输出。
    环境变量 LOG_LEVEL：DEBUG / INFO / WARNING / ERROR（默认 INFO）。
    """
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(fmt)
        root.addHandler(handler)
    else:
        for h in root.handlers:
            h.setFormatter(fmt)

    # 第三方库默认别太吵（需要时可再调）
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
