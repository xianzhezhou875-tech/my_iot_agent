import logging
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# 尽早加载 .env，保证 LangSmith / DeepSeek 等变量在首次读 os.environ 前可用
load_dotenv()
if os.getenv("LANGSMITH_TRACING", "").lower() in ("true", "1"):
    _proj = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "default"
    logger.info("LangSmith 追踪已开启，项目名: %s", _proj)


def create_deepseek_brain():
    load_dotenv()
    if not os.getenv("DEEPSEEK_API_KEY"):
        logger.warning("未设置环境变量 DEEPSEEK_API_KEY，模型请求可能失败")
    model = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base="https://api.deepseek.com",
        temperature=0,
        timeout=120.0,
        max_retries=0,
    )
    return model


# 逻辑：定义质检员的思维方式
AUDITOR_PROMPT = """
你是一个严谨的 AI 审计员。你的任务是核对【AI回答】是否完全基于【参考资料】。
判断标准：如果 AI 回答中包含了资料里没有提到的事实，或者与资料相反，则判断为 FAIL。

【参考资料】：{rag_info}
【AI回答】：{final_answer}

请仅输出一个词：PASS 或 FAIL。不要说任何废话。
"""
