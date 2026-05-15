import logging
import os
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# 向量库目录固定在「本文件所在项目根」，避免因工作目录不同写到别处
_RAG_DIR = Path(__file__).resolve().parent / "my_rag_db"
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
# 距离阈值：不同模型/度量下需自行调参，可用环境变量覆盖
_MAX_DISTANCE = float(os.environ.get("RAG_MAX_DISTANCE", "0.6"))


@lru_cache(maxsize=1)
def _get_embedding_function():
    """首次需要时再加载 SentenceTransformer，避免 import 即拉模型。"""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=_MODEL_NAME
    )


@lru_cache(maxsize=1)
def _get_client() -> chromadb.PersistentClient:
    _RAG_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(_RAG_DIR))


def init_rag():
    collection = _get_client().get_or_create_collection(
        name="manual",
        embedding_function=_get_embedding_function(),
    )
    collection.upsert(
        documents=["传感器 A 的正常范围是 20-30。"],
        ids=["doc_002"],
    )
    logger.info("RAG 集合 manual 已 upsert 示例文档 doc_002")


@tool
def query_repair_manual_tool(question: str):
    """当用户询问设备如何修理、故障原因或技术原理时，调用此工具。"""
    collection = _get_client().get_or_create_collection(
        name="manual",
        embedding_function=_get_embedding_function(),
    )

    n = collection.count()
    logger.debug("RAG collection count: %s", n)
    if n == 0:
        return "知识库当前为空，请先运行初始化程序。"

    results = collection.query(query_texts=[question], n_results=1)

    if results["distances"]:
        dist = results["distances"][0][0]
        logger.debug("RAG query distance=%s threshold=%s", dist, _MAX_DISTANCE)
    if results["distances"] and results["distances"][0][0] < _MAX_DISTANCE:
        return results["documents"][0][0]

    return "知识库中未找到相关手册内容。"
