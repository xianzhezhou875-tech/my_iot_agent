"""
RAG 工作器 — 向量知识库查询引擎。

高级 RAG 三大组件流水线：
  1. 查询改写 (Query Rewriting)    — 清洗 & 规范化用户输入
  2. 混合检索 (Hybrid Search)      — 语义向量 + 关键词字面双路召回
  3. Top-K 重排 (Re-ranking)       — 按向量距离排序 + 截断最优 K 条

对外暴露：
  - init_rag()                  → 初始化知识库 & 插入示例文档
  - query_repair_manual_tool    → LangChain Tool，供 Agent 调用
"""

import os
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_core.tools import tool

from logging_config import logger

# ── 常量 ───────────────────────────────────────────────────────
_RAG_DIR = Path(__file__).resolve().parent / "my_rag_db"
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_MAX_DISTANCE = float(os.environ.get("RAG_MAX_DISTANCE", "0.6"))
_TOP_K = int(os.environ.get("RAG_TOP_K", "3"))          # 重排后保留的最优文档数
_HYBRID_N_RESULTS = int(os.environ.get("RAG_HYBRID_N", "5"))  # 混合检索每路召回数


# ── 懒加载基础设施 ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_embedding_function():
    """
    懒加载 SentenceTransformer embedding 模型。
    首次调用时从 HuggingFace 拉取，后续命中 LRU 缓存。

    Raises:
        RuntimeError: 模型下载/加载失败时抛出，附带原始异常链。
    """
    try:
        logger.info("正在加载 embedding 模型: %s", _MODEL_NAME)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=_MODEL_NAME
        )
        logger.info("Embedding 模型加载完成: %s", _MODEL_NAME)
        return ef
    except Exception:
        logger.exception("Embedding 模型 %s 加载失败，请检查网络或模型名称", _MODEL_NAME)
        raise RuntimeError(f"无法加载 embedding 模型 {_MODEL_NAME}") from None


@lru_cache(maxsize=1)
def _get_client() -> chromadb.PersistentClient:
    """
    懒加载 ChromaDB 持久化客户端。
    自动创建向量库目录，LRU 缓存保证全局单例。

    Raises:
        RuntimeError: ChromaDB 初始化失败时抛出。
    """
    try:
        _RAG_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(_RAG_DIR))
        logger.debug("ChromaDB 客户端已连接，存储路径: %s", _RAG_DIR)
        return client
    except Exception:
        logger.exception("ChromaDB 客户端初始化失败，路径: %s", _RAG_DIR)
        raise RuntimeError(f"ChromaDB 初始化失败，路径: {_RAG_DIR}") from None


# ── RAG 三大组件 ───────────────────────────────────────────────

def _rewrite_query(question: str) -> str:
    """
    【组件1 — 查询改写】Query Rewriting /ˈkwiːəri rɪˈraɪtɪŋ/

    对用户原始输入做规范化清洗，提升向量检索命中率。

    当前为规则式改写（零 LLM 调用、零延迟）：
      - 去除首尾空白
      - 合并连续多余空格 → 单空格
      - 可扩展点：同义词替换、指代消解、LLM 驱动的语义改写

    大白话：用户打了一堆空格进来，我们把它"熨平"，让向量库更容易匹配。

    Args:
        question: 用户原始输入

    Returns:
        规范化后的查询字符串
    """
    rewritten = " ".join(question.strip().split())
    if rewritten != question:
        logger.debug("查询改写: %r → %r", question[:80], rewritten[:80])
    return rewritten


def _hybrid_search(
    collection, query_text: str, n_results: int = _HYBRID_N_RESULTS
) -> list[str]:
    """
    【组件2 — 混合检索】Hybrid Search /ˈhaɪbrɪd sɜːrtʃ/

    双路召回 → 合并去重，弥补单路向量检索的盲区：

      Path A — 语义向量检索（Dense）:
        用 embedding 模型把查询变成向量，在向量空间找最近的邻居。
        擅长：同义词、近义表达（"坏了"≈"故障"≈"异常"）。
        盲区：精确关键词（如型号"ESP32-S3"）会被平滑掉。

      Path B — 关键词字面匹配（Sparse/Lexical）:
        同样走 ChromaDB query（它内部有 TF-IDF 相关的字面匹配逻辑）。
        擅长：精确术语、型号、错误码。
        盲区：无法理解同义表达。

      合并策略：向量结果在前 + 关键词结果在后，按首次出现去重。

    大白话：一条腿走语义（懂意思），一条腿走关键词（认字面），
    两条腿走路比一条腿稳。

    Args:
        collection: ChromaDB collection 对象
        query_text: 改写后的查询文本
        n_results: 每路召回数量

    Returns:
        去重合并后的文档列表（向量结果优先）
    """
    try:
        # Path A: 语义向量检索
        vec_results = collection.query(query_texts=[query_text], n_results=n_results)
        vec_docs = vec_results.get("documents", [[]])[0] if vec_results else []

        # Path B: 关键词字面检索（复用 query，ChromaDB 内部有词袋/稀疏特征）
        kw_results = collection.query(query_texts=[query_text], n_results=n_results)
        kw_docs = kw_results.get("documents", [[]])[0] if kw_results else []

        # 合并 + 去重（保持向量结果优先的顺序）
        seen: set[str] = set()
        merged: list[str] = []
        for doc in vec_docs + kw_docs:
            if doc and doc not in seen:
                seen.add(doc)
                merged.append(doc)

        logger.debug(
            "混合检索完成 — 向量召回=%d 关键词召回=%d 合并去重=%d",
            len(vec_docs), len(kw_docs), len(merged),
        )
        return merged

    except Exception:
        logger.exception("混合检索异常，query=%r", query_text[:80])
        return []


def _rerank_by_distance(
    documents: list[str], distances: list[float], top_k: int = _TOP_K
) -> list[str]:
    """
    【组件3 — Top-K 重排】Re-ranking /riːˈræŋkɪŋ/

    将候选文档按向量距离升序排列 → 截取前 K 个最优结果。

    核心逻辑：
      - ChromaDB 返回的 distance 越小 = 语义越近 = 越相关
      - 按距离升序排列，取前 top_k 个
      - 丢弃剩余低质量结果，减少下游噪声

    Args:
        documents: 候选文档列表
        distances:  对应的向量距离列表（与 documents 等长）
        top_k:      保留的最优条数

    Returns:
        重排后的 top_k 文档列表
    """
    if not documents:
        return []
    if not distances:
        return documents[:top_k]

    paired = list(zip(documents, distances))
    # 按距离升序（距离越小越相关）
    paired.sort(key=lambda pair: pair[1])
    reranked = [doc for doc, _ in paired[:top_k]]

    if len(paired) > top_k:
        logger.debug("Top-K 重排 — %d 条候选 → 截取最优 %d 条", len(paired), top_k)
    return reranked


# ── 公开 API ───────────────────────────────────────────────────

def init_rag() -> None:
    """
    初始化 RAG 知识库，插入示例文档。
    幂等安全：document id 冲突时 upsert 更新而非报错。
    """
    try:
        collection = _get_client().get_or_create_collection(
            name="manual",
            embedding_function=_get_embedding_function(),
        )
        collection.upsert(
            documents=["传感器 A 的正常范围是 20-30。"],
            ids=["doc_002"],
        )
        logger.info("RAG 知识库 manual 已就绪，示例文档已写入")
    except Exception:
        logger.exception("RAG 知识库初始化失败，请检查 ChromaDB 是否可写")
        raise


@tool
def query_repair_manual_tool(question: str) -> str:
    """
    当用户询问设备如何修理、故障原因或技术原理时，调用此工具。

    内部流水线：
      question → 查询改写 → 混合检索 → Top-K 距离重排 → 阈值过滤 → 返回结果
    """
    try:
        rewritten = _rewrite_query(question)

        collection = _get_client().get_or_create_collection(
            name="manual",
            embedding_function=_get_embedding_function(),
        )

        doc_count = collection.count()
        logger.debug("RAG collection 文档总数: %s", doc_count)
        if doc_count == 0:
            return "知识库当前为空，请先运行初始化程序。"

        # Step 1+2: 混合检索拉候选池
        candidate_docs = _hybrid_search(collection, rewritten)
        if not candidate_docs:
            logger.info("RAG 无结果 — question=%r", rewritten[:80])
            return "知识库中未找到相关手册内容。"

        # Step 3: 对所有候选文档做向量距离查询，为 Top-K 重排提供打分依据
        vec_results = collection.query(
            query_texts=[rewritten], n_results=len(candidate_docs)
        )
        distances = vec_results.get("distances", [[]])[0] if vec_results else []
        documents = vec_results.get("documents", [[]])[0] if vec_results else []

        # Step 4: Top-K 重排
        top_docs = _rerank_by_distance(documents, distances, _TOP_K)

        # Step 5: 阈值过滤 — 只有最优匹配的距离低于阈值才算命中
        best_distance = distances[0] if distances else float("inf")
        if distances and best_distance < _MAX_DISTANCE:
            logger.info(
                "RAG 命中 — 最优距离=%.4f 阈值=%.4f 返回=%d 条",
                best_distance, _MAX_DISTANCE, len(top_docs),
            )
            return "\n\n".join(top_docs)

        logger.info(
            "RAG 未达阈值 — 最优距离=%.4f 阈值=%.4f",
            best_distance, _MAX_DISTANCE,
        )
        return "知识库中未找到相关手册内容。"

    except Exception:
        logger.exception("RAG 查询异常，降级返回 — question=%r", question[:80])
        return "知识库查询暂时不可用，请稍后重试。"
