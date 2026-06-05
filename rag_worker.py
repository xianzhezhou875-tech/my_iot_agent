"""
RAG 工作器 — 工业级 Hybrid RAG 铁三角检索引擎。

检索流水线 (5 段式)：
  1. 查询改写 (Query Rewriting)         — 清洗 & 规范化用户输入
  2. BM25 关键词检索 (Sparse/Lexical)   — Okapi BM25 精确字面匹配
  3. ChromaDB 语义向量检索 (Dense)      — SentenceTransformer embedding
  4. RRF 倒数排序融合 (Reciprocal Rank Fusion) — 双路结果无量纲化合并去重
  5. BGE Reranker 交叉注意力重排        — Cross-Attention 精排 → Top-3

对外暴露：
  - init_rag()                  → 初始化知识库 & 插入示例文档
  - query_repair_manual_tool    → LangChain Tool，供 Agent 调用
"""

import os
from functools import lru_cache
from pathlib import Path

import chromadb
import numpy as np
from chromadb.utils import embedding_functions
from langchain_core.tools import tool
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from logging_config import logger

# ── 常量 ───────────────────────────────────────────────────────
_RAG_DIR = Path(__file__).resolve().parent / "my_rag_db"
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # BGE 多语言 Cross-Attention 重排模型
_MAX_DISTANCE = float(os.environ.get("RAG_MAX_DISTANCE", "0.6"))
_TOP_K = int(os.environ.get("RAG_TOP_K", "3"))          # 重排后保留的最优文档数
_HYBRID_N_RESULTS = int(os.environ.get("RAG_HYBRID_N", "5"))  # 混合检索每路召回数
_RRF_K = int(os.environ.get("RAG_RRF_K", "60"))        # RRF 平滑常数


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


def _bm25_search(
    corpus: list[str], query_text: str, n_results: int = _HYBRID_N_RESULTS
) -> list[str]:
    """
    【组件2a — BM25 关键词检索】Sparse / Lexical Retrieval

    基于 Okapi BM25 算法对全量文档做字面级匹配排序。
    擅长：精确型号（ESP32-S3）、错误码（E404）、硬编码特征（V1/V2）。

    Args:
        corpus:      全量文档文本列表
        query_text:  改写后的查询词
        n_results:   返回 Top-N 条

    Returns:
        按 BM25 分数降序排列的文档列表（仅返回 score > 0 的结果）。
    """
    if not corpus:
        return []
    try:
        tokenized_corpus = [doc.split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query_text.split()
        scores = bm25.get_scores(tokenized_query)
        ranked_indices = np.argsort(scores)[::-1]
        ranked = [
            corpus[i] for i in ranked_indices[:n_results] if scores[i] > 0
        ]
        logger.debug("BM25 召回=%d (语料=%d)", len(ranked), len(corpus))
        return ranked
    except Exception:
        logger.exception("BM25 检索异常 — query=%r", query_text[:80])
        return []


def _hybrid_search(
    collection, query_text: str, n_results: int = _HYBRID_N_RESULTS
) -> dict[str, list[str]]:
    """
    【组件2b — 真·双路混合检索】Hybrid Search /ˈhaɪbrɪd sɜːrtʃ/

    Path A — BM25Okapi 关键词字面检索（Sparse/Lexical）
    Path B — ChromaDB 语义向量检索（Dense）

    两路独立并行，各自按得分/距离排序，不在此层合并。
    合并交由下游 RRF 融合器完成。

    Returns:
        {"bm25": [doc_ranked...], "vector": [doc_ranked...]}
    """
    try:
        # Path A: BM25 关键词 — 从 collection 拉全量语料建索引
        all_data = collection.get()
        corpus = all_data.get("documents", []) if all_data else []
        bm25_ranked = _bm25_search(corpus, query_text, n_results)

        # Path B: ChromaDB 语义向量检索
        vec_results = collection.query(query_texts=[query_text], n_results=n_results)
        vec_ranked = vec_results.get("documents", [[]])[0] if vec_results else []
        vec_ranked = [d for d in vec_ranked if d]

        logger.debug(
            "双路检索 — BM25=%d | 向量=%d",
            len(bm25_ranked), len(vec_ranked),
        )
        return {"bm25": bm25_ranked, "vector": vec_ranked}

    except Exception:
        logger.exception("混合检索异常 — query=%r", query_text[:80])
        return {"bm25": [], "vector": []}


def _rrf_fusion(
    ranked_lists: dict[str, list[str]], k: int = _RRF_K
) -> list[str]:
    """
    【组件3 — RRF 倒数排序融合】Reciprocal Rank Fusion
    /rɪˈsɪprəkəl ræŋk ˈfjuːʒən/

    将 BM25 和 Vector 两路排序结果做无量纲化加权合并。

    公式：
      RRF(d) = Σ 1 / (k + rank_i(d))

    其中 k 是平滑常数（默认 60），rank_i 是文档 d 在第 i 路检索中的
    排名（1-indexed）。k 的作用是压制排名末尾文档的噪声权重。

    大白话：
      "传感器正常范围 20-30" 这句话 ——
      BM25 能精确匹配"20-30"的数字面 → 可能排第 1
      Vector 能理解"正常范围"≈"工作区间"的语义 → 可能排第 2
      RRF 综合两路排名：1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
      比单纯只看一路更可靠。

    Args:
        ranked_lists: {"bm25": [doc_ranked...], "vector": [doc_ranked...]}
        k:            平滑常数

    Returns:
        按 RRF 分数降序排列的去重文档列表。
    """
    bm25_docs = ranked_lists.get("bm25", [])
    vec_docs = ranked_lists.get("vector", [])

    if not bm25_docs and not vec_docs:
        return []

    scores: dict[str, float] = {}

    # BM25 路
    for rank, doc in enumerate(bm25_docs, start=1):
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)

    # Vector 路
    for rank, doc in enumerate(vec_docs, start=1):
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)

    # 按 RRF 分数降序
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    logger.debug(
        "RRF 融合 — BM25=%d 向量=%d → 去重=%d 最高分=%.4f",
        len(bm25_docs), len(vec_docs), len(sorted_docs),
        sorted_docs[0][1] if sorted_docs else 0.0,
    )
    return [doc for doc, _ in sorted_docs]


# ── BGE Reranker（懒加载） ──────────────────────────────────────


@lru_cache(maxsize=1)
def _get_reranker_model() -> CrossEncoder:
    """
    懒加载 BGE Reranker Cross-Encoder 模型。

    BAAI/bge-reranker-v2-m3 是多语言 Cross-Attention 重排模型，
    对 (query, document) 对计算交叉注意力得分，比向量距离更精准。

    首次调用从 HuggingFace 拉取（约 1.2GB），后续命中 LRU 缓存。
    """
    try:
        logger.info("正在加载 BGE Reranker 模型: %s", _RERANKER_MODEL)
        model = CrossEncoder(_RERANKER_MODEL, max_length=512)
        logger.info("BGE Reranker 模型加载完成 — max_length=512")
        return model
    except Exception:
        logger.exception("BGE Reranker 模型 %s 加载失败", _RERANKER_MODEL)
        raise RuntimeError(f"无法加载 BGE Reranker 模型 {_RERANKER_MODEL}") from None


def _cross_encoder_rerank(
    query: str, documents: list[str], top_k: int = _TOP_K
) -> list[tuple[str, float]]:
    """
    【组件4 — Cross-Attention 交叉注意力重排】
    Cross-Encoder Re-ranking /krɒs ɪnˈkəʊdə riːˈræŋkɪŋ/

    BGE Reranker 对每个候选 (query, doc) 对独立计算交叉注意力得分，
    精确衡量语义相关性，解决 Lost in the Middle 效应。

    流程：
      1. 构造 (query, doc_i) 对列表
      2. CrossEncoder 批量计算 relevance scores
      3. 按得分降序排列
      4. 截断保留 Top-K 核心 Chunk

    大白话：向量检索看的是"整体方向是不是一致"，
    Cross-Attention 看的是"每个词之间的细粒度交互关系"。
    前者快但粗糙，后者慢但精准——所以先用前者粗筛，再用后者精排。

    Args:
        query:      用户原始/改写后的查询
        documents:  RRF 融合后的候选文档列表
        top_k:      最终保留的最优条数

    Returns:
        [(doc, score), ...] 按得分降序排列的 Top-K 文档-分数对
    """
    if not documents:
        return []

    try:
        reranker = _get_reranker_model()
        # 构造 (query, doc) 对
        pairs = [(query, doc) for doc in documents]
        scores = reranker.predict(pairs, show_progress_bar=False)

        # 归一化到 [0, 1]（sigmoid）
        scores = 1.0 / (1.0 + np.exp(-np.array(scores)))

        # 按得分降序排列
        ranked = sorted(
            zip(documents, scores.tolist()), key=lambda x: x[1], reverse=True
        )
        top = ranked[:top_k]

        logger.debug(
            "CrossEncoder 重排 — 候选=%d → Top-%d | 最高=%.4f 最低=%.4f",
            len(documents), len(top),
            top[0][1] if top else 0.0,
            top[-1][1] if top else 0.0,
        )
        return top
    except Exception:
        logger.exception("CrossEncoder 重排异常 — query=%r", query[:80])
        # 降级：直接截断前 top_k 个
        fallback = [(doc, 0.0) for doc in documents[:top_k]]
        return fallback


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

    Hybrid RAG 铁三角流水线（5 段式）：
      question
        → ① 查询改写 (_rewrite_query)
        → ② BM25 + Vector 双路混合检索 (_hybrid_search)
        → ③ RRF 倒数排序融合 (_rrf_fusion)
        → ④ BGE Reranker Cross-Attention 精排 (_cross_encoder_rerank)
        → ⑤ Reranker 得分阈值过滤 → Top-3 核心 Chunk 返回
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

        # ── Step 1: BM25 + Vector 双路并行召回 ──
        ranked_lists = _hybrid_search(collection, rewritten)
        bm25_count = len(ranked_lists.get("bm25", []))
        vec_count = len(ranked_lists.get("vector", []))
        if bm25_count == 0 and vec_count == 0:
            logger.info("RAG 双路均无结果 — question=%r", rewritten[:80])
            return "知识库中未找到相关手册内容。"

        # ── Step 2: RRF 倒数排序融合 ──
        fused = _rrf_fusion(ranked_lists)
        if not fused:
            logger.info("RAG RRF 融合后为空 — question=%r", rewritten[:80])
            return "知识库中未找到相关手册内容。"

        # ── Step 3: BGE Reranker Cross-Attention 精排 → Top-3 ──
        top_chunks = _cross_encoder_rerank(question, fused, _TOP_K)
        if not top_chunks:
            logger.info("RAG Reranker 无结果 — question=%r", rewritten[:80])
            return "知识库中未找到相关手册内容。"

        # ── Step 4: 阈值过滤 ──
        best_score = top_chunks[0][1]
        if best_score > 0.0:
            logger.info(
                "RAG 命中 — Reranker 最高=%.4f | 返回=%d 条 | 候选池=%d",
                best_score, len(top_chunks), len(fused),
            )
            for i, (doc, score) in enumerate(top_chunks):
                logger.debug(
                    "  Chunk #%d | score=%.4f | preview=%.60s",
                    i + 1, score, doc,
                )
            return "\n\n".join(doc for doc, _ in top_chunks)

        logger.info("RAG 未达 Reranker 阈值 — 最高=%.4f", best_score)
        return "知识库中未找到相关手册内容。"

    except Exception:
        logger.exception("RAG 查询异常，降级返回 — question=%r", question[:80])
        return "知识库查询暂时不可用，请稍后重试。"
