# CLAUDE.md — IoT 智能运维 Agent 系统 · 最高开发宪法

## 项目核心愿景

构建一个基于 **LangGraph 多智能体编排** 的 IoT 智能运维 Agent 系统：

- Supervisor 经理节点 → 按用户意图分诊（Dispatch `/dɪˈspætʃ/`）
- Device Agent → SQLite 设备归属查询
- Manual Agent → RAG 向量知识库维修手册检索
- Auditor → 审计质检（Anti-Hallucination `/həˌluːsɪˈneɪʃən/`）
- Rewriter → 审计失败后强制基于资料重写
- 对外暴露 FastAPI `/chat` 端点，`app.py` 统一接入

## 技术栈

| 层 | 技术 |
|---|---|
| LLM | DeepSeek Chat (via `langchain-openai` ChatOpenAI 兼容接口) |
| 编排 | LangGraph `StateGraph` + `Command` 路由 |
| 向量库 | ChromaDB `PersistentClient` + `SentenceTransformer` embedding |
| Web | FastAPI + tenacity 指数退避重试 |
| 数据库 | SQLite (`database_worker.py`) |
| 日志 | Python `logging` → `TimedRotatingFileHandler`，持久化 D 盘 |
| 可观测 | LangSmith (可选) |

## 项目文件结构

```
MY_AGENT/
├── app.py                # FastAPI 入口，全局异常拦截器，tenacity 重试
├── main_graph.py         # LangGraph 图定义，节点编排 & 审计死循环修复
├── rag_worker.py         # RAG 高级检索器（查询改写/混合检索/Top-K 重排）
├── database_worker.py    # SQLite 设备归属查询工具
├── llm_factory.py        # DeepSeek 模型工厂 & 4 段 Prompt 模板
├── logging_config.py     # 全局日志工厂（TimedRotatingFileHandler + 控制台）
├── ui.py                 # Streamlit 前端（已剥离）
├── run_eval.py           # 评估脚本
├── eval_data.json        # 评估数据集
├── Dockerfile            # Docker 镜像
├── docker-compose.yml    # Docker 编排
├── requirements.txt      # Python 依赖
├── my_rag_db/            # ChromaDB 持久化向量库目录
└── D:/my_agent_logs/     # 【外部】日志轮转持久化目录（TimedRotatingFileHandler）
```

## 工程规范 · 不可触碰红线

### 1. 禁止 print

```python
# ❌ 绝对禁止
print("debug info")

# ✅ 必须使用全局 logger
from logging_config import logger
logger.info("RAG 命中 — distance=%.4f", best_distance)
logger.debug("混合检索完成 — 召回=%d", len(merged))
logger.exception("ChromaDB 初始化失败")  # 自动附带堆栈
```

### 2. 日志持久化

- 所有 `.log` 文件写入 `D:/my_agent_logs/`
- `TimedRotatingFileHandler`：每天午夜轮转，保留最近 30 个切片
- 文件日志 = 法医级（`%(filename)s:%(lineno)d | %(funcName)s()` 全部字段）
- 控制台日志 = 精简版（只保留时间/级别/模块名/消息）
- 第三方库噪音已遏制（httpx/httpcore/chromadb/uvicorn.access → WARNING）

### 3. 防御性编程

```python
# 所有 I/O 边界必须包裹 try/except
# LLM 调用 → _safe_llm_invoke() 统一包装
# ChromaDB → try/except + 降级返回
# SQLite → try/except/finally 防连接泄漏（Connection Leak /kəˈnekʃən liːk/）
```

### 4. 只重构，不重写

- 基于现有类名、函数名、业务逻辑做增量重构
- 保持外部接口签名兼容
- 修改后代码必须能无缝贴回原项目

### 5. 术语拆解与答复风格

- 核心专业名词附带 **【英式音标】**（IPA）
- 用最直白的大白话解释为什么这么改
- 采用 Code Reviewer 引导模式（Pedagogical `/ˌpedəˈɡɒdʒɪkəl/` Approach）
  - 先指出在哪个文件、哪个函数、第几行
  - 拆解修改逻辑，而非直接贴整段代码
  - 推荐修改方案 → 等用户确认 → 再执行

## 对话风格偏好

- 极客范儿，简洁专业
- 修改前先读取代码，确认现状再动刀
- 重构后给出汇总报告（改动总览表）
- 优先使用 `TaskCreate` 跟踪复杂多步骤任务

## Git 提交规范

遵循 Angular Convention：
```
feat(scope): 说明
fix(scope): 说明
refactor(scope): 说明
chore(scope): 说明
```

## 关键上下文（本会话建立）

- 代理端口已修正为 `127.0.0.1:7895`（持久化到全局 git config）
- 对话数据已从 `C:\Users\Addlove\.claude\` 迁移至 `D:\CLAUDE_MEMORY\`
- 环境变量 `CLAUDE_CODE_CONFIG_DIR=D:\CLAUDE_MEMORY`（用户级注册表）
- RAG 三大组件流水线：`_rewrite_query()` → `_hybrid_search()` → `_rerank_by_distance()`
- LangGraph 死循环防护：`MAX_REWRITE_ATTEMPTS=2` + `recursion_limit=30`
- `_safe_llm_invoke()` 统一异常包装器覆盖所有 LLM 调用节点

---

> **宪法版本**: v1.0 · **生效日期**: 2026-06-04 · **最后重构 commit**: `d042a0e`
