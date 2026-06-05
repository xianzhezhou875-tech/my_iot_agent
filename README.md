# 🤖 IoT RAG Agent Assistant (基于 LangGraph + FastAPI 的全栈工业级 HITL 智能体系统)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Framework LangGraph](https://img.shields.io/badge/Framework-LangGraph_2.0-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Backend FastAPI](https://img.shields.io/badge/Backend-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![License MIT](https://img.shields.io/badge/license-MIT-red.svg)](./LICENSE)

这是一个专门针对物联网（IoT）复杂业务场景设计的、具备高防御性与准生产级交付水准的 **Advanced RAG 智能体助理系统**。系统底层依托 **LangGraph 2.0** 状态机拓扑构建，彻底解决了传统线性 Chain 结构在面对高危动作执行、高并发访问、状态长效持久化以及长文本幻觉时的工程痛点。

---

## 🏗️ 系统核心架构拓扑

本系统在研发过程中深度践行**“网络感知层与业务状态机彻底解耦”**的原则，整体拓扑架构如下：



### 1. 状态机级 HITL 门禁网络 (Human-in-the-Loop)
系统依托 LangGraph 内置的刚性 `interrupt()` 语法，在业务链路上架设了 3 处完全独立的防御卡口：
* **HITL-1 (路由分流确认门)**：对主管 Agent（Supervisor）的分流决策进行安全审计。
* **HITL-2 (高危动作执行门)**：针对下发至具体 IoT 设备的控制指令进行人工二次授权。
* **HITL-3 (知识检索内容门)**：对长文本检索召回的核心金块进行合规性阻断审查。

### 2. 双端全异步解冻协议 (Stateless To Stateful Contract)
* **网络感应拦截**：FastAPI 路由层在 `/chat` 接口内部异步捕捉底层抛出的 `GraphInterrupt` 异常，秒级释放 HTTP 请求线程，杜绝高并发网关 I/O 阻塞。
* **无损唤醒反哺**：独立开辟 `/api/approve` 二次握手解冻路由，前端复用同一条 `session_id` 作为线程指针钥匙，利用 `Command(resume=...)` 信号注入器精准注入干预信号，打通全异步审批流。

---

## 💾 持久化底座与并发防御 (Resilient Storage)

为了保障挂起现场的绝对灾备安全，系统全面废弃了脆皮的内存缓存（MemorySaver），重构升级为 **`SqliteSaver` 物理快照城堡**，并注入了三层并发防御机制：

* **`journal_mode=WAL` (预写日志模式)**：强行打通物理读写分离，确保读不掉线、写不互斥。
* **`busy_timeout=5000ms` (写锁排队机制)**：全面防御多用户、多线程高并发抢占操作同一个 `.db` 文件时的 `Database is locked` 毁灭性崩溃。
* **`check_same_thread=False`**：彻底封杀了 Python 在多线程异步环境下操作 SQLite 的跨线程安全隐患。
* **跨进程原位复活**：快照数据刚性写入 `D:\my_agent_checkpoints\checkpoint.db`，即使后端服务器突发断电、进程遭 `taskkill` 强杀，重启后依然能完美捞出中断现场，实现“断电免密原地复活”。

---

## 🔬 检索大脑：Advanced RAG 铁三角流水线

针对传统 RAG 极易发生的“型号幻觉（如模糊 V1/V2 混淆）”与长文本“迷失在中间（Lost in the Middle）”效应，本系统手写重构了原生函数级的检索增强流水线：

```text
用户提问 (Query)
   │
   ├──> [Dense 路] ChromaDB 密集向量检索 ───> 捕捉模糊语义意图
   └──> [Sparse 路] BM25Okapi 稀疏关键词 ──> 精确匹配强特征(如设备型号)
   │
   ▼
[RFF 倒数排序融合] ──> 公式: 1 / (60 + Rank) 抹平不同量纲分数 ──> 免密去重提权
   │
   ▼
[BGE-Reranker v2-m3] ──> 懒加载常驻内存 + 交叉注意力 Cross-Attention 精排
   │
   ▼
[Sigmoid 阈值熔断] ──> 映射置信度区间至 0-1 ──> 剔除脏数据 ──> 截断 Top-3 精华灌入 LLM
