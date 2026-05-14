# my_iot_agent
• 项目背景：针对物联网设备运维资料冗余、故障诊断难的问题，独立设计并开发了一款具备跨会话记忆与混合检索能力的智能诊断 Agent，实现设备排障自动化。 

• 技术架构：采用 LangGraph 构建 ReAct 循环架构，后端使用 FastAPI 封装标准异步 API，前端基于 Streamlit 构建轻量级交互界面，整体服务基于 Docker 实现容器化部署。 

• 核心功能与工程亮点： 

o [智能体架构] Runtime Auditor 与自我纠错 突破单向链式调用限制，基于 LangGraph 设计多节点循环架构，创新性引入“运行时审计 (Auditor)”与“强制重写 (Rewriter)”机制。通过后置校验有效拦截 LLM 幻觉，确保输出的维修方案 100% 具备可追溯的参考来源 (Grounding)。 

o [混合知识检索] 结构化与非结构化数据融合 封装 ChromaDB 实现维修手册的向量语义检索（基于 sentence-transformers）；结合 SQLite 实现设备运行状态的 SQL 关联查询。通过 Function Calling 动态路由，实现业务数据与外部知识的精准匹配。 

o [生产级工程落地] 高可用后端与规范化部署 
 ▪ 接口规范与防御性编程：基于 Pydantic 建立严格的数据契约 (Data Contract)；引入 try-except 全局异常捕获与 pythondotenv 环境变量管理，确保大模型接口超时情况下的服务鲁棒性。 
 ▪ 全链路可观测性：深度接入 LangSmith 实现 Agent 内部节点调用的全链路追踪 (Tracing)，并构建基于 logging 模块的系统运行日志，实现从网络请求到 Agent 思考链路的快速 Bug 溯源。 
 ▪ 容器化敏捷交付：编写高可用 Dockerfile，并使用 Docker Compose 统筹 Agent 后端服务、向量数据库与前端 UI 组件，实现环境隔离与跨平台的一键式部署。 
 
o [质量评测体系] 自动化反馈闭环 自主构建 Golden Dataset (基准测试集)，搭建 Agent 自动化评测流水线 (Evaluation Pipeline)。针对检索准确率 (Retrieval) 与回答事实性 (Groundedness) 进行量化打分，驱动 Prompt 与工作流的持续迭代优化。
