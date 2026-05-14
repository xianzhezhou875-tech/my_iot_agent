import chromadb
from chromadb.utils import embedding_functions
from langchain_core.tools import tool

# 1. 逻辑：定义翻译官和仓库地址
model_name = "paraphrase-multilingual-MiniLM-L12-v2"
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
client = chromadb.PersistentClient(path="./my_rag_db")

# 2. 逻辑：初始化并存入一条样本知识
def init_rag():
    collection = client.get_or_create_collection(name="manual", embedding_function=ef)
    # 模拟维修知识
    collection.upsert(
        documents=["把电源关闭就可以修好风扇了。"],
        metadatas=[{"source": "传感器维修手册_v1.pdf"}], # 确保这里存了！
        ids=["doc_002"],
    )
    print(f"DEBUG: RAG 初始化完成")
@tool
def query_repair_manual_tool(question: str):
    """当用户询问设备如何修理、故障原因或技术原理时，调用此工具。"""
    collection = client.get_or_create_collection(name="manual", embedding_function=ef)

    # 执行查询，记得把 metadatas 也查出来
    results = collection.query(query_texts=[question], n_results=1, include=["documents", "metadatas", "distances"])

    # 检查是否有结果
    if results['distances'] and results['distances'][0] and results['distances'][0][0] < 0.6:
        content = results['documents'][0][0]
        metadata = results['metadatas'][0][0] # 获取元数据
        
        # 组装一个包含来源的字符串
        source_name = metadata.get("source", "未知手册") # 假设你存的时候 key 是 source
        return f"【资料来源: {source_name}】\n内容: {content}"

    return "知识库中未找到相关手册内容。"