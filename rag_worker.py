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
        documents=["传感器 A 的正常范围是 20-30。"],
        ids=["doc_002"],
    )

@tool
def query_repair_manual_tool(question: str):
    """当用户询问设备如何修理、故障原因或技术原理时，调用此工具。"""
    # 1. 确保 collection 已获取
    collection = client.get_or_create_collection(name="manual", embedding_function=ef)

    print(f"DEBUG: 知识库当前条目数: {collection.count()}") # 看看这里输出是多少？
    # 2. 检查是否有数据
    if collection.count() == 0:
        return "知识库当前为空，请先运行初始化程序。"

    # 3. 执行查询
    results = collection.query(query_texts=[question], n_results=1)

    # 4. 判断并返回（必须要有 return）
    if results['distances'] and results['distances'][0][0] < 0.6:
        return results['documents'][0][0]

    return "知识库中未找到相关手册内容。"