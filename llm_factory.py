import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

def create_deepseek_brain():
    load_dotenv()
    # 逻辑：生产一个严谨的 (temp=0) 的模型实例
    model = ChatOpenAI(
        model='deepseek-chat',
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base='https://api.deepseek.com',
        temperature=0
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