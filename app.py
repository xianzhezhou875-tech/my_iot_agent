from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage

from main_graph import app as agent_graph
from database_worker import init_db
from rag_worker import init_rag

app = FastAPI()

init_db()
init_rag()


class ChatRequest(BaseModel):
    user_input: str
    session_id: str = "default"


def _last_assistant_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            c = msg.content
            return c if isinstance(c, str) else str(c)
    last = messages[-1]
    c = getattr(last, "content", None)
    if c is None:
        return str(last)
    return c if isinstance(c, str) else str(c)


@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    # session_id 预留多轮；当前 Agent 为单轮 messages，仅使用本轮 user_input
    try:
        input_state = {"messages": [HumanMessage(content=request.user_input)]}
        result = agent_graph.invoke(input_state)
        final_reply = _last_assistant_text(result["messages"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {"reply": final_reply}