import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from main_graph import app

st.title("IoT 智能运维 Agent")

if "lc_messages" not in st.session_state:
    st.session_state.lc_messages = []


def _render_messages(messages: list) -> None:
    """只展示用户与助手最终文本，跳过 ToolMessage 与带 tool_calls 的中间 AIMessage。"""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                continue
            content = msg.content
            text = content if isinstance(content, str) else str(content)
            if text.strip():
                with st.chat_message("assistant"):
                    st.markdown(text)


_render_messages(st.session_state.lc_messages)

if prompt := st.chat_input("小明，想聊点什么？"):
    st.session_state.lc_messages.append(HumanMessage(content=prompt))
    with st.spinner("思考中…"):
        result = app.invoke({"messages": st.session_state.lc_messages})
    st.session_state.lc_messages = result["messages"]
    st.rerun()
