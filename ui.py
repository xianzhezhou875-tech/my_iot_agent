import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import requests

st.title("IoT 智能运维 Agent")

API_URL = "http://127.0.0.1:8000/chat"

if "lc_messages" not in st.session_state:
    st.session_state.lc_messages = []


def _render_messages(messages: list) -> None:
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
                    # --- 这里是改动部分 ---
                    if "【资料来源:" in text:
                        # 尝试把内容和来源分开
                        try:
                            # 按照第一个【来源标记进行拆分
                            main_content, source_part = text.split("【资料来源:", 1)
                            st.markdown(main_content)
                            
                            # 放入折叠块中
                            with st.expander("查看参考维修手册"):
                                st.info("【资料来源:" + source_part)
                        except:
                            # 如果拆分失败，就当普通文本显示
                            st.markdown(text)
                    else:
                        st.markdown(text)


_render_messages(st.session_state.lc_messages)

if prompt := st.chat_input("小明，想聊点什么？"):
    st.session_state.lc_messages.append(HumanMessage(content=prompt))
    try:
        with st.spinner("思考中…"):
            response = requests.post(
                API_URL,
                json={"user_input": prompt},
                timeout=120,
            )
            response.raise_for_status()
            reply = response.json().get("reply", "")
    except requests.RequestException as e:
        st.error(f"请求后端失败（请先启动 uvicorn）：{e}")
        st.session_state.lc_messages.pop()
        st.stop()
     
    text = reply if isinstance(reply, str) else str(reply)
    st.session_state.lc_messages.append(AIMessage(content=text))
    st.rerun()
    # ui.py 中
# ... 
        # 获取回复
    reply_text = response.json()["reply"]
        
        # --- 加上这一行，看看后台到底收到了什么 ---
    print(f"DEBUG: Agent 返回的内容是: {reply_text}")
        
        # ...