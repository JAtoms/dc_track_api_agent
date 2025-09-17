import streamlit as st
from dotenv import load_dotenv
from agent import sunbird_agent

load_dotenv()

st.title("Sunbird DC Track API Doc Assistant")

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []


for msg in st.session_state["chat_messages"]:
    st.chat_message(msg["role"]).write(msg["content"])


def submit_query():
    user_input = st.session_state.get("input", "")
    if user_input:
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})
        with st.spinner("AI is thinking..."):
            messages = {"messages": st.session_state["chat_messages"]}
            response = sunbird_agent.invoke(messages)
            ai_response = response["messages"][-1]
            st.session_state["chat_messages"].append({"role": "assistant", "content": ai_response.content})
        st.session_state["input"] = ""
        st.rerun()


user_input = st.chat_input("Enter your query:", key="input", on_submit=submit_query)
