from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

st.set_page_config(page_title="SEO Expert AI Assistant", page_icon="🚀")
st.header("🚀 SEO Expert AI Assistant")


try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="openai/gpt-oss-120b", api_key=groq_api_key)


with st.sidebar:
    st.title("Settings")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


chat_template = ChatPromptTemplate.from_messages([
    ("system", """You are a senior SEO Expert. 
    STRICT RULE: You must ONLY answer questions related to Search Engine Optimization (SEO), 
    digital marketing, keyword research, backlinks, and website ranking.
    If the user asks about ANY other topic, politely refuse and say: 
    "I am an SEO Expert assistant and can only help with SEO-related queries."
    Do NOT answer off-topic questions under any circumstances."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{Question}")
])


for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)


user_input = st.chat_input("Ask anything you want about SEO...")

if user_input:
    
    with st.chat_message("user"):
        st.markdown(user_input)

    
    messages = chat_template.invoke({
        "history": st.session_state.chat_history,
        "Question": user_input
    })

    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing SEO query..."):
            result = model.invoke(messages)
            st.markdown(result.content)


    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=result.content))