import streamlit as st
import os
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, Tool, AgentType
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# API Keys
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

index_name = "agenticragmodel"
embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name=index_name, namespace="main", embedding=embed)

# LLM setup
llm = ChatMistralAI(temperature=0.0, model_name="mistral-small", max_tokens=512)

# Conversational Memory
conversational_memory = ConversationBufferWindowMemory(memory_key='chat_history', k=5, return_messages=True)

# Retrieval QA Chain
qa_db = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

# Prompt Template
prompt = hub.pull("hwchase17/react")

# Tools
search_tool = TavilySearchResults(max_results=10, tavily_api_key=TAVILY_API_KEY)
tools = [
    Tool(name="Pinecone Document Store", func=qa_db.run, description="Use it to lookup information from Pinecone"),
    Tool(name="Tavily", func=search_tool.run, description="Use this to lookup information from Tavily")
]

agent = AgentExecutor(tools=tools, agent=llm, handle_parsing_errors=True, verbose=True, memory=conversational_memory)

# Streamlit UI
st.set_page_config(page_title="AI Q&A System", layout="centered")

st.markdown("""
    <style>
        body { font-family: Arial, sans-serif; background-color: #f4f4f4; }
        .stTextInput, .stButton > button { border-radius: 8px; }
        .stTextInput > div > div > input { padding: 10px; }
        .stButton > button { background-color: #4CAF50; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("📚 AI-Powered Q&A System")
st.markdown("Ask questions based on stored knowledge and live search.")

# User Input
query = st.text_input("Enter your question:")
if st.button("Ask"):
    if query:
        with st.spinner("Fetching answer..."):
            response = agent.invoke({"input": query})
            st.write("### Answer:")
            st.success(response)
    else:
        st.warning("Please enter a question.")

# Conversation History
if st.button("Clear Memory"):
    agent.memory.clear()
    st.success("Memory Cleared!")
