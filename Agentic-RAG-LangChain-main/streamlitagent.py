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
from langchain_core.messages import HumanMessage
from langchain.agents import initialize_agent

# Load environment variables
load_dotenv()


MISTRAL_API_KEY = os.environ["mistral_api_key"] ="Wxn2lCBcugXAz0GK265wJR4H6jYRrNl3"
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"] ="pcsk_6L8ZmY_fh3AkwzUNVg87rneaKU2HgoApGK91wJJoVoYtX4HSdnKdxaWpuktSyjkApsBeZ"
TAVILY_API_KEY=os.environ["TAVILY_API_KEY"]="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"


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
tavily = TavilySearchResults(max_results=10, tavily_api_key=TAVILY_API_KEY)
def tavily_wrapper(query):
    print(f"🔍 Tavily invoked for: {query}")
    return tavily.run(query)
tools = [
    Tool(name="Pinecone Document Store", func=qa_db.run, description="Use it to lookup information from Pinecone"),
    Tool(name="Tavily", func=tavily_wrapper, description="Use this to lookup information from Tavily")
]

agent = initialize_agent(tools=tools, llm=llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True, verbose=True, memory=conversational_memory)

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
            print("Agent input keys:", getattr(agent, "input_keys", "No input keys found"))
            response = agent.invoke({"input": query})
            if "Tavily" in response["output"]:
                print("\n🔹 Tavily Search was used!\n")
                search_results = tavily.run(query)
                extracted_urls = [item["url"] for item in search_results]
                
                print(f"Query: {query}")
                print(f"Response: {response}")
                print(f"Extracted URLs: {extracted_urls}")
                st.success(response["output"])
                st.success(f"Tavily url: {extracted_urls}")
            else:
                print("\n🔸 Tavily was NOT used.\n")
                
                # Force Tavily as a backup if Pinecone gives a bad response
                backup_response = tavily.run(query)
                if backup_response:
                    extracted_urls = [item["url"] for item in backup_response]
                    print(f"🔁 Forced fallback to Tavily: {extracted_urls}")
                    st.write("### Answer (from Tavily):")
                    st.write("### Answer:")
                    st.success(response["output"])
                    st.success(f"Tavily url: {extracted_urls}")

# Conversation History
if st.button("Clear Memory"):
    agent.memory.clear()
    st.success("Memory Cleared!")
