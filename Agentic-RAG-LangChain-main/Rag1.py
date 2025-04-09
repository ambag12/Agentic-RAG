import os
import numpy as np
from uuid import uuid4
from dotenv import load_dotenv

# LANGCHAIN
from langchain_community.llms import GPT4All
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory


from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, Tool
from langchain.agents.react.agent import create_react_agent
from langchain import hub

# Load environmental variables
load_dotenv()

# Configure paths and models
MODEL_NAME = "C:\Agentic-Rag-by-Model\ggml-model-q4_0.gguf"


llm = GPT4All(
    model=MODEL_NAME,
    max_tokens=512,
    backend="llama",  # Changed from "gptj" to "llama" for a .gguf model
    temp=0.0,
    verbose=False
)

# Pinecone configuration
TAVILY_API_KEY=os.environ["TAVILY_API_KEY"]="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"
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"] ="pcsk_6L8ZmY_fh3AkwzUNVg87rneaKU2HgoApGK91wJJoVoYtX4HSdnKdxaWpuktSyjkApsBeZ"
index_name = "agenticragmodel"
embed = embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = PineconeVectorStore(
    index_name=index_name,
    namespace="main",
    embedding=embed
)

# Conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

# Retrieval QA chain
qa_db = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Agent setup
template = '''Answer the following questions as best you can. You have access to the following tools:
{tools}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]. Always look first in Pinecone Document Store
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat 2 times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

# Tavily setup
tavily = TavilySearchResults(max_results=10, tavily_api_key=os.getenv("TAVILY_API_KEY"))

tools = [
    Tool(
        name="Pinecone Document Store",
        func=qa_db.run,
        description="Use it to lookup information from the Pinecone Document Store"
    ),
    Tool(
        name="Tavily",
        func=tavily.run,
        description="Use this to lookup information from Tavily",
    )
]

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    tools=tools,
    agent=agent,
    handle_parsing_errors=True,
    verbose=True,
    memory=conversational_memory
)

# Query execution
queries = [
    "Can you give me one title of a TED talk of Al Gore as main speaker? "
    "Please look in the Pinecone Document Store metadata as it has the title based on the transcripts.",
    
    "Did you find the previous title 'The Case for Optimism on Climate Change' in the Pinecone Document Store?",
    
    "Can you look for a title within the Pinecone Document Store?",
    
    "Is Dan Gilbert a main speaker of TEDx talks? If yes, give me the source of your answer.",
    
    "What is the main topic of Dan Gilbert TEDx talks?"
]

for query in queries:
    response = agent_executor.invoke({"input": query})
    print(f"Query: {query}")
    print(f"Response: {response['output']}\n")
    if "Pinecone Document Store" in response["output"]:
        print("🔹 Pinecone Document Store was used!\n")

    elif "Tavily" in response["output"]:  
        print("🔹 Tavily Search was used!")
        search_results = tavily.run(query)
        extracted_urls = [item["url"] for item in search_results]
        print(f"Extracted URLs: {extracted_urls}\n")
    else:
        print("🔸 No Output :( .\n")
# Memory management
conversational_memory.load_memory_variables({})
agent_executor.memory.clear()
conversational_memory.clear()