
import os
import pandas as pd
import numpy as np
import tiktoken
from uuid import uuid4
# from tqdm import tqdm
from dotenv import load_dotenv
from tqdm.autonotebook import tqdm


# LANGCHAIN
from langchain_huggingface import HuggingFaceEmbeddings
import langchain
from langchain.llms import OpenAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.memory import ConversationBufferWindowMemory

# VECTOR STORE
import pinecone
from pinecone import Pinecone, ServerlessSpec
# AGENTS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, Tool, AgentType
from langchain.agents.react.agent import create_react_agent
from langchain import hub

# Load environmental variables from a .env file
load_dotenv()
import os


MISTRAL_API_KEY = os.environ["mistral_api_key"] ="Wxn2lCBcugXAz0GK265wJR4H6jYRrNl3"
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"] ="pcsk_6L8ZmY_fh3AkwzUNVg87rneaKU2HgoApGK91wJJoVoYtX4HSdnKdxaWpuktSyjkApsBeZ"
index_name = "agenticragmodel"
embed = embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = PineconeVectorStore(index_name=index_name,
                                  namespace="main",
                                  embedding=embed)

llm = ChatMistralAI(temperature=0.0,model_name="mistral-small", max_tokens=512)


# Conversational memory
conversational_memory = ConversationBufferWindowMemory(
                        memory_key='chat_history',
                        k=5,
                        return_messages=True)

# Retrieval qa chain
qa_db = RetrievalQA.from_chain_type(
                                    llm=llm,
                                    chain_type="stuff",
                                    retriever=vectorstore.as_retriever())
									
									

#Augmented

prompt = hub.pull("hwchase17/react")

print(prompt.template)


template= '''
          Answer the following questions as best you can. You have access to the following tools:

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
          Thought:{agent_scratchpad}
          '''

prompt = PromptTemplate.from_template(template)

TAVILY_API_KEY=os.environ["TAVILY_API_KEY"]="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"

tavily = TavilySearchResults(max_results=10, tavily_api_key=TAVILY_API_KEY)

tools = [
    Tool(
        name = "Pinecone Document Store",
        func = qa_db.run,
        description = "Use it to lookup information from the Pinecone Document Store"
    ),

    Tool(
        name="Tavily",
        func=tavily.run,
        description="Use this to lookup information from Tavily",
    )
]

agent = create_react_agent(llm,
                           tools,
                           prompt)

agent_executor = AgentExecutor(tools=tools,
                         agent=agent,
                         handle_parsing_errors=True,
                         verbose=True,
                         memory=conversational_memory)
						 
#now we can use the RAG to ask questions 

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
    print(response)

conversational_memory.load_memory_variables({})


agent_executor.memory.clear()
conversational_memory.clear()
