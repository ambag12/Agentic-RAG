# GLOBAL
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
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate

# VECTOR STORE
import pinecone
from pinecone import Pinecone, ServerlessSpec
# AGENTS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, Tool, AgentType
from langchain.agents.react.agent import create_react_agent
from langchain import hub
import os

load_dotenv()

MISTRAL_API_KEY = os.environ["mistral_api_key"] ="Wxn2lCBcugXAz0GK265wJR4H6jYRrNl3"


loader = CSVLoader(
    file_path=r"C:\Users\hussa\FastAPI\AGENTIC RAG\Agentic-RAG-LangChain-main\tedx_document.csv",
    encoding='utf-8',
    source_column="transcript",
    metadata_columns=["main_speaker", "name", "speaker_occupation", "title", "url", "description"]
)


data = loader.load()

len(data)


def num_tokens_from_string(question, encoding_name):

    encoding = tiktoken.get_encoding(encoding_name)

    num_tokens = encoding.encode(question)

    return encoding, num_tokens


question = "How many TEDx talks are on the list?"

encoding, num_tokens = num_tokens_from_string(question, "cl100k_base")

print(f'Number of Words: {len(question.split())}')
print(f'Number of Characters: {len(question)}')
print(f'List of Tokens: {num_tokens}')
print(f'Nr of Tokens: {len(num_tokens)}')
encoding.decode([4438, 1690, 84296, 87, 13739, 527, 389, 279, 1160, 30])
# Define cosine similarity function
from langchain_huggingface import HuggingFaceEmbeddings

def cosine_similarity(query_emb, document_emb):

    # Calculate the dot product of the query and document embeddings
    dot_product = np.dot(query_emb, document_emb)

    # Calculate the L2 norms (magnitudes) of the query and document embeddings
    query_norm = np.linalg.norm(query_emb)
    document_norm = np.linalg.norm(document_emb)

    # Calculate the cosine similarity
    cosine_sim = dot_product / (query_norm * document_norm)
    print(cosine_sim)
    return cosine_sim
question = "What is the topic of the TEDx talk from Al Gore?"
document = "Averting the climate crisis"
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
query_emb=embedding.embed_query(question)
document_emb=embedding.embed_query(document)


query_emb = embedding.embed_query(question)
document_emb = embedding.embed_query(document)

cosine_sim = cosine_similarity(query_emb, document_emb)

# print(f'Query Vector: {query_emb}')
# print(f'Document Vector: {document_emb}')

print(f'Query Dimensions: {len(query_emb)}')
print(f'Document Dimensions: {len(document_emb)}')

print("Cosine Similarity:", cosine_sim)


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-3.5-turbo-0125",
    chunk_size=512,
    chunk_overlap=20,
    separators= ["\n\n", "\n", " ", ""])
	
	# Pinecone Initialization
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"] ="pcsk_6L8ZmY_fh3AkwzUNVg87rneaKU2HgoApGK91wJJoVoYtX4HSdnKdxaWpuktSyjkApsBeZ"
index_name = "agenticragmodel"
print(PINECONE_API_KEY)
pc = Pinecone(api_key = PINECONE_API_KEY)


pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"))

index = pc.Index(index_name)

pc.list_indexes()


index = pc.Index(index_name)

index.describe_index_stats()


splits = text_splitter.split_documents(data[:100])

embed = embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = PineconeVectorStore.from_documents(documents=splits,
                                        embedding=embed,
                                        index_name=index_name,
                                        namespace="main"
                                        )

vectorstore = PineconeVectorStore(index_name=index_name,
                                  namespace="main",
                                  embedding=embed)

query = "Who is Al Gore"

similarity = vectorstore.similarity_search(query, k=4)

for i in range(len(similarity)):
  print(f"-------Result Nr. {i}-------")
  print(f"Main Speaker: {similarity[i].metadata['main_speaker']}")
  print(f" ")
  
# Search for similarity with score

query = "Who is Al Gore"

similarity_with_score = vectorstore.similarity_search_with_score(query, k=4)

for i in range(len(similarity_with_score)):
  print(f"-------Result Nr. {i}-------")
  print(f"Title: {similarity_with_score[i][0].metadata['title']}")
  print(f"Main Speaker: {similarity_with_score[i][0].metadata['main_speaker']}")
  print(f"Score: {similarity_with_score[i][1]}")
  print(f" ")


def chunked_metadata_embeddings(documents, embed):

    chunked_metadata = []

    chunked_text = text_splitter.split_documents(documents)

    for index, text in enumerate(tqdm(chunked_text)):


        payload = {
              "metadata": {
                  "source": text.metadata['source'],
                  "row": text.metadata['row'],
                  "chunk_num": index,
                  "main_speaker": text.metadata['main_speaker'],
                  "name": text.metadata['name'],
                  "speaker_occupation": text.metadata['speaker_occupation'],
                  "title": text.metadata['title'],
                  "url": text.metadata['url'],
                  "description": text.metadata['description'],
              },
              "id": str(uuid4()),
              "values": embed.embed_documents([text.page_content])[0]  # Assuming `embed` is defined elsewhere
          }

        chunked_metadata.append(payload)

    return chunked_metadata

split_one = chunked_metadata_embeddings(data[:50], embed)
len(split_one)

split_two = chunked_metadata_embeddings(data[50:100], embed)
len(split_two)


def batch_upsert(split,
                 index ,
                 namespace,
                 batch_size):

    print(f"Split Length: {len(split)}")
    for i in range(0, len(split), batch_size):

      batch = split[i:i + batch_size]

      index.upsert(vectors=batch,
                   namespace=namespace)

batch_upsert(split_one, index, "first_split", 10)


def find_item_with_row(metadata_list, main_speaker):
    for item in metadata_list:
        if item['metadata']['main_speaker'] == main_speaker:
            return item

# Call the function to find item with main_speaker = Al Gore
result_item = find_item_with_row(split_one, "Al Gore")

# Print the result
print(f'Chunk Nr: {result_item["metadata"]["chunk_num"]}')
print(f'Chunk ID: {result_item["id"]}')
print(f'Chunk Title: {result_item["metadata"]["title"]}')

index.describe_index_stats()
batch_upsert(split_two, index, "last_split", 20)

index.describe_index_stats()


query_one = "Who is Al Gore?"
query_two = "Who is Rick Warren?"
query_three="Who is Mr.Gilbert ? And what is his profesion"
# Users dictionary
users = [{
            'name': 'John',
            'namespace': 'first_split',
            'query': query_one

            },
           {
             "name": "Jane",
             "namespace": 'last_split',
             "query": query_two
           },
           {
               "name":"Ahmed Mujtaba",
               "namespace":"admin",
               "query":query_three
           }
           ]

def vectorize_query(embed, query):

    return embed.embed_query(query)

query_vector_one = vectorize_query(embed, query_one)
query_vector_two = vectorize_query(embed, query_two)
query_vector_three=vectorize_query(embed, query_three)
len(query_vector_one), len(query_vector_two),len(query_vector_three)

new_key_value_pairs = [
    {'vector_query': query_vector_one},
    {'vector_query': query_vector_two},
    {'vector_query': query_vector_three},

]

# Loop through the list of users and the list of new key-value pairs
for user, new_pair in zip(users, new_key_value_pairs):
    user.update(new_pair)
users[0]["name"], users[1]["name"]
users[0].keys()
print(f"Name: {users[0]['name']}")
print(f"Namespace: {users[0]['namespace']}")
print(f"Query: {users[0]['query']}")
print(f"Vector Query: {users[0]['vector_query'][:3]}")

john = [t for t in users if t.get('name') == 'John'][0]

john_query_vector = john['vector_query']
john_namespace = john['namespace']

index.query(vector=john_query_vector, top_k=2, include_metadata=True, namespace=john_namespace)

#NOW we finally set our database in PineCone for RAG 