import asyncio
from pathlib import Path
import json
import time
import numpy as np

from aiohttp import ClientSession, ClientTimeout
from tqdm.notebook import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import faiss
from dotenv import load_dotenv
import os
import requests

load_dotenv()

API_URL = "https://ponb0989se5bi47n.us-east-1.aws.endpoints.huggingface.cloud"

# Constants
HEADERS = {
	"Authorization": f"Bearer {os.environ.get('HUGGINGFACEHUB_API_TOKEN')}",
	"Content-Type": "application/json"
}
MAX_WORKERS = 512


def query(payload):
	response = requests.post(API_URL, headers=HEADERS, json=payload)
	return response.json()


def init_embedder(embedding_model_name):
    print(embedding_model_name)
    return HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': False}
    )

async def request(document, semaphore):
    # Semaphore guard
    async with semaphore:
        payload = {
            "inputs": document['content'],
            "truncate": True
        }
        
        timeout = ClientTimeout(total=10)  # Set a timeout for requests (10 seconds here)

        async with ClientSession(timeout=timeout, headers=HEADERS) as session:
            async with session.post(API_URL, json=payload) as resp:
                if resp.status != 200:
                    raise RuntimeError(await resp.text())
                result = await resp.json()
                
        document['embedding'] = result[0]  # Assuming the API's output can be directly assigned
        return document

async def main(documents):
    # Semaphore to limit concurrent requests. Adjust the number as needed.
    semaphore = asyncio.BoundedSemaphore(512)

    # Creating a list of tasks
    tasks = [request(document, semaphore) for document in documents]
    
    # Using tqdm to show progress. It's been integrated into the async loop.
    for f in tqdm(asyncio.as_completed(tasks), total=len(documents)):
        await f


async def get_embeddings(documents):
    # Get embeddings
    await main(documents)
    # Make sure we got it all
    count = 0
    for document in documents:
        if document['embedding'] and len(document['embedding']) > 10:
            count += 1
    return [document['embedding'] for document in documents]
    

def retrieve_faiss(text_embeddings, search_vector, top_k):
    vector_dimension = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(vector_dimension)
    faiss.normalize_L2(text_embeddings)
    index.add(text_embeddings)
    
    _vector = np.array([search_vector])
    faiss.normalize_L2(_vector)

    k = index.ntotal
    distances, ann = index.search(_vector, k=top_k)
    return distances, ann


async def retrieve_relevant_excerpts_quickly(long_text, question, embedding, chunk_size=500, top_k=6):
    """
    Retrieves relevant excerpts from a long text using a question and an embedding model
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap  = 50,
        length_function = len,
        add_start_index = True,
    )
    texts = text_splitter.create_documents([long_text])
    texts = [{'content': text.page_content} for text in texts]

    text_embeddings = await get_embeddings(texts)
    text_embeddings = np.array(text_embeddings, dtype=np.float32)

    search_vector = np.array(embedding.embed_query(question), dtype=np.float32)
    
    _, ann = retrieve_faiss(text_embeddings, search_vector, top_k)
    retrieved_docs = [texts[i]['content'] for i in ann[0]]

    return 'DOCUMENT\n'+'\nDOCUMENT:\n'.join(retrieved_docs)

