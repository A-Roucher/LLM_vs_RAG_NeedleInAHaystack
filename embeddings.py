import asyncio
import numpy as np

from aiohttp import ClientSession, ClientTimeout
from tqdm.notebook import tqdm
import faiss
from dotenv import load_dotenv
import os
import requests
from haystack import Document
from haystack.nodes import PreProcessor

load_dotenv()

PREPROCESSOR = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=False,
)

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
    

def create_faiss_index(text_embeddings):
    vector_dimension = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(vector_dimension)
    faiss.normalize_L2(text_embeddings)
    index.add(text_embeddings)
    return index


def retrieve_in_faiss_index(index, search_vector, top_k):
    _vector = np.array([search_vector])
    faiss.normalize_L2(_vector)

    k = index.ntotal
    distances, ann = index.search(_vector, k=top_k)
    return distances, ann


async def retrieve_relevant_excerpts_quickly(long_text, question, embedding, words_per_chunk=50, top_k=15, flag_mentions_of_paris=False):
    """
    Retrieves relevant excerpts from a long text using a question and an embedding model
    """
    docs = PREPROCESSOR.split(
        Document(long_text),
        split_by="word",
        split_length=words_per_chunk,
        split_overlap=5,
        split_respect_sentence_boundary=True,
    )
    texts = [{'content': text.content} for text in docs]

    text_embeddings = await get_embeddings(texts)
    text_embeddings = np.array(text_embeddings, dtype=np.float32)


    search_vector = np.array(embedding.embed_query(question), dtype=np.float32)
    # test distance
    
    index = create_faiss_index(text_embeddings)
    distances, ann = retrieve_in_faiss_index(index, search_vector, top_k)

    if flag_mentions_of_paris:
        paris_indexes = []
        for i, text in enumerate(texts):
            if 'Paris' in text['content']:
                paris_indexes.append(i)
        faiss.normalize_L2(text_embeddings)
        _vector = np.array([search_vector])
        faiss.normalize_L2(_vector)
        for i, el in enumerate(ann[0]):
            faiss_distance, square_L2_distance = distances[0][i], np.square(np.linalg.norm(_vector - text_embeddings[el]))
            np.testing.assert_almost_equal(faiss_distance, square_L2_distance, decimal=5)
            if i == len(ann[0]) - 1:
                print("Last selected text distance: ", faiss_distance)
        print("Flagged text:", texts[paris_indexes[0]]['content'])
        print("Text was selected? :", (paris_indexes[0] in ann[0]))
        print("Text distance", np.square(np.linalg.norm(_vector - text_embeddings[paris_indexes[0]])))

    retrieved_docs = [texts[i]['content'] for i in ann[0]]

    return 'DOCUMENT\n'+'\nDOCUMENT:\n'.join(retrieved_docs)

