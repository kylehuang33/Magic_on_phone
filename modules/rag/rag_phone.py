import os
import numpy as np
import requests
import json
import sys

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_objectbox.vectorstores import ObjectBox
from objectbox.model import Entity, Id, Property, Vector
from langchain.embeddings.base import Embeddings

print("--- RAG on a Phone Assistant (llama.cpp Embeddings) ---")
print("Initializing...")

# --- Custom Embedding Class to use llama.cpp server ---
class LlamaCppEmbeddingModel(Embeddings):
    def __init__(self, server_url="http://localhost:8088/embedding"):
        # NOTE: Default llama.cpp port is 8080, changed to 8088 here to avoid potential conflicts.
        # Make sure your server runs on this port or change it here.
        self.server_url = server_url

    def _embed(self, text):
        headers = {"Content-Type": "application/json"}
        data = {"content": text}
        try:
            response = requests.post(self.server_url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            return response.json()["embedding"]
        except requests.exceptions.RequestException:
            # On failure, return a zero vector of the correct dimension.
            # This prevents the script from crashing if the server is down.
            return [0.0] * 4096

    def embed_documents(self, texts):
        print(f"   Embedding {len(texts)} document chunks via llama.cpp...")
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        print("   Embedding query via llama.cpp...")
        return self._embed(text)

# --- 1. Define the Data Model for ObjectBox ---
@Entity()
class DocumentChunk:
    id = Id()
    text = Property(str)
    # The embedding dimension for the Phi-3 model is 4096
    embedding = Property(np.ndarray, type=Vector, hnsw_index=True, dimensions=4096)

# --- 2. Document Loading and Preparation ---
# This is the directory on your phone where your documents are stored.
DATA_DIR = "/data/data/com.termux/files/home/storage/downloads/testing"
print(f"1. Loading documents from: {DATA_DIR}")

if not os.path.isdir(DATA_DIR):
    print(f"  ERROR: Directory not found at {DATA_DIR}")
    print("  Please ensure the path is correct and you have granted Termux storage access.")
    print("  Run 'termux-setup-storage' in your terminal if you haven't already.")
    sys.exit(1) # Exit the script with an error code

all_texts = []
try:
    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.isfile(filepath): # Make sure it's a file
            print(f"  - Reading file: {filename}")
            # Read file, ignoring any characters that might cause errors
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                all_texts.append(f.read())
except Exception as e:
    print(f"  ERROR reading files: {e}")
    sys.exit(1)

if not all_texts:
    print("  WARNING: No documents found in the directory.")
    print("  Please add your .txt files to the 'testing' folder in your phone's Downloads.")
    sys.exit(1)

# Join all document contents into one large string, separated by newlines
document_text = "\n\n---\n\n".join(all_texts)
print(f"   Successfully loaded {len(all_texts)} documents.")

# --- 3. Text Splitting ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,      # Size of each chunk
    chunk_overlap=50     # Overlap to maintain context between chunks
)
chunks = text_splitter.split_text(document_text)
print(f"2. Text split into {len(chunks)} chunks.")

# --- 4. Initialize Embedding Model ---
print("3. Initializing llama.cpp embedding model client...")
embedding_model = LlamaCppEmbeddingModel()

# --- 5. Setup ObjectBox Vector Store ---
print("4. Initializing ObjectBox vector store...")
# The DB will be stored in a new folder to reflect the new data source
vector_store = ObjectBox.from_texts(
    texts=chunks,
    embedding=embedding_model,
    db_entity=DocumentChunk,
    db_directory="objectbox_db_mydocs"
)
print("   ObjectBox store is ready and data has been indexed.")

# --- 6. User Query and Retrieval ---
user_query = "What is the core idea of RAG?" # <-- CHANGE THIS TO ASK ABOUT YOUR DOCUMENTS
print(f"\n5. User Query: '{user_query}'")

print("   Querying ObjectBox for relevant context...")
results = vector_store.similarity_search(user_query, k=1)

if not results:
    print("  ERROR: Could not find any relevant context in your documents for this query.")
    sys.exit(1)

retrieved_context = results[0].page_content
print("   Context retrieved.")
print(f"   Retrieved Context: \"{retrieved_context}\"")

# --- 7. Augmenting the Prompt and Generating a Response ---
print("\n6. Generating final response with LLM...")
prompt_template = f"Use the following context to answer the question.\n\nContext: {retrieved_context}\n\nQuestion: {user_query}\n\nAnswer:"

def get_llm_response(prompt):
    url = "http://localhost:8088/completion" # Ensure this port matches the server
    headers = {"Content-Type": "application/json"}
    data = {"prompt": prompt, "n_predict": 256}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        return json.loads(response.text)["content"]
    except requests.exceptions.RequestException as e:
        print(f"\nERROR: Could not connect to LLM server at {url}")
        print("Please ensure the llama.cpp server is running with the '-e' flag in another Termux session.")
        return ""

final_response = get_llm_response(prompt_template)

if final_response:
    print("\n--- Final Result ---")
    print(f"Query: {user_query}")
    print(f"Response:{final_response}")