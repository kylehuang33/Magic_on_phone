import os
import numpy as np
import requests
import json
import sys

print("--- RAG on a Phone (Lightweight Edition) ---")
print("Initializing...")

# --- CORE COMPONENTS (NO LANGCHAIN) ---

class LlamaCppServer:
    """A simple client to interact with our llama.cpp server."""
    def __init__(self, host="localhost", port="8088"):
        self.base_url = f"http://{host}:{port}"

    def get_embedding(self, text):
        """Gets a vector embedding for a piece of text."""
        headers = {"Content-Type": "application/json"}
        data = {"content": text}
        try:
            response = requests.post(f"{self.base_url}/embedding", headers=headers, data=json.dumps(data))
            response.raise_for_status()
            # The embedding dimension for Phi-3 is 4096
            return response.json()["embedding"]
        except requests.exceptions.RequestException as e:
            print(f"\nERROR: Could not connect to llama.cpp server: {e}")
            sys.exit(1)

    def get_completion(self, prompt):
        """Gets a text completion from the LLM."""
        headers = {"Content-Type": "application/json"}
        data = {"prompt": prompt, "n_predict": 256, "temperature": 0.2}
        try:
            response = requests.post(f"{self.base_url}/completion", headers=headers, data=json.dumps(data))
            response.raise_for_status()
            return response.json()["content"]
        except requests.exceptions.RequestException:
            print(f"\nERROR: Could not connect to llama.cpp server.")
            return ""

def simple_text_splitter(text, chunk_size=512, chunk_overlap=50):
    """A basic text splitter that splits text by character count."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def find_most_similar(query_embedding, document_embeddings):
    """Finds the most similar document embedding using cosine similarity."""
    query_norm = np.linalg.norm(query_embedding)
    doc_norms = np.linalg.norm(document_embeddings, axis=1)
    
    # Calculate cosine similarities
    similarities = np.dot(document_embeddings, query_embedding) / (doc_norms * query_norm)
    
    # Return the index of the highest similarity score
    return np.argmax(similarities)

# --- 1. SETUP ---
llm_server = LlamaCppServer(port="8088")
DATA_DIR = "/data/data/com.termux/files/home/storage/downloads/testing"
vector_store = {"chunks": [], "embeddings": []}

# --- 2. DOCUMENT LOADING ---
print(f"1. Loading documents from: {DATA_DIR}")
if not os.path.isdir(DATA_DIR):
    print(f"  ERROR: Directory not found. Run 'termux-setup-storage' and ensure the path is correct.")
    sys.exit(1)

all_texts = []
for filename in os.listdir(DATA_DIR):
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.isfile(filepath):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            all_texts.append(f.read())

if not all_texts:
    print("  WARNING: No documents found in the directory.")
    sys.exit(1)

document_text = "\n\n---\n\n".join(all_texts)
print(f"   Successfully loaded {len(all_texts)} documents.")

# --- 3. TEXT SPLITTING ---
print("2. Splitting documents into chunks...")
chunks = simple_text_splitter(document_text)
vector_store["chunks"] = chunks
print(f"   Text split into {len(chunks)} chunks.")

# --- 4. EMBEDDING & STORING ---
print("3. Generating and storing embeddings via llama.cpp...")
embeddings = [llm_server.get_embedding(chunk) for chunk in chunks]
vector_store["embeddings"] = np.array(embeddings)
print("   Embeddings created and stored in memory.")

# --- 5. RAG QUERY ---
user_query = "What is the core idea of RAG?" # <-- CHANGE THIS TO ASK ABOUT YOUR DOCUMENTS
print(f"\n4. User Query: '{user_query}'")

print("   Embedding user query...")
query_embedding = np.array(llm_server.get_embedding(user_query))

print("   Searching for the most relevant context...")
best_match_index = find_most_similar(query_embedding, vector_store["embeddings"])
retrieved_context = vector_store["chunks"][best_match_index]
print(f"   Context retrieved: \"{retrieved_context[:100]}...\"")

# --- 6. GENERATION ---
print("\n5. Generating final response...")
prompt = f"Use the following context to answer the question.\n\nContext: {retrieved_context}\n\nQuestion: {user_query}\n\nAnswer:"
final_response = llm_server.get_completion(prompt)

if final_response:
    print("\n--- Final Result ---")
    print(f"Query: {user_query}")
    print(f"Response:{final_response}")