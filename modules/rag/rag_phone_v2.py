import os
import numpy as np
import requests
import json
import sys

print("--- RAG on a Phone (Ollama Edition) ---")
print("Initializing...")

# --- CORE COMPONENTS (NO LANGCHAIN) ---

class OllamaClient:
    """A simple client to interact with the Ollama server."""
    def __init__(self, host="localhost", port="11434", model="phi3"):
        self.base_url = f"http://{host}:{port}/api"
        self.model = model

    def get_embedding(self, text):
        """Gets a vector embedding for a piece of text from Ollama."""
        try:
            endpoint = f"{self.base_url}/embeddings"
            data = {"model": self.model, "prompt": text}
            response = requests.post(endpoint, json=data)
            response.raise_for_status()
            # The embedding dimension for Ollama's phi3 is 3072
            return response.json()["embedding"]
        except requests.exceptions.RequestException as e:
            print(f"\nERROR: Could not connect to Ollama server: {e}")
            sys.exit(1)

    def get_completion(self, prompt):
        """Gets a text completion from the Ollama LLM."""
        try:
            endpoint = f"{self.base_url}/generate"
            data = {"model": self.model, "prompt": prompt, "stream": False}
            response = requests.post(endpoint, json=data)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            print(f"\nERROR: Could not connect to Ollama server: {e}")
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
    similarities = np.dot(document_embeddings, query_embedding) / (doc_norms * query_norm)
    return np.argmax(similarities)

# --- 1. SETUP ---
ollama_client = OllamaClient() # Our new client
DATA_DIR = "/data/data/com.termux/files/home/storage/downloads/testing"
vector_store = {"chunks": [], "embeddings": []}

# --- 2. DOCUMENT LOADING ---
print(f"1. Loading documents from: {DATA_DIR}")
if not os.path.isdir(DATA_DIR):
    print(f"  ERROR: Directory not found. Run 'termux-setup-storage' and ensure path is correct.")
    sys.exit(1)
all_texts = [open(os.path.join(DATA_DIR, f), 'r', encoding='utf-8', errors='ignore').read() for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
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
print("3. Generating and storing embeddings via Ollama...")
embeddings = [ollama_client.get_embedding(chunk) for chunk in chunks]
vector_store["embeddings"] = np.array(embeddings)
print("   Embeddings created and stored in memory.")

# --- 5. RAG QUERY ---
user_query = "What is the core idea of RAG?" # <-- CHANGE THIS TO ASK ABOUT YOUR DOCUMENTS
print(f"\n4. User Query: '{user_query}'")
print("   Embedding user query...")
query_embedding = np.array(ollama_client.get_embedding(user_query))
print("   Searching for the most relevant context...")
best_match_index = find_most_similar(query_embedding, vector_store["embeddings"])
retrieved_context = vector_store["chunks"][best_match_index]
print(f"   Context retrieved: \"{retrieved_context[:100]}...\"")

# --- 6. GENERATION ---
print("\n5. Generating final response...")
prompt = f"Use the following context to answer the question.\n\nContext: {retrieved_context}\n\nQuestion: {user_query}\n\nAnswer:"
final_response = ollama_client.get_completion(prompt)
if final_response:
    print("\n--- Final Result ---")
    print(f"Query: {user_query}")
    print(f"Response:{final_response.strip()}")