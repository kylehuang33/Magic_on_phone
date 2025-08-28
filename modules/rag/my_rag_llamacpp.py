import os
import requests
from typing import List

# --- NEW: Import Llama from llama_cpp ---
from llama_cpp import Llama

# Ensure necessary packages are installed
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import CharacterTextSplitter

# --- Custom Embeddings Class (No changes) ---
class CustomOllamaEmbeddings(Embeddings):
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip('/')
    def _get_embedding(self, text: str) -> List[float]:
        try:
            resp = requests.post(f"{self.base_url}/api/embeddings", json={"model": self.model, "prompt": text}, timeout=60)
            resp.raise_for_status()
            return resp.json()["embedding"]
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            raise
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding(text) for text in texts]
    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding(text)
# --------------------------------------------------

# --- Model and Cache Setup (No changes) ---
model_name = "nomic-embed-text:latest"
safe_namespace = model_name.replace(":", "_")
underlying_embeddings = CustomOllamaEmbeddings(model=model_name)
store = LocalFileStore("./cache/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=safe_namespace
)
# --------------------------------------------------

# 4. LOAD AND PROCESS THE SYNTHETIC DATA (No changes)
print("Loading and processing documents from the synthetic dataset...")
DATA_DIR = "./data/magic_cue" # The base directory where your files are
all_raw_documents = []

# Loader for notifications.jsonl
notifications_loader = DirectoryLoader(DATA_DIR, glob="**/notifications.jsonl", loader_cls=JSONLoader,
                                       loader_kwargs={'jq_schema': '"Notification from \(.title): \(.text)"', 'json_lines': True}, show_progress=True)
all_raw_documents.extend(notifications_loader.load())

# Loader for calendar_events.json
calendar_loader = DirectoryLoader(DATA_DIR, glob="**/calendar_events.json", loader_cls=JSONLoader,
                                  loader_kwargs={'jq_schema': '.[] | "Calendar Event: \(.title) at \(.location)"', 'json_lines': False}, show_progress=True)
all_raw_documents.extend(calendar_loader.load())

# Loader for photos_index.jsonl
photos_loader = DirectoryLoader(DATA_DIR, glob="**/photos_index.jsonl", loader_cls=JSONLoader,
                                loader_kwargs={'jq_schema': '"Photo Album: \(.album). Tags: \(.tags | join(", "))"', 'json_lines': True}, show_progress=True)
all_raw_documents.extend(photos_loader.load())

# Loader for facts_appsearch.jsonl
facts_loader = DirectoryLoader(DATA_DIR, glob="**/facts_appsearch.jsonl", loader_cls=JSONLoader,
                               loader_kwargs={'jq_schema': '"Fact: \(.title). Details: \(.entities | tostring)"', 'json_lines': True}, show_progress=True)
all_raw_documents.extend(facts_loader.load())


print(f"Loaded a total of {len(all_raw_documents)} relevant documents.")

# 5. SPLIT, EMBED, AND STORE (No changes)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(all_raw_documents)
print(f"Split into {len(documents)} document chunks.")

print("Embedding documents and creating vector store...")
vector_store = InMemoryVectorStore.from_documents(documents, cached_embedder)
print("Vector store created successfully.")
# --------------------------------------------------

# 6. QUERY (No changes)
query = "Where and when is the dinner with Jamie?"
retriever = vector_store.as_retriever()
result_docs = retriever.get_relevant_documents(query)

print("\n--- Query Results ---")
if result_docs:
    for doc in result_docs:
      print(f"Content: {doc.page_content}")
      print(f"Source: {doc.metadata.get('source', 'N/A')}\n")
else:
    print("No relevant documents found.")
print("---------------------\n")

# 7. --- MODIFIED: QUESTION ANSWERING WITH LLAMA-CPP ---
print("--- Generating Answer with llama-cpp ---")
context = "\n\n".join(doc.page_content for doc in result_docs)

# Build the prompt
prompt = f"""
Answer the following question based only on the provided context.

Context:
{context}

Question: {query}
"""

try:
    # --- CHANGE: Initialize the Llama model ---
    # IMPORTANT: Replace "path/to/your/model.gguf" with the actual path to your downloaded model file.
    llm = Llama(
        model_path="/data/data/com.termux/files/home/models/gemma-3-1b-it-GGUF/gemma-3-1b-it-Q4_K_M.gguf",
        n_ctx=2048,      # Context window size
        n_gpu_layers=-1, # -1 to offload all possible layers to GPU
        verbose=False    # Set to True to see more detailed output
    )

    # --- CHANGE: Call the Llama.cpp API ---
    # The API is similar to OpenAI's chat completions
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based only on the provided context."},
            {"role": "user", "content": prompt},
        ]
    )

    # Extract and print the answer
    answer = response['choices'][0]['message']['content']
    print("Generated Answer:")
    print(answer)

except Exception as e:
    print(f"\nAn error occurred during generation: {e}")
    print("Please ensure you have installed llama-cpp-python and provided the correct path to your model file.")

finally:
    print("-------------------------------------------\n")