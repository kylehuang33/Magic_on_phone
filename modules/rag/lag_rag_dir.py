import os
import requests
from typing import List

# Make sure you have the necessary packages installed:
# pip install pypdf "langchain-community[pandas]"
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    JSONLoader,
    CSVLoader,
    PyPDFLoader
)
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
store = LocalFileStore("./cache/magic_cue")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=safe_namespace
)
# --------------------------------------------------

# 4. LOAD DOCUMENTS FROM MULTIPLE FORMATS
print("Loading documents from directory...")
DATA_DIR = "/data/data/com.termux/files/home/storage/downloads/testing"
all_raw_documents = []

# Loader for .txt files
txt_loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
all_raw_documents.extend(txt_loader.load())

# Loader for .json files
json_loader = DirectoryLoader(DATA_DIR, glob="**/*.json", loader_cls=JSONLoader, loader_kwargs={'jq_schema': '.content'}, show_progress=True)
all_raw_documents.extend(json_loader.load())

# Loader for .jsonl files
jsonl_loader = DirectoryLoader(DATA_DIR, glob="**/*.jsonl", loader_cls=JSONLoader, loader_kwargs={'jq_schema': '.message', 'json_lines': True}, show_progress=True)
all_raw_documents.extend(jsonl_loader.load())

# Loader for .csv files
csv_loader = DirectoryLoader(DATA_DIR, glob="**/*.csv", loader_cls=CSVLoader, loader_kwargs={'source_column': 'employee_id', 'encoding': 'utf-8'}, show_progress=True)
all_raw_documents.extend(csv_loader.load())

# Loader for .pdf files
pdf_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
all_raw_documents.extend(pdf_loader.load())


print(f"Loaded a total of {len(all_raw_documents)} documents from all sources.")

# 5. SPLIT, EMBED, and STORE (No changes)
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
    print(result_docs[0].page_content)
    print(f"\nSource: {result_docs[0].metadata.get('source', 'N/A')}")
else:
    print("No relevant documents found.")
print("---------------------\n")