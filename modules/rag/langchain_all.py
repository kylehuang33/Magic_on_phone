import requests
from typing import List

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import TextLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import Embeddings # Import the base class
from langchain_text_splitters import CharacterTextSplitter

# --- Your Custom Embeddings Class from Step 1 ---
class CustomOllamaEmbeddings(Embeddings):
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip('/')

    def _get_embedding(self, text: str) -> List[float]:
        try:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=60,
            )
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


# 1. Define the base embeddings model using YOUR custom class
model_name = "nomic-embed-text:latest"
# The 'namespace' for the cache MUST be a valid filename. The model name itself
# is now an attribute of your class, not the primary identifier for LangChain.
# We will create a simple, safe namespace string.
safe_namespace = model_name.replace(":", "_")

underlying_embeddings = CustomOllamaEmbeddings(model=model_name)


# 2. Set up the local file store for caching
store = LocalFileStore("./cache/")


# 3. Create the CacheBackedEmbeddings
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=safe_namespace
)

# 4. Load and split the documents
print("Loading documents...")
raw_documents = TextLoader("./data/rag/meeting.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
print(f"Split into {len(documents)} documents.")

# 5. Create the InMemoryVectorStore using the cached embedder
print("Embedding documents and creating vector store...")
vector_store = InMemoryVectorStore.from_documents(documents, cached_embedder)
print("Vector store created successfully.")


# 6. Use the vector_store for similarity searches
query = "What did the president say about Ketanji Brown Jackson"
retriever = vector_store.as_retriever()
result_docs = retriever.get_relevant_documents(query)

print("\n--- Query Results ---")
print(result_docs[0].page_content)
print("---------------------\n")