import os
import numpy as np
import requests
import sys

print("--- RAG on a Phone (Ollama Edition) ---")
print("Initializing...")

# -------- CONFIG --------
DATA_DIR = "/data/data/com.termux/files/home/storage/downloads/testing"
OLLAMA_HOST = "localhost"
OLLAMA_PORT = "11434"
EMBED_MODEL = "nomic-embed-text"  # embedding model
LLM_MODEL = "gemma3:1b"           # generation model
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
TOP_K = 3

# -------- OLLAMA CLIENT --------
class OllamaClient:
    """Minimal Ollama client for embeddings and generation."""
    def __init__(self, host="localhost", port="11434"):
        self.base_url = f"http://{host}:{port}/api"

    def get_embedding(self, text, model=EMBED_MODEL):
        try:
            resp = requests.post(
                f"{self.base_url}/embeddings",
                json={"model": model, "prompt": text},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except requests.exceptions.RequestException as e:
            if getattr(e, "response", None) and e.response is not None and e.response.status_code == 404:
                print("\nERROR: /api/embeddings not found. Update Ollama or check the endpoint.")
                print("On Termux: `pkg update && pkg upgrade` then reinstall/upgrade ollama if needed.")
            else:
                print(f"\nERROR: Embedding request failed: {e}")
            sys.exit(1)

    def generate(self, prompt, model=LLM_MODEL):
        try:
            resp = requests.post(
                f"{self.base_url}/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"\nERROR: Generation request failed: {e}")
            return ""

# -------- HELPERS --------
def simple_text_splitter(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Simple fixed-size splitter with overlap."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step
    return chunks

def cosine_sim_matrix(query_vec, doc_mat):
    """Cosine similarity between 1D query and 2D doc matrix."""
    q = np.asarray(query_vec, dtype=np.float32)
    D = np.asarray(doc_mat, dtype=np.float32)
    qn = np.linalg.norm(q)
    Dn = np.linalg.norm(D, axis=1)
    # Avoid division by zero
    qn = qn if qn != 0 else 1e-8
    Dn = np.where(Dn == 0, 1e-8, Dn)
    sims = (D @ q) / (Dn * qn)
    return sims

def top_k_indices(scores, k=TOP_K):
    k = min(k, len(scores))
    return np.argpartition(scores, -k)[-k:][np.argsort(scores[np.argpartition(scores, -k)[-k:]])[::-1]]

# -------- 1) SETUP --------
ollama = OllamaClient(OLLAMA_HOST, OLLAMA_PORT)
vector_store = {"chunks": [], "embeddings": None}

# -------- 2) LOAD DOCS --------
print(f"1. Loading documents from: {DATA_DIR}")
if not os.path.isdir(DATA_DIR):
    print("  ERROR: Directory not found. Run `termux-setup-storage` and confirm the path.")
    sys.exit(1)

texts = []
for f in os.listdir(DATA_DIR):
    p = os.path.join(DATA_DIR, f)
    if not os.path.isfile(p):
        continue
    # Only try obvious text-like files; adjust as needed
    if any(p.lower().endswith(ext) for ext in [".txt", ".md", ".log", ".py", ".json", ".csv", ".yaml", ".yml", ".html"]):
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                texts.append(fh.read())
        except Exception as e:
            print(f"  Skipped {f} ({e})")

if not texts:
    print("  WARNING: No text-like documents found in the directory.")
    sys.exit(1)

document_text = "\n\n---\n\n".join(texts)
print(f"   Successfully loaded {len(texts)} documents.")

# -------- 3) SPLIT --------
print("2. Splitting documents into chunks...")
chunks = simple_text_splitter(document_text)
vector_store["chunks"] = chunks
print(f"   Text split into {len(chunks)} chunks.")

# -------- 4) EMBED --------
print("3. Generating embeddings via Ollama (nomic-embed-text)...")
emb_list = []
for i, ch in enumerate(chunks):
    emb = ollama.get_embedding(ch)
    emb_list.append(emb)
    if (i + 1) % 20 == 0:
        print(f"   ...{i+1} / {len(chunks)} chunks embedded")
vector_store["embeddings"] = np.array(emb_list, dtype=np.float32)
print("   Embeddings stored in memory.")

# -------- 5) QUERY --------
user_query = "What is the core idea of RAG?"  # change as needed
print(f"\n4. User Query: '{user_query}'")
print("   Embedding user query...")
q_emb = np.array(ollama.get_embedding(user_query), dtype=np.float32)

print("   Ranking contexts by similarity...")
scores = cosine_sim_matrix(q_emb, vector_store["embeddings"])
idxs = top_k_indices(scores, k=TOP_K)
contexts = [vector_store["chunks"][i] for i in idxs]

# Concatenate top-k contexts (you can add separators or dedupe)
joined_context = "\n\n---\n\n".join(contexts)
print(f"   Top-{TOP_K} contexts selected.")

# -------- 6) GENERATE --------
print("\n5. Generating final response...")
prompt = (
    "You are a helpful assistant. Use ONLY the provided context to answer.\n"
    "If the answer isn't in the context, say you don't know.\n\n"
    f"Context:\n{joined_context}\n\n"
    f"Question: {user_query}\n\n"
    "Answer:"
)
final = ollama.generate(prompt)
if final:
    print("\n--- Final Result ---")
    print(f"Query: {user_query}")
    print("Response:", final.strip())
else:
    print("   No response from the model.")
