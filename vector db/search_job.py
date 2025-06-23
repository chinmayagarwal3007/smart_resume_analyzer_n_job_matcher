import pickle
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("faiss_index.index")
with open("faiss_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

def search_jobs_faiss(query, top_k=3):
    query_vector = model.encode([query])
    D, I = index.search(query_vector, top_k)
    return [metadata[i] for i in I[0]]