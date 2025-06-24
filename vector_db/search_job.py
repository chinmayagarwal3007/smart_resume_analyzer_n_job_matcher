import pickle
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

try:
    index = faiss.read_index("faiss_index.index")
    with open("faiss_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    def search_jobs_faiss(query:str, top_k:int):
        query_vector = model.encode([query])
        D, I = index.search(query_vector, top_k)
        return [metadata[i] for i in I[0]]
except:
    def search_jobs_faiss(query:str, top_k:int):
        return []