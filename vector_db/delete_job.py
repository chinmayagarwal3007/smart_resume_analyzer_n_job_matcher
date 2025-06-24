import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def delete_job_by_id(job_id: str):
    # Load metadata
    with open("faiss_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    print("metadata:", metadata)
    # Find index of job_id in metadata
    index_to_remove = next((i for i, job in enumerate(metadata) if job['job_id'] == job_id), None)
    print(f"Index to remove: {index_to_remove}")
    if index_to_remove is None:
        print(f"❌ Job ID '{job_id}' not found.")
        return

    # Remove the entry
    removed_job = metadata.pop(index_to_remove)
    print("metadata:", metadata)
    # Rebuild documents
    documents = [
        f"""Title: {job['title']}
Description: {job['description']}
Job ID: {job['job_id']}"""
        for job in metadata
    ]

    if len(documents) != 0:
        embeddings = model.encode(documents, convert_to_numpy=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, "faiss_index.index")
        with open("faiss_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
    else:
        embedding_dim = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(embedding_dim)
        faiss.write_index(index, "faiss_index.index")
        with open("faiss_metadata.pkl", "wb") as f:
            pickle.dump([], f)

    print(f"✅ Deleted job '{removed_job['job_id']}' and updated index.")
