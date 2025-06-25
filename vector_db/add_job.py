import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def add_job_to_index(title: str, description: str, job_id: str, skills: list[str], location: str = "Remote"):
    
   
     # Create document and embed it
    document = f"""Title: {title}
Description: {description}
Job ID: {job_id}
Skills: {', '.join(skills)}
Location: {location}"""
    
    embedding = model.encode([document], convert_to_numpy=True)

    
    try:
        # Load existing FAISS index
        index = faiss.read_index("faiss_index.index")
    except:
        index = faiss.IndexFlatL2(embedding.shape[1])

    # Add to index
    index.add(embedding)

    try:
        # Load existing metadata
        with open("faiss_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
    except:
        metadata = []
    # Add to metadata
    metadata.append({
        "title": title,
        "description": description,
        "job_id": job_id
    })

    # Save index and metadata again
    faiss.write_index(index, "faiss_index.index")
    with open("faiss_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("✅ Job added to FAISS index and metadata.")
