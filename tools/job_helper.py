from langchain.tools import tool
from typing import TypedDict
from vector_db.add_job import add_job_to_index
from vector_db.delete_job import delete_job_by_id
from vector_db.search_job import search_jobs_faiss

class Job(TypedDict):
   title: str
   description: str
   job_id: str 
   skills: list[str]
   location: str



@tool
def add_job_to_vd(job: Job):
    """Add job given by the user to vector faiss db"""
    try:
        add_job_to_index(job["title"], job["description"], job["job_id"], job["skills"], job["location"])
        return "Job has been successfully added to the Database"
    except Exception as e:
        return f"Failed to add job into the database due to error: {e}"

@tool
def del_job_from_vd(job_id:str):
    """Delete job given by the user from vector faiss db"""
    try:
        delete_job_by_id(job_id)
        return "Job has been successfully deleted from the Database"
    except Exception as e:
        return f"Failed to delete job from the database due to error: {e}"
    
@tool
def search_job_from_vd(query:str, top_k = 3):
    """Search job given by the user from vector faiss db"""
    try:
        result = search_jobs_faiss(query, top_k)
        return f"These are top 3 results: {result}"
    except Exception as e:
        return f"Failed to search job from the database due to error: {e}"