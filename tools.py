import os
import requests
from langchain.tools import tool
from dotenv import load_dotenv
from vector_db.add_job import add_job_to_index
from vector_db.search_job import search_jobs_faiss
from typing import TypedDict, List


class Job(TypedDict):
   title: str
   description: str
   job_id: str 
   skills: List[str]
   location: str
   work_experience: str

# --- Tool Definition ---

@tool
def draft_email_tool(instructions: str) -> dict:
    """
    Use this tool ONLY when you need to draft a professional email.
    The input should be a clear instruction or prompt on what the email should be about.
    """
    print("--- 🛠️ Calling Email Drafting Tool ---")
    
    # IMPORTANT: Replace this with the public URL of your other Codespace
    tool_api_url = "https://fuzzy-space-carnival-x4r95gqrrgw3p7wr-8000.app.github.dev/draft_email"
    
    try:
        response = requests.post(tool_api_url, json={"prompt": instructions})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to call the API tool: {e}"}


@tool
def add_job_to_vd(job: Job):
    """Add job given by the user to vector faiss db"""
    return "Job added to vector db successfully"
    # try:
    #     add_job_to_index(job["title"], job["description"], job["job_id"], job["skills"], job["location"])
    #     return "Job has been successfully added to the Database"
    # except Exception as e:
    #     return f"Failed to add job into the database due to error: {e}"

@tool
def search_job_from_vd(query:str, top_k = 3):
    """Search job given by the user from vector faiss db"""
    try:
        result = search_jobs_faiss(query, top_k)
        return f"These are top 3 results: {result}"
    except Exception as e:
        return f"Failed to search job from the database due to error: {e}"
