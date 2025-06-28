import os
import requests
from langchain.tools import tool
from dotenv import load_dotenv

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

# List of tools the agent can use
