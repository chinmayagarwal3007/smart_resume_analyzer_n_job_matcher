from tools import draft_email_tool
from gemini_model import llm

tools = [draft_email_tool]

llm = llm.bind_tools(tools)