import PyPDF2
from langchain.tools import tool

@tool
def parse_pdf(file_path: str) -> str:
    """Extract text from a PDF resume file."""
    reader = PyPDF2.PdfReader(file_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

