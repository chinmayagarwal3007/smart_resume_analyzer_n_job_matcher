import PyPDF2
from langchain.tools import tool
from resume_parser import resumeparse


@tool
def parse_resume(file_path: str) -> str:
    """Parse resume from PDF"""
    from resume_parser import resumeparse
    data = resumeparse.read_file(file_path)
    print(data)
    return data