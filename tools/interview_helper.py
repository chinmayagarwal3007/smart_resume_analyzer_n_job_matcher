from langchain.tools import tool
from gemini_model import llm
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage


@tool
def extract_information_from_resume_text(resume_text: str) -> str:
    """Extract key information from a resume."""

    prompt_template = PromptTemplate.from_template("""Extract structured information from the following resume text: {resume_text} 
And return it strictly in the given JSON format. Only extract if the information is explicitly mentioned. Do not infer or guess. Use empty strings ("") for missing fields.
Respond in this JSON format:
    {{"title": "The job title the candidate is applying for, if mentioned",
  "description": "A 1-2 line summary of what the candidate can do, based on their skills and projects",
  "job_id": "",  // Leave empty
  "skills": ["List", "of", "skills", "mentioned"],
  "location": "Preferred or current location, if mentioned",
  "work_experience": "Total years of work experience, e.g., '3 years', '5+ years'"
}}
""")
    
    
    final_prompt = prompt_template.format(resume_text=resume_text)
    result = llm.invoke([HumanMessage(content=final_prompt)])
    return result

@tool
def extract_information_from_job_text(job_text: str) -> str:
    """Extract key information from a resume."""

    prompt_template = PromptTemplate.from_template("""Extract structured information from the following job text: {job_text} 
And return it strictly in the given JSON format. Only extract if the information is explicitly mentioned. Do not infer or guess. Use empty strings ("") for missing fields.
Respond in this JSON format:
    {{"title": "Title of the job, ex: 'Software Engineer', 'Data Scientist'",
  "description": "A 2-3 line summary of what the candidate have to do, what skills are required, etc.",
  "job_id": "Job id",
  "skills": ["List", "of", "skills", "mentioned"],
  "location": "Location of the job, if mentioned",
  "work_experience": "Total years of work experience required, e.g., '3 years', '5+ years'"
}}
""")
    
    
    final_prompt = prompt_template.format(job_text=job_text)
    result = llm.invoke([HumanMessage(content=final_prompt)])
    return result

@tool
def explain_candidate_fit(resume_text: str, job_text: str) -> str:
    """Explain why candidate is a good fit for the given job."""
    prompt_template =  PromptTemplate.from_template("""
Given the following job description and resume, explain why this candidate is a good fit.

Job Description:
{job_text}

Resume:
{resume_text}
""")
    final_prompt = prompt_template.format(
        job_text=job_text,
        resume_text=resume_text
    )
    result = llm.invoke([HumanMessage(content=final_prompt)])
    return result

@tool
def suggest_interview_questions(resume_text: str, job_text: str) -> str:
    """Generate Interview questions based on user resume"""
    prompt_template = PromptTemplate.from_template("""
Based on the following resume and job description, suggest 3 interview questions to evaluate this candidate.

Job Description:
{job_text}

Resume:
{resume_text}
""")
    final_prompt = prompt_template.format(
        job_text=job_text,
        resume_text=resume_text
    )
    result = llm.invoke([HumanMessage(content=final_prompt)])
    return result
