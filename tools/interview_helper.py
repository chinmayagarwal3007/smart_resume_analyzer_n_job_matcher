from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
from gemini_model import llm
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage


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
