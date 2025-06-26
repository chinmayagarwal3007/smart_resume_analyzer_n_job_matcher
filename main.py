from tools.interview_helper import explain_candidate_fit, suggest_interview_questions
from tools.job_helper import add_job_to_vd, del_job_from_vd, search_job_from_vd
from tools.extract_info import extract_info_from_resume_text, extract_info_from_job_text
from tools.pdf_parser import parse_pdf
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from pydantic import BaseModel
from typing import List
from langchain_core.messages import BaseMessage
from gemini_model import llm



class AgentState(BaseModel):
    messages: List[BaseMessage]

# Step 1: Define tools
tools = [explain_candidate_fit, suggest_interview_questions,add_job_to_vd, del_job_from_vd, search_job_from_vd, parse_pdf, extract_info_from_resume_text, extract_info_from_job_text]

# Step 2: Setup Gemini model with tools
llm = llm.bind_tools(tools)


# Step 3: LLM node
def agent_node(state):
    messages =  state.messages
    response = llm.invoke(messages)
    return {"messages": messages + [response]}

# Tool execution node
def tool_node(state):
    last_message = state.messages[-1]
    tool_calls = last_message.tool_calls
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        args = tool_call["args"]
        for tool in tools:
            if tool.name == tool_name:
                try:
                    result = tool.invoke(args)                
                except Exception as e:
                    new_messages = state.messages + [AIMessage(content="Can you please provide more details?")]
                    return {"messages": new_messages}
                break

        new_messages = state.messages + [
            ToolMessage(
                tool_call_id=tool_call["id"],
                content=str(result)
            )
        ]
        state.messages = new_messages
    return {"messages": state.messages}

def should_use_tool(state):
    last =  state.messages[-1]
    return "tools" if hasattr(last, "tool_calls") and last.tool_calls else "end"



# Build LangGraph
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_use_tool)
    graph.add_edge("tools", "agent")
    return graph.compile()

#Test
if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({"messages": [HumanMessage(content="Delete Job from database with job id = '12345'")]})
    for msg in result["messages"]:
        print(msg.content)

#Add a job in database with title 'Software Engineer' and description 'Develop software solutions.' and job id '12345'
#Delete Job from database with job id = '12345'