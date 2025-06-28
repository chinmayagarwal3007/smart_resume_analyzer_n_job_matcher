from tools import draft_email_tool, add_job_to_vd, search_job_from_vd
from gemini_model import llm
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from pydantic import BaseModel
from typing import List
from langchain_core.messages import BaseMessage

tools = [draft_email_tool, add_job_to_vd, search_job_from_vd]

llm = llm.bind_tools(tools)


class AgentState(BaseModel):
    messages: List[BaseMessage]


def agent_node(state):
    messages =  state.messages
    response = llm.invoke(messages)
    return {"messages": messages + [response]}

# Tool execution node
def tool_node(state):
    last_message = state.messages[-1]
    print("Last message:", last_message)
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
    print("Last message content:", last)
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
    result = app.invoke({"messages": [HumanMessage(content="add a job with title: 'Software Engineer', description: 'Develop software', job_id: '12345', skills: ['Python'], location: 'Remote', work_experience: '5 years'")]})
    for msg in result["messages"]:
        print(msg.content)