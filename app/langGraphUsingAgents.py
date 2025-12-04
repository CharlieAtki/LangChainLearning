from typing import Optional, Literal
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI

from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ----- STATE -----
class State(BaseModel):
    input: str
    action: Optional[Literal["reverse", "upper"]] = None
    result: Optional[str] = None

# ----- LLM -----
llm = ChatOpenAI(model="gpt-5-mini")

# ----- Nodes -----
def agent_node(state: State) -> State:
    prompt = f"Given this input: {state.input}, choose 'reverse' or 'upper'. Only output one word.  If neither applies, say 'none'."
    response = llm.invoke(prompt)
    action = response.content.strip().lower()

    if action not in ["reverse", "upper"]:
        action = None

    return State(input=state.input, action=action)

def reverse_node(state: State) -> State:
    return State(
        input=state.input,
        action=state.action,
        result=state.input[::-1]
    )

def upper_node(state: State) -> State:
    return State(
        input=state.input,
        action=state.action,
        result=state.input.upper()
    )

workflow = StateGraph(State)

# IMPORTANT: Give nodes explicit names
workflow.add_node("agent", agent_node)
workflow.add_node("reverse_node", reverse_node)
workflow.add_node("upper_node", upper_node)

# ROUTING: from agent → chosen node
workflow.add_conditional_edges(
    "agent",
    lambda s: s.action,
    {
        "reverse": "reverse_node",
        "upper": "upper_node",
        None: END,
    }
)

# END EDGES
workflow.add_edge("reverse_node", END)
workflow.add_edge("upper_node", END)

# Start at agent node
workflow.set_entry_point("agent")

graph = workflow.compile()

# Draw PNG
png_data = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)

print("Graph saved to graph.png")

# Run graph
result = graph.invoke(
    input={
        "input": "Can you add 2+2 for me?",
    }
)

if result.get("result") is None:
    print("Workflow ended early — no transformation applied.")
else:
    print(f"Final result: {result['result']}")

