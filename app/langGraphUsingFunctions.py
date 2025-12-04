from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Literal
from langgraph.graph import StateGraph, END, START


class State(BaseModel):
    input: str
    action: Literal["reverse", "upper"]  # strict allowed actions
    result: Optional[str] = None

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

workflow.add_node(reverse_node)
workflow.add_node(upper_node)

def routing_fuction(state: State) -> str:
    if state.action == "reverse":
        return "reverse_node"
    elif state.action == "upper":
        return "upper_node"
    else:
        raise ValueError(f"Unknown action: {state.action}")

workflow.add_conditional_edges(
    source=START,
    path=routing_fuction,
    path_map=["reverse_node", "upper_node"]
)

workflow.add_edge("reverse_node", END)
workflow.add_edge("upper_node", END)

graph = workflow.compile()

# Instead of display(Image(...))
# png_data = graph.get_graph().draw_mermaid_png()

# # Save to file
# with open("graph.png", "wb") as f:
#     f.write(png_data)

# print("Graph saved to graph.png")

result = graph.invoke(
    input = {
        "input": "Some input",
        "action": "reverse",
    }
)

print(f"Final result: {result["result"]}")
