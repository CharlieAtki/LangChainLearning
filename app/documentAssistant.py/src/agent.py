from typing import Optional, Literal
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from schemas import AgentState, UserIntent, AnswerResponse

from assistant import triage_agent_node, qa_agent_node, summarisation_agent_node, calculation_agent_node

def agent_workflow():
    workflow = StateGraph(AgentState)

    # Add all agent nodes
    workflow.add_node("triage_agent", triage_agent_node)
    workflow.add_node("qa_agent", qa_agent_node)
    workflow.add_node("summarisation_agent", summarisation_agent_node)
    workflow.add_node("calculation_agent", calculation_agent_node)

    # Conditional routing from triage agent
    workflow.add_conditional_edges(
        "triage_agent",
        lambda s: s.next_step,
        {
            "qa_agent": "qa_agent",
            "summarisation_agent": "summarisation_agent",
            "calculation_agent": "calculation_agent",
            None: END,
        }
    )

    # All specialized agents end the workflow
    workflow.add_edge("qa_agent", END)
    workflow.add_edge("summarisation_agent", END)
    workflow.add_edge("calculation_agent", END)

    # Set entry point
    workflow.set_entry_point("triage_agent")

    # Compile the graph
    graph = workflow.compile()

    # Draw PNG
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(png_data)
        print("Graph saved to graph.png")
    except Exception as e:
        print(f"Could not save graph visualization: {e}")

    return graph