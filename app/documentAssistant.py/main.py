from typing import Optional, Literal
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI

from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-5-mini")

# ----- Nodes -----

def triage_agent_node(state: State) -> State: # Agent to classify user intent
    prompt = f"User input: {state.input}\nDecide whether this is a 'question', 'summarise', or 'calculation'. Only output one word. If none apply, say 'none'."
    response = llm.invoke(prompt)
    action = response.content.strip().lower()

    if action not in ["question", "summarise", "calculation"]:
        action = None

    return State(input=state.input, action=action)

def question_agent_node(state: State) -> State:
    
