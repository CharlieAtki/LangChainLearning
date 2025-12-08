# Document Assistant with LangGraph

A multi-agent document assistant that intelligently routes user queries to specialized agents for question-answering, summarization, and calculations. Built with LangGraph and featuring stateful memory persistence.

## Features

- **Intelligent Intent Classification**: Automatically routes queries to the right agent
- **Multi-Agent System**: Specialized agents for QA, summarization, and calculations
- **Tool Calling**: Dynamic document retrieval and mathematical calculations
- **State Persistence**: Conversation history maintained across sessions
- **Action Tracking**: Accumulates all actions taken using state reducers

## Quick Start

### Installation

```bash
# Install dependencies
pip install langchain-openai langgraph langchain-core python-dotenv pydantic

# Set up environment
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Basic Usage

```python
from langchain_openai import ChatOpenAI
from agent import agent_workflow
from schemas import AgentState
from tools import retrieve_documents, search_specific_document, calculate
import uuid

# Initialize
llm = ChatOpenAI(model="gpt-5-mini")
workflow = agent_workflow()  # Create once and reuse
session_id = str(uuid.uuid4())

# Configure
config = {
    "configurable": {
        "thread_id": session_id,  # Required for state persistence
        "llm": llm,
        "tools": [retrieve_documents, search_specific_document, calculate]
    }
}

# Process message
state = AgentState(user_input="Summarize doc_3", session_id=session_id)
result = workflow.invoke(state, config)

# Get response
print(result['current_response'].answer)
print(f"Actions: {result['actions_taken']}")
```

### Run Demo

```bash
python main.py
```

## Architecture

```
User Input → Triage Agent (classifies intent)
                ↓
    ┌───────────┼───────────┐
    ↓           ↓           ↓
QA Agent   Summarize   Calculate
    ↓           ↓           ↓
Response    Response    Response
```

## Project Structure

```
src/
├── agent.py         # Workflow graph and compilation
├── assistant.py     # Agent node implementations
├── schemas.py       # State and response models
├── prompts.py       # System prompts for each agent
├── tools.py         # Document retrieval and calculator tools
└── main.py          # Entry point and examples
```

## Key Components

### Agents

1. **Triage Agent**: Classifies intent (`qa`, `summarisation`, `calculation`)
2. **QA Agent**: Answers questions using document retrieval
3. **Summarization Agent**: Summarizes documents
4. **Calculation Agent**: Performs math operations on retrieved data

### Tools

- `retrieve_documents(query, max_results)`: Search documents by ID or content
- `search_specific_document(doc_id, query)`: Search within a specific document
- `calculate(expression)`: Evaluate mathematical expressions

### State Management

```python
class AgentState(BaseModel):
    user_input: str
    messages: List[str]
    intent: UserIntent
    current_response: AnswerResponse
    actions_taken: Annotated[List[str], operator.add]  # Accumulates with reducer
    session_id: str
    # ... more fields
```

## Example Queries

```python
# Question answering
"What is machine learning?"

# Summarization
"Summarize doc_3"

# Calculations
"Calculate the total revenue from doc_5"
```

## Multi-Turn Conversations

```python
workflow = agent_workflow()  # Create once
session_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": session_id, "llm": llm, "tools": tools}}

# Message 1
result1 = workflow.invoke(AgentState(user_input="Summarize doc_3"), config)
# actions_taken: ['classify_intent', 'summarisation_agent']

# Message 2 (same session)
result2 = workflow.invoke(AgentState(user_input="What about doc_1?"), config)
# actions_taken: ['classify_intent', 'summarisation_agent', 'classify_intent', 'summarisation_agent']
```

## Important Notes

### ⚠️ Critical: Reuse Workflow Instance

```python
# ❌ WRONG - Creates new checkpointer each time (no state persistence)
result1 = agent_workflow().invoke(state1, config)
result2 = agent_workflow().invoke(state2, config)

# ✅ CORRECT - Reuses same checkpointer (state persists)
workflow = agent_workflow()
result1 = workflow.invoke(state1, config)
result2 = workflow.invoke(state2, config)
```

### State Reducer Pattern

Agents return **dictionaries** (not full AgentState objects) to enable proper state merging:

```python
# ✅ Correct - returns dict, reducer accumulates
def agent_node(state, config):
    return {
        "current_response": response,
        "actions_taken": ["agent_name"]  # Accumulates via operator.add
    }

# ❌ Wrong - returns full state, bypasses reducer
def agent_node(state, config):
    return AgentState(
        actions_taken=state.actions_taken + ["agent_name"]
    )
```

## Configuration

### Change LLM Model

```python
llm = ChatOpenAI(model="gpt-4")  # Use GPT-4
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)  # Adjust temperature
```

### Use Persistent Storage

```python
from langgraph.checkpoint.sqlite import SqliteSaver

def agent_workflow():
    # ... build workflow ...
    checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
    return workflow.compile(checkpointer=checkpointer)
```

### Add Custom Tools

```python
from langchain_core.tools import tool

@tool
def your_tool(param: str) -> dict:
    """Tool description for the LLM."""
    return {"result": "..."}

# Add to config
config = {
    "configurable": {
        "tools": [retrieve_documents, search_specific_document, calculate, your_tool]
    }
}
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Actions not accumulating | Create workflow once: `workflow = agent_workflow()` then reuse it |
| State not persisting | Use same `thread_id` in config for both invocations |
| "KeyError: 'llm'" | Ensure config has `{"configurable": {"llm": llm}}` structure |
| Documents not found | Search by exact ID: `"doc_3"` or by content: `"machine learning"` |

## API Reference

### Main Classes

```python
# State
AgentState(
    user_input: str,
    session_id: str,
    messages: List[str] = [],
    actions_taken: List[str] = []  # Auto-accumulated
)

# Response
AnswerResponse(
    question: str,
    answer: str,
    sources: List[str],
    confidence: float,
    timestamp: datetime
)

# Intent
UserIntent(
    intent_type: Literal["qa", "summarisation", "calculation", "unknown"],
    confidence: float,
    reasoning: str
)
```
---

**Built with**: LangGraph • LangChain • OpenAI • Pydantic
