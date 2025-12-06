from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Literal, Optional, Dict, Any

# ----- Response Schemas -----

# Pydantic model for structured Q&A responses
class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    confidence: float = Field(ge=0, le=1)
    timestamp: datetime
    retrieved_documents: List[Dict[str, Any]] = []  # Store retrieved docs

# Pydantic model for formatting answers and tracking which documents were used
class UserIntent(BaseModel):
    intent_type: Literal["qa", "summarisation", "calculation", "unknown"]
    confidence: float = Field(ge=0, le=1)
    reasoning: str

class AgentState(BaseModel):
    # The raw user message
    user_input: Optional[str] = None
    
    # Full conversation history (LangGraph annotated)
    messages: List[str] = []
    
    # Output of the classify_intent node
    intent: Optional[UserIntent] = None
    
    # Determines which node should run next
    next_step: Optional[str] = None
    
    # A rolling summary of the conversation
    conversation_summary: Optional[str] = None
    
    # Which document IDs are currently considered relevant
    active_documents: List[str] = []
    
    # The main structured response
    current_response: Optional[AnswerResponse] = None
    
    # Which tools were used in this turn (retriever, calculator, etc.)
    tools_used: List[str] = []
    
    # Session and user tracking
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Log of nodes executed this turn
    actions_taken: List[str] = []
    
    # Store retrieved documents for the current session
    retrieved_documents: Optional[AnswerResponse] = None