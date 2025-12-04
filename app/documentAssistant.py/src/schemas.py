from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Literal, Optional

# ----- Response Schemas -----

# Pydantic model for strcutured Q&A responses
class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    confidence: float = Field(ge=0, le=1)
    timestamp: datetime

# Pydantic model for formatting answers and tracking which documents were used
class UserIntent(BaseModel):
    intent_type: Literal["qa", "summarisation", "calculation", "unknown"]
    confidence: float = Field(ge=0, le=1)
    reasoning: str

