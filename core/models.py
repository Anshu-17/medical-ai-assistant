from typing import List,Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    
class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    tool_used: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
class QueryResult(BaseModel):
    response: str
    success: bool
    rejected: bool = False
    rejected_reason: Optional[str] = None
    tools_used: List[str] = Field(default_factory=list)
    tool_call_counts: Dict[str, int] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
class ImageMetadata(BaseModel):
    filename: str
    size: tuple
    format: str
    uploaded_at: datetime = Field(default_factory=datetime.now)