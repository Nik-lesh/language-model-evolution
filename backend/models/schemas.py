from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from uuid import uuid4

class GenerateRequest(BaseModel):
    """Request for text generation."""
    
    prompt: str = Field(..., min_length=1, max_length=500, 
                       description="Starting text for generation")
    max_length: int = Field(default=100, ge=10, le=500,
                           description="Number of words to generate")
    temperature: float = Field(default=0.8, ge=0.1, le=2.0,
                              description="Sampling temperature (higher = more creative)")
    
    @validator('prompt')
    def prompt_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Prompt cannot be empty')
        return v.strip()

class GenerateResponse(BaseModel):
    """Response from text generation."""
    
    generated_text: str
    prompt: str
    word_count: int
    model_info: dict

    model_config = {"protected_namespaces": ()}

class ChatMessage(BaseModel):
    """Single chat message."""
    
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatRequest(BaseModel):
    """Chat request with conversation context."""
    
    message: str = Field(..., min_length=1, max_length=1000)
    conversation_id: Optional[str] = None
    max_length: int = Field(default=100, ge=10, le=500)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)

class ChatResponse(BaseModel):
    """Chat response with metadata."""
    
    conversation_id: str
    response: str
    model_used: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict

    model_config = {"protected_namespaces": ()}

class HealthResponse(BaseModel):
    """API health check response."""
    
    status: str
    version: str
    model_loaded: bool
    device: str

    model_config = {"protected_namespaces": ()}
