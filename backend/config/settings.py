from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    API_TITLE: str = "Financial Advisor AI"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "AI-powered financial advice using custom language models"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    
    # CORS
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000"
    ]
    
    # Model paths
    MODEL_PATH: str = "checkpoints/transformer_1gb_balanced_best.pth"
    DATASET_PATH: str = "data/mega_word_dataset.pkl"
    
    # Generation defaults
    DEFAULT_MAX_LENGTH: int = 100
    DEFAULT_TEMPERATURE: float = 0.8
    MIN_TEMPERATURE: float = 0.1
    MAX_TEMPERATURE: float = 2.0
    
    # Database (for later - Phase 4B)
    DATABASE_URL: Optional[str] = None
    
    # Redis (for later - Phase 4C)
    REDIS_URL: Optional[str] = None
    
    class Config:
        env_file = ".env"

settings = Settings()