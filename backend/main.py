from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))  # project root
sys.path.insert(0, str(Path(__file__).parent))  # backend/

# Now import with relative paths
from config.settings import settings
from services.model_service import model_service
from models.schemas import (
    GenerateRequest, GenerateResponse,
    ChatRequest, ChatResponse,
    HealthResponse
)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# LIFESPAN MANAGEMENT
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    
    # Startup
    logger.info("🚀 Starting Financial Advisor API...")
    try:
        model_service.load()
        logger.info("✅ Models loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down API...")

# ============================================================
# APP INITIALIZATION
# ============================================================

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# ROUTES
# ============================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """API health check and info."""
    
    return HealthResponse(
        status="healthy",
        version=settings.API_VERSION,
        model_loaded=model_service.model is not None,
        device=str(model_service.device)
    )

@app.get("/health")
async def health():
    """Detailed health check."""
    
    return {
        "status": "healthy",
        "model_loaded": model_service.model is not None,
        "dataset_loaded": model_service.dataset is not None,
        "device": str(model_service.device),
        "model_info": model_service.get_model_info() if model_service.model else None
    }

@app.get("/models")
async def list_models():
    """List available models and their info."""
    
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "models": [model_service.get_model_info()],
        "count": 1
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """
    Generate text completion from a prompt.
    
    This is the main text generation endpoint.
    """
    
    try:
        # Generate text
        generated = model_service.generate(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        return GenerateResponse(
            generated_text=generated,
            prompt=request.prompt,
            word_count=len(generated.split()),
            model_info=model_service.get_model_info()
        )
    
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Text generation failed: {str(e)}"
        )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint with conversation context.
    
    Maintains conversation history and context.
    """
    
    try:
        # Generate conversation ID if new
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # For now, simple generation (Phase 4B will add conversation memory)
        response_text = model_service.generate(
            prompt=request.message,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        return ChatResponse(
            conversation_id=conversation_id,
            response=response_text,
            model_used="transformer_1gb_balanced",
            metadata={
                "prompt": request.message,
                "temperature": request.temperature,
                "max_length": request.max_length,
                "device": str(model_service.device)
            }
        )
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )

# ============================================================
# ERROR HANDLERS
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.RELOAD else "An error occurred"
        }
    )

if __name__ == "__main__":
    import uvicorn
    import uuid
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD
    )