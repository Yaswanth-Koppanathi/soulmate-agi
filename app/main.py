import os
import json
from typing import List, Dict, Any
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model.model_handler import SoulMateModel
from app.memory.memory_handler import MemorySystem
from app.emotion.emotion_analyzer import EmotionAnalyzer
from app.learning.learning_engine import IncrementalLearner

# Initialize FastAPI app
app = FastAPI(title="SoulMate.AGI API")

# Initialize components
model = SoulMateModel("gpt2")  # For hackathon, use a smaller model
memory = MemorySystem()
emotion = EmotionAnalyzer()
learner = IncrementalLearner("gpt2")  # Same model as base

# Data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Request/Response models
class MessageRequest(BaseModel):
    message: str
    user_id: str = "default"

class ProfileUpdateRequest(BaseModel):
    trait: str
    value: Any
    user_id: str = "default"

class LearningRequest(BaseModel):
    user_id: str = "default"
    days_to_simulate: int = 30

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    print("SoulMate.AGI is starting up...")
    
    # Try to load saved data
    user_dir = os.path.join(DATA_DIR, "default")
    if os.path.exists(user_dir):
        try:
            memory.load_memory(os.path.join(user_dir, "memory"))
            print("Memory data loaded successfully")
        except Exception as e:
            print(f"Error loading memory: {e}")

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "Welcome to SoulMate.AGI API",
        "version": "0.1.0",
        "status": "active"
    }

@app.post("/