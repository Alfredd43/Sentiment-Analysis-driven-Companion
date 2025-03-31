from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import ollama
import logging
import time
import os
import re
from datetime import datetime
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("chatbot.log")
    ]
)
logger = logging.getLogger(__name__)

# Configuration 
class Settings:
    # API Settings
    ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
    MAX_HISTORY_LENGTH = int(os.environ.get("MAX_HISTORY_LENGTH", "15"))
    MAX_MESSAGE_LENGTH = int(os.environ.get("MAX_MESSAGE_LENGTH", "5000"))
    REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "30"))
    
    # Model Settings
    DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "mistral")
    TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
    TOP_P = float(os.environ.get("TOP_P", "0.9"))
    
    # Updated System Prompt for conversational, human-like responses
    SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", """
    You are a compassionate AI therapist who speaks directly to the heart.
    IMPORTANT: Do NOT repeat the user's message or pretend you know the people they mention.
    Always respond as a therapist would to someone seeking advice about the situation.
    Provide complete responses that fully address the user's concerns.
    Write in a natural, conversational style - never use numbered lists, bullet points, or structured formats.
    Use natural transitions between ideas and speak as if you were a professional therapist.
    Focus on emotional validation, empathy, and providing helpful perspectives.
    Never provide medical advice or diagnosis. For crisis situations, direct to emergency services.
    """)

@lru_cache()
def get_settings():
    return Settings()

# Initialize FastAPI app
app = FastAPI(
    title="Mental Health Chatbot API",
    description="API for a mental health support chatbot powered by Mistral via Ollama",
    version="1.0.0"
)

# Add CORS middleware with updated settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware
@app.middleware("http")
async def add_rate_limit(request: Request, call_next):
    # Simple IP-based rate limiting
    client_ip = request.client.host
    current_time = time.time()
    
    # This could be replaced with Redis for production
    if not hasattr(app, "rate_limit_store"):
        app.rate_limit_store = {}
    
    # Clean up old entries (simple expiration)
    app.rate_limit_store = {k: v for k, v in app.rate_limit_store.items() if current_time - v["last_reset"] < 60}
    
    if client_ip not in app.rate_limit_store:
        app.rate_limit_store[client_ip] = {"count": 0, "last_reset": current_time}
    
    # Reset counter after a minute
    if current_time - app.rate_limit_store[client_ip]["last_reset"] >= 60:
        app.rate_limit_store[client_ip] = {"count": 0, "last_reset": current_time}
    
    app.rate_limit_store[client_ip]["count"] += 1
    
    # Limit to 20 requests per minute per IP
    if app.rate_limit_store[client_ip]["count"] > 20:
        return {"error": "Rate limit exceeded. Please try again later."}
    
    response = await call_next(request)
    return response

# Request and Response Models
class Message(BaseModel):
    role: str
    content: str
    
    @validator("role")
    def validate_role(cls, v):
        if v not in ["system", "user", "assistant"]:
            raise ValueError("Role must be 'system', 'user', or 'assistant'")
        return v
    
    @validator("content")
    def validate_content(cls, v, values):
        settings = get_settings()
        if len(v) > settings.MAX_MESSAGE_LENGTH:
            # Instead of raising an error, truncate the message to the maximum allowed length
            logger.warning(f"Message exceeds {settings.MAX_MESSAGE_LENGTH} characters. Truncating.")
            return v[:settings.MAX_MESSAGE_LENGTH]
        return v

class ChatRequest(BaseModel):
    message: str = Field(..., max_length=5000)
    history: List[Message] = Field(default_factory=list)
    model: Optional[str] = Field(default="mistral")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)
    
    @validator("history")
    def validate_history_length(cls, v):
        settings = get_settings()
        if len(v) > settings.MAX_HISTORY_LENGTH:
            # Only keep the most recent messages if history is too long
            return v[-settings.MAX_HISTORY_LENGTH:]
        return v

class ChatResponse(BaseModel):
    response: Dict[str, Any]
    history: List[Message]
    model: str
    created_at: str

# Helper functions
def sanitize_user_input(text: str) -> str:
    """Basic sanitization of user input"""
    # Remove any potentially harmful characters or patterns
    sanitized = text.replace("<script>", "").replace("</script>", "")
    return sanitized

def modify_user_message(text: str) -> str:
    """Add 'in short' to the end of the user message if not already present"""
    # Check if "in short" is already in the message
    if "in short like 3 lines" not in text.lower():
        # Check if the message ends with punctuation
        if text.rstrip()[-1] in ['.', '?', '!']:
            modified = text.rstrip() + " In short like 3 lines."
        else:
            modified = text.rstrip() + " in shortlike 3 lines."
        
        logger.info(f"Modified user message: '{text}' -> '{modified}'")
        return modified
    return text

def detect_mental_health_crisis(message: str) -> bool:
    """Basic detection of potential crisis situations"""
    crisis_keywords = [
        "suicide", "kill myself", "end my life", "don't want to live", 
        "harm myself", "hurt myself", "die", "death"
    ]
    
    message_lower = message.lower()
    for keyword in crisis_keywords:
        if keyword in message_lower:
            return True
    
    return False

def log_anonymized_interaction(user_message: str, assistant_response: str):
    """Log interactions without personal identifiers for quality monitoring"""
    # This would typically write to a database in production
    # For now, just log to file with timestamp
    logger.info(
        f"INTERACTION: User: {user_message[:30]}... | Assistant: {assistant_response[:30]}..."
    )

def ensure_conversational_format(text: str) -> str:
    """Ensure response is in a conversational format without numbered points"""
    # Remove numbered points format (1., 2., etc.)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Remove bullet points if present
    text = re.sub(r'^\s*[-•*]\s+', '', text, flags=re.MULTILINE)
    
    # Convert multiple newlines to double newlines for paragraph separation
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Ensure paragraphs are properly formatted
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    # Join with double newlines to create proper paragraph separation
    return '\n\n'.join(paragraphs)

def ensure_complete_response(text: str) -> str:
    """Ensure response is complete without unnecessary truncation"""
    # Trim leading/trailing whitespace
    text = text.strip()
    
    # Check if text is empty or too short
    if not text or len(text) < 5:
        return "I hear you. Your feelings matter."
    
    # Ensure response ends with proper punctuation
    if text[-1] not in ['.', '!', '?']:
        text += '.'
    
    # We still need to make sure responses don't exceed the MAX_MESSAGE_LENGTH
    settings = get_settings()
    if len(text) > settings.MAX_MESSAGE_LENGTH - 100:  # Leave some margin
        # Try to find a good sentence break
        last_period = max(
            text.rfind('. ', 0, settings.MAX_MESSAGE_LENGTH - 100),
            text.rfind('! ', 0, settings.MAX_MESSAGE_LENGTH - 100),
            text.rfind('? ', 0, settings.MAX_MESSAGE_LENGTH - 100)
        )
        
        if last_period > 100:  # Make sure we have substantial content
            return text[:last_period+1]
        else:
            # If no good break found, just truncate with some margin
            return text[:settings.MAX_MESSAGE_LENGTH - 103] + "..."
    
    return text

async def query_ollama(messages: List[Dict], model: str, temperature: float, top_p: float) -> Dict:
    """Send a conversation history to the locally running Ollama model and return the response."""
    settings = get_settings()
    
    try:
        start_time = time.time()
        
        # Add contextual instruction for conversational style
        context_instruction = {
             "role": "system", 
             "content": "You are a compassionate AI therapist providing guidance. Do NOT repeat or rephrase the user's message as if it's your own experience. Do NOT pretend you know the people mentioned. Instead, provide supportive therapeutic responses and advice about the situation described. Respond in a conversational, naturally flowing style. Be brief and direct for simple questions, more detailed for complex emotional issues."
             }
        messages.append(context_instruction)
        
        response = ollama.chat(
            model=model,
            messages=messages,
            options={
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": 800,  # Significantly increased to allow longer, complete responses
            }
        )
        elapsed = time.time() - start_time
        logger.info(f"Ollama response time: {elapsed:.2f}s")
        
        if "message" in response and "content" in response["message"]:
            # Get the response and ensure it's complete
            content = response["message"]["content"]
            processed_content = ensure_conversational_format(ensure_complete_response(content))
            
            # Log the original and processed content
            logger.info(f"Original: {content[:100]}...")
            logger.info(f"Processed: {processed_content[:100]}...")
            
            return {"role": "assistant", "content": processed_content}
        else:
            logger.error(f"Unexpected response format: {response}")
            return {"role": "assistant", "content": "I hear you and I'm here for you."}
    
    except TimeoutError:
        logger.error("Request to Ollama timed out")
        return {"role": "assistant", "content": "I'm here for you. Let's take a moment and try again."}
    
    except ConnectionError:
        logger.error("Connection to Ollama failed")
        return {"role": "assistant", "content": "I'm here for you, but need a moment to reconnect."}
    
    except Exception as e:
        logger.error(f"Error querying Ollama: {e}")
        return {"role": "assistant", "content": "Your feelings matter. Let's try again in a moment."}

# API Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest, 
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings)
):
    try:
        # Sanitize user input
        sanitized_message = sanitize_user_input(request.message)
        
        # Modify user message to append "in short" if not already present
        modified_message = modify_user_message(sanitized_message)
        
        # Check for potential crisis
        is_crisis = detect_mental_health_crisis(modified_message)
        
        # Construct message history
        messages = [
            {"role": "system", "content": settings.SYSTEM_PROMPT}
        ]
        
        # Add crisis intervention if detected
        if is_crisis:
            crisis_prompt = """
            The user may be in crisis. Respond with compassion but direct them to immediate help.
            Speak in a warm, caring voice like a human therapist would. Tell them: "I hear your pain. Please call a crisis hotline right now at 988 or text HOME to 741741."
            Do not use any numbered points or bullet points - respond as a real therapist would in a crisis situation.
            """
            messages.append({"role": "system", "content": crisis_prompt})
        
        # Convert Pydantic models to dictionaries for the history
        history_dicts = [msg.dict() for msg in request.history]
        messages.extend(history_dicts)
        
        # Add the modified user message
        messages.append({"role": "user", "content": modified_message})
        
        # Get model parameters
        model = request.model or settings.DEFAULT_MODEL
        temperature = request.temperature or settings.TEMPERATURE
        
        # Get response from Ollama
        response = await query_ollama(messages, model, temperature, settings.TOP_P)
        
        # Ensure we have a valid response
        if not response["content"] or len(response["content"]) < 5:
            response["content"] = "I'm here with you. Your feelings are valid."
        
        # Log the interaction (anonymized)
        background_tasks.add_task(
            log_anonymized_interaction,
            modified_message,
            response["content"]
        )
        
        # Convert response to Pydantic model
        response_message = Message(role=response["role"], content=response["content"])
        
        # Important: For the history, use the original message, not the modified one
        # This keeps the conversation natural for the user
        user_message = Message(role="user", content=sanitized_message)
        new_history = request.history + [user_message, response_message]
        
        # Create response object
        return ChatResponse(
            response=response,
            history=new_history,
            model=model,
            created_at=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check if Ollama is available
        ollama.list()
        return {"status": "healthy", "ollama": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

@app.get("/")
def root():
    return {
        "message": "Mental Health Chatbot API is running!",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)