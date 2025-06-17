from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import ollama

import logging
import time
import os
import re
import random
import json
import asyncio
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

# Enhanced Configuration 
class Settings:
    # API Settings
    ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
    MAX_HISTORY_LENGTH = int(os.environ.get("MAX_HISTORY_LENGTH", "12"))
    MAX_MESSAGE_LENGTH = int(os.environ.get("MAX_MESSAGE_LENGTH", "2000"))
    REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "15"))
    
    # Model Settings
    DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "mistral")
    TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.8"))
    TOP_P = float(os.environ.get("TOP_P", "0.9"))
    
    # Enhanced System Prompt for better continuity
    SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", """
    You are a compassionate therapeutic companion. Always maintain conversation continuity and context.
    
    CORE PRINCIPLES:
    - Build on previous conversation naturally
    - Reference what the user has shared before
    - Provide personalized, contextual responses
    - Never give generic advice - always relate to their specific situation
    - Keep responses conversational and flowing (3-4 sentences)
    
    CONVERSATION FLOW:
    - Acknowledge what they've shared previously
    - Validate their current feelings in context
    - Offer specific, situational guidance
    - Keep the conversation naturally progressing
    
    AVOID:
    - Generic responses that could apply to anyone
    - Ignoring previous conversation context
    - Sudden topic changes without acknowledgment
    - Robotic or template-like responses
    - Asking obvious questions when context is clear
    
    REMEMBER:
    - Each response should feel like a natural continuation
    - Reference their specific situation (job search, college completion, etc.)
    - Show you're listening and remembering their story
    - Provide evolving support as their situation develops
    """)

@lru_cache()
def get_settings():
    return Settings()

# Initialize FastAPI app
app = FastAPI(
    title="Mental Health Chatbot API",
    description="Contextual therapeutic support with conversation continuity",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced rate limiting
@app.middleware("http")
async def add_rate_limit(request: Request, call_next):
    client_ip = request.client.host
    current_time = time.time()
    
    if not hasattr(app, "rate_limit_store"):
        app.rate_limit_store = {}
    
    # Clean up old entries
    if current_time % 60 < 1:
        app.rate_limit_store = {k: v for k, v in app.rate_limit_store.items() if current_time - v["time"] < 60}
    
    if client_ip not in app.rate_limit_store:
        app.rate_limit_store[client_ip] = {"count": 0, "time": current_time}
    
    if current_time - app.rate_limit_store[client_ip]["time"] >= 60:
        app.rate_limit_store[client_ip] = {"count": 0, "time": current_time}
    
    app.rate_limit_store[client_ip]["count"] += 1
    
    if app.rate_limit_store[client_ip]["count"] > 40:
        return {"error": "Rate limit exceeded"}
    
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
    def validate_content(cls, v):
        if len(v) > 2000:
            return v[:2000]
        return v

class ChatRequest(BaseModel):
    message: str = Field(..., max_length=2000)
    history: List[Message] = Field(default_factory=list)
    model: Optional[str] = Field(default="mistral")
    temperature: Optional[float] = Field(default=0.8, ge=0.0, le=1.0)
    
    @validator("history")
    def validate_history_length(cls, v):
        return v[-12:]  # Increased for better context

class ChatResponse(BaseModel):
    response: Dict[str, Any]
    history: List[Message]
    model: str
    created_at: str

# Enhanced helper functions
def sanitize_user_input(text: str) -> str:
    """Enhanced sanitization while preserving emotional content"""
    text = text.replace("<script>", "").replace("</script>", "")
    text = re.sub(r'[<>&]', '', text)
    return text.strip()

def detect_mental_health_crisis(message: str) -> bool:
    """Enhanced crisis detection"""
    crisis_keywords = [
        "suicide", "kill myself", "end my life", "want to die", "harm myself",
        "cutting myself", "overdose", "jump off", "end it all", "no point living",
        "better off dead", "suicide plan", "hanging myself", "pills to die"
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in crisis_keywords)

def extract_conversation_context(history: List[Message]) -> Dict[str, Any]:
    """Extract ongoing conversation context for continuity"""
    context = {
        "ongoing_situation": None,
        "mentioned_topics": [],
        "emotional_progression": [],
        "key_details": [],
        "conversation_theme": None
    }
    
    # Return empty context if no history
    if not history or len(history) == 0:
        return context
    
    # Only analyze if there are at least 2 user messages (indicating ongoing conversation)
    user_messages = [msg.content.lower() for msg in history if msg.role == "user"]
    if len(user_messages) < 2:
        return context
    
    # Analyze recent messages for context
    recent_messages = [msg.content.lower() for msg in history[-6:] if msg.role == "user"]
    all_content = " ".join(recent_messages)
    
    # Extract ongoing situations
    if any(phrase in all_content for phrase in ["job", "unemployment", "work", "career", "applying"]):
        context["ongoing_situation"] = "job_search"
        context["conversation_theme"] = "career_anxiety"
    
    if any(phrase in all_content for phrase in ["college", "university", "graduated", "finished", "completed"]):
        context["mentioned_topics"].append("recent_graduation")
    
    if any(phrase in all_content for phrase in ["course", "study", "education", "learning"]):
        context["mentioned_topics"].append("education_decisions")
    
    if any(phrase in all_content for phrase in ["rejection", "failed", "didn't get", "no response"]):
        context["mentioned_topics"].append("rejection_experience")
    
    if any(phrase in all_content for phrase in ["decision", "choose", "can't decide", "torn between"]):
        context["mentioned_topics"].append("decision_making")
    
    # Extract emotional progression
    if any(word in all_content for word in ["tensed", "tense", "stressed"]):
        context["emotional_progression"].append("stress")
    
    if any(word in all_content for word in ["can't deal", "overwhelming", "dragging down"]):
        context["emotional_progression"].append("overwhelm")
    
    return context

def generate_contextual_response(message: str, conversation_context: Dict[str, Any], history: List[Message]) -> str:
    """Generate responses that maintain conversation continuity"""
    
    situation = conversation_context.get("ongoing_situation")
    topics = conversation_context.get("mentioned_topics", [])
    emotions = conversation_context.get("emotional_progression", [])
    
    message_lower = message.lower()
    
    # FIRST: Check if this is a completely new topic without any relevant history
    if not history or len(history) == 0:
        return generate_fresh_topic_response(message)
    
    # Context-aware responses based on conversation flow
    if situation == "job_search":
        if "can't decide" in message_lower and "course" in message_lower:
            return "I can see you're caught between two paths right now - continuing your education or diving into the job market. Given everything you've shared about the stress of job searching and the rejections you've faced, it's completely understandable that you're questioning whether more education might be the better route. This decision feels especially heavy because you're already feeling worn down by the job search process, and now you're wondering if you should step back and invest in more qualifications instead. Both paths have merit, but the real question is what feels right for your mental health and long-term goals right now."
        
        elif "failures" in message_lower or "dragging" in message_lower:
            return "Those rejections are really taking a toll on you, aren't they? When you keep putting yourself out there and facing 'no' after 'no', it starts to feel personal even though it's not. The weight of each rejection builds up, and I can hear how it's affecting your confidence and motivation. It's important to remember that job rejections are often about fit, timing, or internal factors you can't control - they're not a reflection of your worth or capabilities. Your resilience in continuing to try despite feeling dragged down shows real strength, even if it doesn't feel that way right now."
        
        elif "tensed" in message_lower and ("future" in message_lower or "completed" in message_lower):
            return "That post-graduation transition is hitting you hard, and the tension you're feeling is so valid. You've just completed this major milestone - college - and instead of feeling celebratory, you're faced with uncertainty about what comes next. The pressure to have it all figured out right after graduation is immense, and society doesn't acknowledge how anxiety-provoking this period really is. You're not behind or failing - you're in one of life's most challenging transition phases, and the stress you're experiencing is a normal response to genuine uncertainty about your future."
    
    # Decision-making context
    if "decision_making" in topics and "can't decide" in message_lower:
        return "This decision between pursuing another course or continuing the job search feels overwhelming because both options come with risks and unknowns. After experiencing the stress of job rejections, part of you might feel like more education could give you better chances, but another part probably worries about delaying your entry into the workforce even longer. The truth is, there's no universally 'right' choice here - both paths can lead to success. What matters most is which option aligns better with your current mental health needs and your long-term vision for yourself, even if that vision feels unclear right now."
    
    # Emotional progression responses
    if "overwhelm" in emotions and "stress" in emotions:
        return "I can see how the stress has been building and building for you - from the initial tension about your future after college, to the ongoing rejection cycle, and now this difficult decision about your next step. When we're in this state of chronic stress, even smaller decisions can feel monumentally difficult because our emotional resources are already stretched thin. It's like trying to think clearly when you're already carrying a heavy emotional load. Your feelings of being overwhelmed are completely justified given everything you're processing right now."
    
    # Only use contextual response if there's actual relevant history
    recent_user_messages = [msg.content for msg in history[-4:] if msg.role == "user"]
    if len(recent_user_messages) > 1:
        # Check if current message is related to previous topics
        current_topic_keywords = set(message_lower.split())
        previous_topics = set(" ".join(recent_user_messages[:-1]).lower().split())
        
        # Only provide contextual response if there's topic overlap
        if len(current_topic_keywords.intersection(previous_topics)) > 2:
            return f"I hear you continuing to work through these feelings, and I can see how this situation is really weighing on you. The combination of everything you've shared is creating stress and uncertainty. It's natural that you're feeling pulled in different directions. Take a breath and remember that you don't have to solve everything at once. What feels like the most pressing concern for you right now?"
    
    # If no relevant context, treat as fresh topic
    return generate_fresh_topic_response(message)

def generate_fresh_topic_response(message: str) -> str:
    """Generate responses for completely new topics without prior context"""
    message_lower = message.lower()
    
    # Social connection/friendship fears
    if any(phrase in message_lower for phrase in ["never make friends", "no friends", "can't make friends", "lonely", "friendless"]):
        return "That fear of never making friends is genuinely painful and isolating, and I can hear how much this worry is affecting you. Social connection is such a fundamental human need, and when we feel like we might not find it, it can be incredibly scary and overwhelming. The truth is that meaningful friendships often develop naturally over time through shared experiences, common interests, or simply being in the right place at the right moment. Your fear doesn't predict your future - many people who felt exactly like you do now have gone on to build beautiful, lasting friendships."
    
    # General emotional support for new topics
    if any(word in message_lower for word in ["worried", "afraid", "scared", "anxious", "concerned"]):
        return "I can hear the worry in your words, and whatever you're feeling concerned about is valid and understandable. When our minds get caught up in 'what if' scenarios about the future, it can feel overwhelming and scary. These worries often feel so real and immediate, even when they're about things that haven't happened yet. Remember that worrying about something doesn't make it more likely to occur - you're just experiencing the emotional weight of uncertainty, which is genuinely difficult to carry."
    
    # Default empathetic response for new topics
    return "What you're sharing sounds really difficult to carry, and I want you to know that your feelings are completely valid. Sometimes our minds can get caught in cycles of worry about the future, and it's exhausting to live with that kind of uncertainty. You're not alone in having these concerns - many people struggle with similar fears and anxieties. Take a deep breath and remember that you're stronger than you realize, even when everything feels uncertain."

def is_mental_health_related(message: str, conversation_history: List[Message] = None) -> bool:
    """Enhanced mental health topic detection with context awareness"""
    
    # If we have conversation history, be more lenient about topic detection
    if conversation_history:
        context = extract_conversation_context(conversation_history)
        if context["ongoing_situation"] or context["mentioned_topics"]:
            return True
    
    keywords = [
        # Emotions and feelings
        "sad", "anxious", "depressed", "stressed", "worried", "afraid", "angry",
        "lonely", "hopeless", "overwhelmed", "frustrated", "confused", "hurt",
        "doubt", "insecure", "uncertain", "nervous", "scared", "tensed", "tense",
        "disappointed", "heartbroken", "devastated", "empty", "numb", "lost",
        
        # Life situations
        "no job", "unemployed", "job search", "graduated", "breakup", "relationship",
        "family problems", "exam", "failed", "test", "presentation", "school",
        "university", "college", "work stress", "financial problems", "health issues",
        "decision", "can't decide", "choose", "future", "career", "course",
        
        # Mental health terms
        "anxiety", "depression", "therapy", "mental health", "counseling",
        "panic attack", "mood", "emotional", "feelings", "trauma", "grief",
        "self-esteem", "confidence", "self-worth", "identity crisis",
        
        # Help-seeking language
        "help", "support", "talk", "listen", "advice", "cope", "deal with",
        "struggle", "difficult time", "going through", "need someone", "alone",
        "don't know what to do", "can't handle", "overwhelmed", "lost"
    ]
    
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in keywords)

def get_empathetic_redirect(conversation_context: Dict[str, Any] = None) -> str:
    """Context-aware redirection for better conversation flow"""
    if conversation_context and conversation_context.get("ongoing_situation"):
        return "I can see we've been talking about some important things in your life. I'm here to continue supporting you through whatever you're facing. What's weighing on your mind right now?"
    
    responses = [
        "I'm here to provide emotional support and help you work through life's challenges. What's been weighing on your mind or heart lately? I'm here to listen and support you through whatever you're facing.",
        
        "I specialize in helping people navigate difficult emotions and life situations. Is there something troubling you or causing you stress that you'd like to talk about? I'm here to offer support and understanding.",
        
        "I'm designed to be your companion during tough times and emotional struggles. Whether you're dealing with stress, anxiety, relationship issues, or any other challenge, I'm here to listen and help. What's going on in your life right now that you'd like support with?"
    ]
    return random.choice(responses)

async def query_ollama(messages: List[Dict], model: str, temperature: float, top_p: float, user_message: str = "", conversation_history: List[Message] = None) -> Dict:
    """Enhanced Ollama query with better context awareness"""
    try:
        # Extract conversation context
        conversation_context = extract_conversation_context(conversation_history or [])
        
        # Add context-aware instruction
        if conversation_context["ongoing_situation"] or conversation_context["mentioned_topics"]:
            context_instruction = f"""
            CONVERSATION CONTEXT:
            - Ongoing situation: {conversation_context.get('ongoing_situation', 'None')}
            - Topics discussed: {', '.join(conversation_context.get('mentioned_topics', []))}
            - Emotional progression: {', '.join(conversation_context.get('emotional_progression', []))}
            
            IMPORTANT: Reference this context in your response. Build naturally on what's been discussed.
            Don't ignore previous conversation. Make your response feel like a continuation of an ongoing supportive conversation.
            Be specific to their situation, not generic.
            """
            messages.insert(-1, {"role": "system", "content": context_instruction})
        
        response = ollama.chat(
            model=model,
            messages=messages,
            options={
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": 200,
                "num_ctx": 3072,  # Increased context window
                "stop": ["User:", "Human:", "Assistant:"]
            }
        )
        
        if "message" in response and "content" in response["message"]:
            content = response["message"]["content"].strip()
            
            # Clean up response
            content = re.sub(r'^\s*\d+\.\s+', '', content, flags=re.MULTILINE)
            content = re.sub(r'^\s*[-•*]\s+', '', content, flags=re.MULTILINE)
            content = re.sub(r'\s+', ' ', content).strip()
            
            # If response is too generic or short, use contextual response
            if len(content) < 80 or any(generic in content.lower() for generic in ["it's important to", "you should", "try to", "consider"]):
                content = generate_contextual_response(user_message, conversation_context, conversation_history or [])
            
            return {"role": "assistant", "content": content}
        else:
            # Fallback to contextual response
            content = generate_contextual_response(user_message, conversation_context, conversation_history or [])
            return {"role": "assistant", "content": content}
    
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        conversation_context = extract_conversation_context(conversation_history or [])
        content = generate_contextual_response(user_message, conversation_context, conversation_history or [])
        return {"role": "assistant", "content": content}

async def stream_chat_response(request: ChatRequest, settings: Settings):
    """Enhanced streaming with better conversation continuity"""
    try:
        sanitized_message = sanitize_user_input(request.message)
        conversation_context = extract_conversation_context(request.history)
        
        # Check if mental health related (more lenient with context)
        if not is_mental_health_related(sanitized_message, request.history):
            response = get_empathetic_redirect(conversation_context)
            for char in response:
                yield json.dumps({"content": char}) + "\n"
                await asyncio.sleep(0.02)
            return
        
        # Crisis check
        is_crisis = detect_mental_health_crisis(sanitized_message)
        
        # Build enhanced message context
        messages = [{"role": "system", "content": settings.SYSTEM_PROMPT}]
        
        if is_crisis:
            messages.append({"role": "system", "content": "CRISIS SITUATION: User in immediate danger. Provide immediate emotional validation, then direct to emergency services (112 in India or 988 in US). Be calm, direct, and supportive."})
        
        # Add more history for better context
        history_dicts = [msg.dict() for msg in request.history[-8:]]
        messages.extend(history_dicts)
        messages.append({"role": "user", "content": sanitized_message})
        
        # Get contextual response
        response = await query_ollama(messages, request.model or settings.DEFAULT_MODEL, 
                                    request.temperature or settings.TEMPERATURE, settings.TOP_P, 
                                    sanitized_message, request.history)
        
        # Stream with natural pacing
        content = response["content"]
        for i, char in enumerate(content):
            yield json.dumps({"content": char}) + "\n"
            if char in '.!?':
                await asyncio.sleep(0.1)
            elif char in ',;':
                await asyncio.sleep(0.05)
            else:
                await asyncio.sleep(0.02)
        
    except Exception as e:
        logger.error(f"Stream error: {e}")
        error_message = "I'm still here with you. Let's continue our conversation - what's on your mind?"
        for char in error_message:
            yield json.dumps({"content": char}) + "\n"
            await asyncio.sleep(0.02)

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, settings: Settings = Depends(get_settings)):
    """Enhanced streaming endpoint with conversation continuity"""
    return StreamingResponse(
        stream_chat_response(request, settings),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest, 
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings)
):
    try:
        sanitized_message = sanitize_user_input(request.message)
        conversation_context = extract_conversation_context(request.history)
        
        # Enhanced topic check with context awareness
        if not is_mental_health_related(sanitized_message, request.history):
            response_text = get_empathetic_redirect(conversation_context)
            
            user_message = Message(role="user", content=sanitized_message)
            response_message = Message(role="assistant", content=response_text)
            new_history = (request.history + [user_message, response_message])[-12:]
            
            return ChatResponse(
                response={"role": "assistant", "content": response_text},
                history=new_history,
                model=request.model or settings.DEFAULT_MODEL,
                created_at=datetime.now().isoformat()
            )
        
        # Crisis detection
        is_crisis = detect_mental_health_crisis(sanitized_message)
        
        # Build enhanced therapeutic context
        messages = [{"role": "system", "content": settings.SYSTEM_PROMPT}]
        
        if is_crisis:
            messages.append({"role": "system", "content": "CRISIS SITUATION: User in immediate danger. Provide immediate emotional validation, then direct to emergency services (112 in India or 988 in US). Be calm, direct, and supportive."})
        
        # Add more conversation history for better continuity
        history_dicts = [msg.dict() for msg in request.history[-8:]]
        messages.extend(history_dicts)
        messages.append({"role": "user", "content": sanitized_message})
        
        # Get contextual therapeutic response
        response = await query_ollama(messages, request.model or settings.DEFAULT_MODEL, 
                                    request.temperature or settings.TEMPERATURE, settings.TOP_P, 
                                    sanitized_message, request.history)
        
        # Build response with enhanced history tracking
        user_message = Message(role="user", content=sanitized_message)
        response_message = Message(role="assistant", content=response["content"])
        new_history = (request.history + [user_message, response_message])[-12:]
        
        return ChatResponse(
            response=response,
            history=new_history,
            model=request.model or settings.DEFAULT_MODEL,
            created_at=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="I'm still here to help you. Please try reaching out again.")

@app.get("/health")
def health_check():
    """Health check with model verification"""
    try:
        models = ollama.list()
        return {"status": "healthy", "available_models": [model["name"] for model in models.get("models", [])]}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/")
def root():
    return {
        "message": "Mental Health Therapeutic Chatbot API", 
        "version": "3.0.0",
        "description": "Contextual therapeutic responses with conversation continuity",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)