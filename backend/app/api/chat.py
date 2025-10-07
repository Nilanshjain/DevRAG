"""
Chat API Endpoints
Handles AI-powered chat with document context retrieval (RAG)
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
import logging
import os

from app.core.database import get_db
from app.core.vector_storage import PostgreSQLVectorStorage
from app.services.embeddings import get_embedding_service
from app.core.config import settings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1/chat",
    tags=["chat"]
)


# Request/Response models
class ChatMessage(BaseModel):
    """A single chat message"""
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    """Request for chat completion"""
    message: str
    conversation_history: Optional[List[ChatMessage]] = []
    max_context_chunks: int = 3
    use_context: bool = True
    document_filter: Optional[str] = None  # Filter context by specific document filename


class ChatResponse(BaseModel):
    """Response from chat"""
    response: str
    context_used: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    suggested_questions: Optional[List[str]] = []  # Follow-up questions for user


class ContextSearchRequest(BaseModel):
    """Request for context search only"""
    query: str
    limit: int = 5
    similarity_threshold: float = 0.5


@router.post("/message", response_model=ChatResponse)
async def chat_message(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Send a message and get an AI response with document context

    This endpoint:
    1. Takes user's message
    2. Finds relevant context from uploaded documents
    3. Generates AI response using that context
    4. Returns response with sources
    """
    try:
        # Step 1: Get relevant context from vector database
        context_chunks = []

        if request.use_context:
            try:
                # Generate embedding for the query
                embedding_service = get_embedding_service()
                query_embedding = embedding_service.generate_embedding(request.message)

                # Search for similar documents
                vector_store = PostgreSQLVectorStorage(db)
                results = vector_store.similarity_search(
                    query_embedding=query_embedding,
                    limit=request.max_context_chunks,
                    similarity_threshold=0.1,  # Very low threshold to catch more results
                    filename_filter=request.document_filter  # Filter by specific document if provided
                )

                # Format context
                for doc, similarity in results:
                    context_chunks.append({
                        "text": doc.chunk_text,
                        "filename": doc.filename,
                        "similarity": similarity,
                        "chunk_index": doc.chunk_index
                    })

                logger.info(f"Found {len(context_chunks)} relevant context chunks")

            except Exception as e:
                logger.warning(f"Context retrieval failed: {e}")
                # Continue without context rather than failing

        # Step 2: Build prompt with context
        system_prompt = """You are an expert AI assistant that provides in-depth analysis and explanations.

CRITICAL INSTRUCTIONS:

Priority Guidelines:
   PRIORITIZE information from the provided document context above all
   When the document has relevant information, clearly cite it and base your answer on it
   If the document doesn't have enough information, supplement with your knowledge while clearly stating: "Based on the document and my understanding..."
   NEVER say "I don't have this information" - always try to provide a helpful answer

Content Depth Requirements:
   Explain concepts thoroughly with comprehensive detail
   Provide context and background when relevant
   Analyze implications and significance
   Connect related pieces of information
   Break down complex information into understandable parts
   Be comprehensive yet clear in your explanations

FORMATTING RULES - STRICTLY FOLLOW THESE:

DO NOT USE:
   ❌ Asterisks (*) - NEVER use these
   ❌ Hashtags (#) - NEVER use these
   ❌ Single dashes for bullets (-)
   ❌ Markdown formatting symbols

DO USE:
   ✓ Plain text section headings (no symbols before them)
   ✓ Numbered lists: 1. 2. 3. for sequential steps
   ✓ Simple paragraph structure
   ✓ Clear line breaks between sections
   ✓ Professional business document style

Example of CORRECT formatting:
"
Overview
The document discusses three main concepts.

Key Points
1. First important concept with detailed explanation
2. Second important concept with analysis
3. Third important concept with implications

Analysis
This demonstrates the significance because...
"

Tone:
   Professional and authoritative
   Clear and accessible
   Analytical and insightful"""

        # Add context to prompt
        context_text = ""
        if context_chunks:
            context_text = "\n\nRelevant Context:\n"
            for i, chunk in enumerate(context_chunks, 1):
                context_text += f"\n[Document: {chunk['filename']}, Chunk {chunk['chunk_index']}]\n{chunk['text']}\n"

        # Step 3: Generate response using AI (Gemini or OpenAI)
        try:
            # Try Gemini first, then OpenAI
            gemini_key = settings.gemini_api_key or settings.google_api_key
            openai_key = settings.openai_api_key

            if not gemini_key and not openai_key:
                # Fallback: Return context-based response without AI
                if context_chunks:
                    response_text = f"I found relevant information in your documents:\n\n"
                    for chunk in context_chunks:
                        response_text += f"From {chunk['filename']}:\n{chunk['text'][:200]}...\n\n"
                    response_text += "\n(Note: No AI API key configured. Set GEMINI_API_KEY or OPENAI_API_KEY to enable AI responses.)"
                else:
                    response_text = "No relevant information found in your documents. (Note: No AI API key configured.)"

                return ChatResponse(
                    response=response_text,
                    context_used=context_chunks,
                    metadata={
                        "context_chunks_used": len(context_chunks),
                        "ai_enabled": False
                    }
                )

            # Use Gemini if available
            if gemini_key:
                import google.generativeai as genai

                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel('gemini-2.5-pro')

                # Build prompt with context
                full_prompt = system_prompt + context_text + f"\n\nUser Question: {request.message}"

                # Add conversation history
                if request.conversation_history:
                    full_prompt += "\n\nConversation History:\n"
                    for msg in request.conversation_history[-5:]:
                        full_prompt += f"{msg.role}: {msg.content}\n"

                # Call Gemini
                response = model.generate_content(full_prompt)
                response_text = response.text

                # Generate follow-up questions if this is first question (no history)
                suggested_questions = []
                if not request.conversation_history or len(request.conversation_history) == 0:
                    try:
                        followup_prompt = f"""Based on the document context and the user's question: "{request.message}", generate 3 relevant follow-up questions that the user might want to ask the chatbot to:
1. Explore the document deeper
2. Confirm understanding
3. Discover related content

Format each question from the user's perspective asking the chatbot (e.g., "What is...", "Can you explain...", "How does...", "Why does...").
Each question should be a complete, natural question the user would type.
Return ONLY the 3 questions, one per line, without numbering or extra text."""

                        followup_response = model.generate_content(followup_prompt + context_text)
                        questions = [q.strip() for q in followup_response.text.strip().split('\n') if q.strip()]
                        suggested_questions = questions[:3]  # Take first 3
                    except Exception as e:
                        logger.warning(f"Failed to generate follow-up questions: {e}")

                return ChatResponse(
                    response=response_text,
                    context_used=context_chunks,
                    suggested_questions=suggested_questions,
                    metadata={
                        "context_chunks_used": len(context_chunks),
                        "ai_enabled": True,
                        "model": "gemini-2.5-pro",
                        "provider": "google"
                    }
                )

            # Fallback to OpenAI
            elif openai_key:
                import openai

                client = openai.OpenAI(api_key=openai_key)

                messages = [
                    {"role": "system", "content": system_prompt + context_text}
                ]

                # Add conversation history
                for msg in request.conversation_history[-5:]:
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })

                # Add current message
                messages.append({
                    "role": "user",
                    "content": request.message
                })

                # Call OpenAI
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500
                )

                response_text = completion.choices[0].message.content

                return ChatResponse(
                    response=response_text,
                    context_used=context_chunks,
                    metadata={
                        "context_chunks_used": len(context_chunks),
                        "ai_enabled": True,
                        "model": "gpt-3.5-turbo",
                        "provider": "openai",
                        "tokens_used": completion.usage.total_tokens
                    }
                )

        except Exception as e:
            logger.error(f"AI response generation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate AI response: {str(e)}"
            )

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}"
        )


@router.post("/search-context")
async def search_context(
    request: ContextSearchRequest,
    db: Session = Depends(get_db)
):
    """
    Search for relevant context without generating AI response

    Useful for:
    - Debugging context retrieval
    - Finding relevant documents
    - Testing similarity search
    """
    try:
        # Generate embedding for query
        embedding_service = get_embedding_service()
        query_embedding = embedding_service.generate_embedding(request.query)

        # Search vector database
        vector_store = PostgreSQLVectorStorage(db)
        results = vector_store.similarity_search(
            query_embedding=query_embedding,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold
        )

        # Format results
        context_chunks = []
        for doc, similarity in results:
            context_chunks.append({
                "text": doc.chunk_text,
                "filename": doc.filename,
                "similarity": float(similarity),
                "chunk_index": doc.chunk_index,
                "document_id": doc.id
            })

        return {
            "query": request.query,
            "results_found": len(context_chunks),
            "context_chunks": context_chunks
        }

    except Exception as e:
        logger.error(f"Context search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Context search failed: {str(e)}"
        )


@router.get("/health")
async def chat_health_check():
    """Check if chat service is ready"""
    try:
        # Check embedding service
        embedding_service = get_embedding_service()
        embedding_info = embedding_service.get_embedding_info()

        # Check AI providers
        gemini_available = bool(settings.gemini_api_key or settings.google_api_key)
        openai_available = bool(settings.openai_api_key)

        return {
            "status": "healthy",
            "embedding_service": embedding_info,
            "ai_providers": {
                "gemini": gemini_available,
                "openai": openai_available
            },
            "features": {
                "context_retrieval": True,
                "ai_responses": gemini_available or openai_available
            }
        }

    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }
