"""
File Upload API Endpoints
Handles document uploads, validation, and processing
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List
from sqlalchemy.orm import Session
import os
import shutil
from pathlib import Path
import hashlib
from datetime import datetime
import logging

from app.core.database import get_db
from app.core.vector_storage import PostgreSQLVectorStorage
from app.services.text_extraction import TextExtractor
from app.services.chunking import DocumentChunker
from app.services.embeddings import get_embedding_service

logger = logging.getLogger(__name__)

# Create a router - this groups related endpoints together
# Think of it like a chapter in a book
router = APIRouter(
    prefix="/api/v1",  # All routes in this file start with /api/v1
    tags=["upload"]    # Groups endpoints in documentation
)

# Configuration constants (in production, these would be in a config file)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB in bytes
ALLOWED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".py", ".js", ".ts", ".java",
    ".c", ".cpp", ".html", ".css", ".json", ".yaml", ".yml"
}
UPLOAD_DIR = Path("uploads")  # Where we store uploaded files

# Create uploads directory if it doesn't exist
UPLOAD_DIR.mkdir(exist_ok=True)


def validate_file(file: UploadFile) -> None:
    """
    Validate uploaded file for security and compatibility

    Args:
        file: The uploaded file from FastAPI

    Raises:
        HTTPException: If file fails validation
    """
    # Check if file was actually provided
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Check file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_extension} not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Check file size (we'll check this after reading the file)
    # In production, you'd want to check this while streaming to avoid memory issues


def generate_unique_filename(original_filename: str) -> str:
    """
    Generate a unique filename to avoid collisions

    Args:
        original_filename: The original uploaded filename

    Returns:
        A unique filename with timestamp and hash
    """
    # Get file extension
    extension = Path(original_filename).suffix

    # Create a hash of the original filename
    name_hash = hashlib.md5(original_filename.encode()).hexdigest()[:8]

    # Add timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Combine into unique filename
    return f"{timestamp}_{name_hash}{extension}"


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a single file for processing

    This endpoint:
    1. Receives a file from the user
    2. Validates it for safety
    3. Saves it to disk
    4. Returns file information

    Args:
        file: The uploaded file (automatically parsed by FastAPI)

    Returns:
        JSON response with file details and upload status
    """
    try:
        # Step 1: Validate the file
        validate_file(file)

        # Step 2: Generate unique filename
        unique_filename = generate_unique_filename(file.filename)
        file_path = UPLOAD_DIR / unique_filename

        # Step 3: Save file to disk
        # We read the file content into memory first
        contents = await file.read()

        # Check file size after reading
        file_size = len(contents)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
            )

        # Write file to disk
        with open(file_path, "wb") as f:
            f.write(contents)

        # Step 4: Prepare response
        response = {
            "status": "success",
            "message": "File uploaded successfully",
            "file": {
                "original_name": file.filename,
                "saved_name": unique_filename,
                "size": file_size,
                "size_mb": round(file_size / (1024 * 1024), 2),
                "type": file.content_type,
                "extension": Path(file.filename).suffix,
                "upload_time": datetime.now().isoformat()
            }
        }

        return JSONResponse(content=response, status_code=200)

    except HTTPException:
        # Re-raise HTTP exceptions (validation errors)
        raise
    except Exception as e:
        # Log error in production
        print(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during upload")
    finally:
        # Always close the file
        await file.close()


@router.post("/upload/multiple")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """
    Upload multiple files at once

    Args:
        files: List of uploaded files

    Returns:
        JSON response with details for each file
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed at once")

    results = []

    for file in files:
        try:
            # Validate file
            validate_file(file)

            # Generate unique filename
            unique_filename = generate_unique_filename(file.filename)
            file_path = UPLOAD_DIR / unique_filename

            # Save file
            contents = await file.read()

            # Check size
            if len(contents) > MAX_FILE_SIZE:
                results.append({
                    "original_name": file.filename,
                    "status": "failed",
                    "error": "File too large"
                })
                continue

            # Write to disk
            with open(file_path, "wb") as f:
                f.write(contents)

            results.append({
                "original_name": file.filename,
                "saved_name": unique_filename,
                "status": "success",
                "size_mb": round(len(contents) / (1024 * 1024), 2)
            })

        except Exception as e:
            results.append({
                "original_name": file.filename,
                "status": "failed",
                "error": str(e)
            })
        finally:
            await file.close()

    # Count successes and failures
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = len(results) - success_count

    return {
        "message": f"Processed {len(files)} files",
        "success_count": success_count,
        "failed_count": failed_count,
        "results": results
    }


@router.get("/files")
async def list_uploaded_files():
    """
    List all uploaded files

    Returns:
        JSON response with list of uploaded files
    """
    try:
        files = []
        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "filename": file_path.name,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "uploaded_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "extension": file_path.suffix
                })

        # Sort by upload time (newest first)
        files.sort(key=lambda x: x["uploaded_at"], reverse=True)

        return {
            "total_files": len(files),
            "files": files
        }
    except Exception as e:
        print(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail="Error listing files")


@router.delete("/files/{filename}")
async def delete_file(filename: str):
    """
    Delete an uploaded file

    Args:
        filename: Name of the file to delete

    Returns:
        JSON response confirming deletion
    """
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        file_path.unlink()  # Delete the file
        return {
            "status": "success",
            "message": f"File {filename} deleted successfully"
        }
    except Exception as e:
        print(f"Error deleting file: {str(e)}")
        raise HTTPException(status_code=500, detail="Error deleting file")


@router.post("/process")
async def process_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload and process a document through the complete RAG pipeline

    This endpoint:
    1. Uploads the file
    2. Extracts text
    3. Chunks the document
    4. Generates embeddings
    5. Stores in vector database

    Returns complete processing status and metadata
    """
    file_path = None

    try:
        # Step 1: Validate and save file
        validate_file(file)
        unique_filename = generate_unique_filename(file.filename)
        file_path = UPLOAD_DIR / unique_filename

        contents = await file.read()
        file_size = len(contents)

        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
            )

        with open(file_path, "wb") as f:
            f.write(contents)

        logger.info(f"File saved: {unique_filename}")

        # Step 2: Extract text
        extractor = TextExtractor()
        extraction_result = extractor.extract_text(file_path)
        text_content = extraction_result["text"]

        if not text_content or len(text_content.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Could not extract any text from document. The file may be empty or corrupted."
            )

        logger.info(f"Extracted {len(text_content)} characters from {file.filename}")

        # Step 3: Chunk the document
        chunker = DocumentChunker(min_chunk_size=50)  # Lower minimum for small documents
        chunks = chunker.chunk_text(text_content, strategy="smart")

        # If no chunks created but we have text, create a single chunk
        if not chunks and text_content:
            chunks = [chunker._chunk_fixed_size(text_content, file.filename)[0]] if chunker._chunk_fixed_size(text_content, file.filename) else []

        # If still no chunks, treat entire text as one chunk
        if not chunks and text_content:
            from app.services.chunking import Chunk
            chunks = [Chunk(
                text=text_content,
                chunk_id=f"{file.filename}_chunk_0",
                source_file=file.filename,
                chunk_index=0,
                char_count=len(text_content),
                word_count=len(text_content.split()),
                start_char=0,
                end_char=len(text_content),
                metadata={"strategy": "single_chunk"}
            )]

        chunk_texts = [chunk.text for chunk in chunks]
        logger.info(f"Created {len(chunks)} chunks")

        # Step 4: Generate embeddings
        try:
            embedding_service = get_embedding_service()
            embeddings = embedding_service.generate_embeddings(chunk_texts)
            logger.info(f"Generated {len(embeddings)} embeddings")
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate embeddings: {str(e)}"
            )

        # Step 5: Store in vector database
        vector_store = PostgreSQLVectorStorage(db)
        document_ids = vector_store.store_document_with_embeddings(
            filename=file.filename,
            content=text_content,
            chunks=chunk_texts,
            embeddings=embeddings
        )

        logger.info(f"Stored document with {len(document_ids)} chunks")

        # Step 6: Check similarity with previous documents
        similar_to_previous = False
        try:
            # Get all other documents
            from app.core.vector_storage import VectorDocument
            other_docs = db.query(VectorDocument).filter(
                VectorDocument.filename != file.filename,
                VectorDocument.embedding_json.isnot(None)
            ).all()

            if other_docs and embeddings:
                # Compare first chunk of new doc with other docs
                new_embedding = embeddings[0]
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np

                for other_doc in other_docs[:5]:  # Check last 5 documents
                    other_embedding = other_doc.get_embedding()
                    if other_embedding:
                        similarity = cosine_similarity(
                            np.array(new_embedding).reshape(1, -1),
                            np.array(other_embedding).reshape(1, -1)
                        )[0][0]

                        if similarity > 0.7:  # High similarity threshold
                            similar_to_previous = True
                            logger.info(f"Document {file.filename} is similar to {other_doc.filename} (similarity: {similarity:.2f})")
                            break
        except Exception as e:
            logger.warning(f"Similarity check failed: {e}")

        # Return success response
        return {
            "status": "success",
            "message": "Document processed and stored successfully",
            "file": {
                "original_name": file.filename,
                "saved_name": unique_filename,
                "size_mb": round(file_size / (1024 * 1024), 2),
            },
            "processing": {
                "characters_extracted": len(text_content),
                "word_count": extraction_result.get("word_count", 0),
                "chunks_created": len(chunks),
                "chunks_stored": len(document_ids),
                "embedding_dimension": len(embeddings[0]) if embeddings else 0
            },
            "database_ids": document_ids,
            "similar_to_previous": similar_to_previous
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        if file_path and file_path.exists():
            file_path.unlink()
        raise

    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        if file_path and file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(e)}"
        )

    finally:
        await file.close()