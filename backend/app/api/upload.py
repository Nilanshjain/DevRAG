"""
File Upload API Endpoints
Handles document uploads, validation, and processing
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import os
import shutil
from pathlib import Path
import hashlib
from datetime import datetime

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