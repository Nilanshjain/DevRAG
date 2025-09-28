"""
Development server runner
Run this file to start the FastAPI development server
"""

import uvicorn

if __name__ == "__main__":
    # Start the development server
    uvicorn.run(
        "app.main:app",  # Path to our FastAPI app instance
        host="0.0.0.0",  # Listen on all network interfaces
        port=8000,       # Port number - our API will be at localhost:8000
        reload=True      # Auto-restart when code changes (development only)
    )