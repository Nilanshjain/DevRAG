# FastAPI Complete Guide

## What is FastAPI?
FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints. Created by Sebastián Ramírez in 2018, it's one of the fastest Python frameworks available, comparable to NodeJS and Go.

## Key Features
- **Fast**: Very high performance, on par with NodeJS and Go
- **Fast to code**: Increases development speed by 200-300%
- **Fewer bugs**: Reduces human errors by 40%
- **Intuitive**: Great editor support with auto-completion
- **Easy**: Designed to be easy to learn and use
- **Standards-based**: Based on OpenAPI and JSON Schema

## Installation
```bash
pip install fastapi
pip install uvicorn[standard]
```

## Basic Application
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

Run with:
```bash
uvicorn main:app --reload
```

## Path Parameters
Path parameters are captured from the URL:
```python
@app.get("/users/{user_id}")
async def read_user(user_id: int):
    return {"user_id": user_id}

# With type validation
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    # item_id is automatically validated as integer
    return {"item_id": item_id}
```

## Query Parameters
Query parameters come after the `?` in the URL:
```python
@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10):
    # /items/?skip=0&limit=10
    return {"skip": skip, "limit": limit}

# Optional parameters
@app.get("/search/")
async def search(q: str = None, page: int = 1):
    if q:
        return {"query": q, "page": page}
    return {"message": "No query provided"}
```

## Request Body with Pydantic
Use Pydantic models for request validation:
```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

@app.post("/items/")
async def create_item(item: Item):
    item_dict = item.dict()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict
```

## Response Models
Define response structure:
```python
class UserResponse(BaseModel):
    id: int
    username: str
    email: str

@app.post("/users/", response_model=UserResponse)
async def create_user(user: User):
    # Password won't be included in response
    return user
```

## Dependency Injection
FastAPI's dependency injection system:
```python
from fastapi import Depends

def get_db():
    db = Database()
    try:
        yield db
    finally:
        db.close()

@app.get("/items/")
async def read_items(db = Depends(get_db)):
    return db.query("SELECT * FROM items")
```

## Database with SQLAlchemy
```python
from sqlalchemy.orm import Session
from fastapi import Depends

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/users/")
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user
```

## Authentication
JWT token authentication example:
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/users/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    user = decode_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return user
```

## File Uploads
Handle file uploads:
```python
from fastapi import File, UploadFile

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    return {"filename": file.filename, "size": len(contents)}

# Multiple files
@app.post("/uploadfiles/")
async def create_upload_files(files: list[UploadFile] = File(...)):
    return {"filenames": [file.filename for file in files]}
```

## Background Tasks
Run tasks in the background:
```python
from fastapi import BackgroundTasks

def send_email(email: str, message: str):
    # Simulate sending email
    print(f"Sending email to {email}: {message}")

@app.post("/send-notification/")
async def send_notification(
    email: str,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(send_email, email, "Thank you for signing up!")
    return {"message": "Notification sent in background"}
```

## CORS Configuration
Enable Cross-Origin Resource Sharing:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Error Handling
Custom exception handlers:
```python
from fastapi import HTTPException

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail}
    )

# Raise exceptions
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    return items[item_id]
```

## Testing
Test your FastAPI application:
```python
from fastapi.testclient import TestClient

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_create_item():
    response = client.post("/items/", json={
        "name": "Test Item",
        "price": 10.5
    })
    assert response.status_code == 200
```

## Performance Tips
1. Use `async def` for I/O-bound operations
2. Use regular `def` for CPU-bound operations
3. Implement database connection pooling
4. Use background tasks for non-critical operations
5. Enable response caching when appropriate
6. Use Pydantic for data validation
7. Implement proper logging and monitoring

## Best Practices
- Use type hints for all parameters
- Define clear request and response models
- Implement proper error handling
- Use dependency injection for shared logic
- Write comprehensive tests
- Document your API with docstrings
- Follow REST principles
- Version your API endpoints
- Implement rate limiting for production
- Use environment variables for configuration

## Common Use Cases
- **RESTful APIs**: Building standard REST APIs
- **Microservices**: Creating microservice architectures
- **Real-time Applications**: WebSocket support
- **Machine Learning APIs**: Serving ML models
- **Database Operations**: CRUD applications
- **Authentication Systems**: User management
- **File Processing**: Upload and download services
