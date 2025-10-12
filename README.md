# 🤖 PrecisionRAG - Chat with Your Documents Using AI

A full-stack RAG (Retrieval Augmented Generation) system that lets you upload documents and ask intelligent questions powered by Google Gemini AI.

![Tech Stack](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791?style=for-the-badge&logo=postgresql&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## ✨ Features

- 📄 **Document Upload** - Support for PDF, TXT, MD, and code files
- 🔍 **Intelligent Search** - Semantic similarity search using sentence transformers
- 💬 **AI Chat** - Powered by Google Gemini 2.5 Flash
- 🎯 **Context-Aware** - Retrieves relevant chunks from your documents
- 🎨 **Beautiful UI** - Modern React interface with real-time updates
- ⚡ **Fast Processing** - Efficient chunking and embedding generation

---

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   React     │────▶ │   FastAPI    │────▶ │ PostgreSQL  │
│  Frontend   │      │   Backend    │      │   Database  │
└─────────────┘      └──────────────┘      └─────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │ Sentence     │
                     │ Transformers │
                     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   Gemini AI  │
                     └──────────────┘
```

### How It Works

1. **Upload** - User uploads a document through the React frontend
2. **Extract** - Text is extracted from the document
3. **Chunk** - Document is split into intelligent chunks (paragraphs, sentences)
4. **Embed** - Each chunk is converted to a 384-dimensional vector using sentence transformers
5. **Store** - Vectors and text are stored in PostgreSQL
6. **Query** - User asks a question
7. **Search** - System finds relevant chunks using cosine similarity
8. **Generate** - Gemini AI generates an answer using the retrieved context

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 14+
- PostgreSQL 12+
- Google Gemini API Key

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/PrecisionRag.git
cd PrecisionRag
```

### 2. Setup Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup PostgreSQL database
psql -U postgres
CREATE DATABASE precisionrag_dev;
\q

# Configure environment variables
# Edit .env file and add your Gemini API key:
GEMINI_API_KEY=your_gemini_api_key_here

# Start the backend server
python run.py
```

Backend will run on **http://localhost:8000**

### 3. Setup Frontend

```bash
cd ../frontend

# Install dependencies
npm install

# Start the development server
npm start
```

Frontend will open automatically at **http://localhost:3000**

---

## 📖 Usage

### Upload a Document

1. Open **http://localhost:3000** in your browser
2. Click "Choose File" and select a document
3. Click "📤 Upload & Process"
4. Wait for processing (usually 5-10 seconds)

### Ask Questions

1. Type your question in the text box at the bottom
2. Press Enter or click "📤 Send"
3. AI will retrieve relevant context and generate an answer
4. See which document chunks were used in the metadata

### Example Questions

```
What is supervised learning?
How does reinforcement learning work?
Explain the key concepts from this document
What are the main topics covered?
```

---

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **PostgreSQL** - Vector storage (JSON-based, upgradeable to pgvector)
- **SQLAlchemy** - Database ORM
- **Sentence Transformers** - Local embedding generation (no API costs)
- **Google Gemini** - AI chat completion
- **PyPDF2** - PDF text extraction

### Frontend
- **React 18** - UI framework
- **Modern CSS** - Gradient themes and animations
- **Fetch API** - HTTP requests

### AI/ML
- **Sentence Transformers** - `all-MiniLM-L6-v2` model (384 dimensions)
- **Google Gemini 2.5 Flash** - Fast, efficient AI responses
- **Cosine Similarity** - Vector search algorithm

---

## 📂 Project Structure

```
PrecisionRag/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── upload.py          # File upload & processing endpoints
│   │   │   └── chat.py            # Chat & search endpoints
│   │   ├── core/
│   │   │   ├── config.py          # Configuration management
│   │   │   ├── database.py        # Database connection
│   │   │   └── vector_storage.py  # Vector storage operations
│   │   ├── services/
│   │   │   ├── text_extraction.py # Extract text from files
│   │   │   ├── chunking.py        # Smart document chunking
│   │   │   └── embeddings.py      # Generate embeddings
│   │   └── main.py               # FastAPI application
│   ├── requirements.txt          # Python dependencies
│   ├── .env                      # Environment variables
│   └── run.py                    # Development server
├── frontend/
│   ├── src/
│   │   ├── App.js               # Main React component
│   │   └── App.css              # Styles
│   ├── public/
│   └── package.json             # Node dependencies
└── README.md                    # This file
```

---

## 🔧 Configuration

### Backend (.env)

```env
# Database
DATABASE_URL=postgresql://postgres:password@localhost:5433/precisionrag_dev

# AI API
GEMINI_API_KEY=your_api_key_here

# Application
DEBUG=True
ENVIRONMENT=development
```

### Supported File Types

- PDF (`.pdf`)
- Text (`.txt`, `.md`)
- Code (`.py`, `.js`, `.ts`, `.java`, `.c`, `.cpp`)
- Web (`.html`, `.css`, `.json`, `.yaml`, `.yml`)

---

## 🎯 API Endpoints

### Upload & Processing

```
POST /api/v1/process
```
Upload and process a document through the complete RAG pipeline

### Chat

```
POST /api/v1/chat/message
```
Send a message and get AI response with context

### Context Search

```
POST /api/v1/chat/search-context
```
Search for relevant context without AI response

### Health Check

```
GET /api/v1/chat/health
```
Check if chat service and AI are configured

### Documentation

```
GET /docs
```
Interactive API documentation (Swagger UI)

---

## 🧪 Testing

Test the backend pipeline:

```bash
cd backend
python test_complete_pipeline.py
```

Test individual components:

```python
# Test text extraction
python -c "from app.services.text_extraction import test_extraction; test_extraction()"

# Test embeddings
python -c "from app.services.embeddings import test_embeddings; test_embeddings()"
```

---

## 🚀 Deployment

### Backend Deployment (Example with Render/Railway)

1. Create a PostgreSQL database
2. Set environment variables (DATABASE_URL, GEMINI_API_KEY)
3. Deploy with:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```

### Frontend Deployment (Vercel/Netlify)

1. Build the React app:
   ```bash
   npm run build
   ```
2. Update API_BASE_URL in App.js to your backend URL
3. Deploy the `build` folder

---

## 📊 Performance

- **Document Processing**: 5-10 seconds per document
- **Embedding Generation**: ~100ms per chunk (local)
- **Similarity Search**: <50ms for 1000 chunks
- **AI Response**: 1-3 seconds (Gemini 2.5 Flash)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📝 License

MIT License - feel free to use this project for learning and commercial purposes.

---

## 🙏 Acknowledgments

- Built with FastAPI and React
- Powered by Google Gemini AI
- Embeddings by Sentence Transformers
- Inspired by modern RAG systems

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ for learning AI and RAG systems**
