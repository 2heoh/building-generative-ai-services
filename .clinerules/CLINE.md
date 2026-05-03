# CLINE.md - Project Context for AI Assistant

## Project Overview

**Name:** simple-web-server (Building Generative AI Services)

**Description:** A FastAPI-based web server providing generative AI services including text generation, image generation, RAG (Retrieval-Augmented Generation), and PDF document processing.

**Repository:** git@github.com:2heoh/building-generative-ai-services.git

## Tech Stack

- **Python** (primary language)
- **FastAPI** (web framework)
- **Pydantic** (data validation)
- **OpenAI API** (GPT-4 integration)
- **HuggingFace Transformers** (tinyLlama, gemma2b models)
- **Sentence Transformers** (embeddings)
- **Qdrant** (vector database)
- **PyTorch** (ML backend)
- **BeautifulSoup4 + lxml** (web scraping)
- **pypdf** (PDF processing)

## Python Environment

**Virtual Environment:** `.venv` directory

### Setup Instructions

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Activate virtual environment first
source .venv/bin/activate

# Run FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or use FastAPI CLI (if installed)
fastapi dev main.py
```

### Running Tests

```bash
# Activate virtual environment first
source .venv/bin/activate

# Run all tests (PYTHONPATH is required for module imports)
PYTHONPATH=. pytest

# Run with verbose output
PYTHONPATH=. pytest -v

# Run specific test file
PYTHONPATH=. pytest tests/rag/test_transform.py -v
```

## Project Structure

```
simple-web-server/
├── main.py              # FastAPI application entry point
├── models.py            # ML model loading and inference
├── schemas.py           # Pydantic request/response schemas
├── settings.py          # Application settings (pydantic-settings)
├── dependencies.py      # FastAPI dependencies
├── upload.py            # File upload handling
├── scraper.py           # Web scraping utilities
├── utils.py             # Utility functions
├── client.py            # API client
├── requirements.txt     # Python dependencies
├── rag/                 # RAG module
│   ├── extractor.py     # Text extraction from documents
│   ├── service.py       # RAG service logic
│   ├── repository.py    # Vector database operations
│   └── transform.py     # Token/chunking transformations
├── tests/
│   └── rag/
│       └── test_transform.py
├── .env                 # Environment variables (not committed)
├── .gitignore
└── .venv/               # Python virtual environment (not committed)
```

## Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here
APP_SECRET=your_secret_key_at_least_32_chars

# Optional (with defaults in settings.py)
DATABASE_URL=postgres://user:pass@localhost:5432/database
CORS_WHITELIST=http://localhost:3000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/generate/text` | Simple text generation |
| POST | `/generate/text` | Text generation with request body |
| GET | `/chat` | OpenAI GPT-4 chat |
| POST | `/upload` | PDF file upload and processing |

## Key Notes

- **Models are loaded on startup** via FastAPI lifespan
- **RAG integration** uses Qdrant for vector storage and retrieval
- **PDF processing** extracts text and stores embeddings in background tasks
- **Token counting** is done via tiktoken
- **Async operations** are used throughout where applicable

## Coding Conventions

- Use `async/await` for I/O operations
- Use Pydantic schemas for request/response validation
- Dependencies injected via FastAPI's `Depends()`
- Settings managed via pydantic-settings with `.env` file
- Logging via loguru