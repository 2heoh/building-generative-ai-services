# Simple Web Server - Building Generative AI Services

A FastAPI-based web server providing generative AI services including text generation, RAG (Retrieval-Augmented Generation), and PDF document processing.

## Features

- **Text Generation**: Simple text generation using pre-trained models
- **OpenAI Integration**: GPT-4 chat endpoint
- **RAG (Retrieval-Augmented Generation)**: Document-based question answering using vector embeddings
- **PDF Processing**: Upload and process PDF documents for RAG
- **Vector Database**: Qdrant integration for storing and retrieving embeddings

## Tech Stack

- **Python** - Primary language
- **FastAPI** - Web framework
- **Pydantic** - Data validation
- **OpenAI API** - GPT-4 integration
- **HuggingFace Transformers** - tinyLlama, gemma2b models
- **Sentence Transformers** - Embeddings
- **Qdrant** - Vector database
- **PyTorch** - ML backend
- **BeautifulSoup4 + lxml** - Web scraping
- **pypdf** - PDF processing

## Quick Start

### Prerequisites

- Python 3.10+

### Installation

```bash
# Clone the repository
git clone git@github.com:2heoh/building-generative-ai-services.git
cd simple-web-server

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here
APP_SECRET=your_secret_key_at_least_32_chars

# Optional (with defaults)
DATABASE_URL=postgres://user:pass@localhost:5432/database
CORS_WHITELIST=http://localhost:3000
```

### Running the Server

```bash
# Activate virtual environment
source .venv/bin/activate

# Run FastAPI server with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server will be available at `http://localhost:8000`.

### API Documentation

Once the server is running, interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/generate/text` | Simple text generation |
| POST | `/generate/text` | Text generation with request body |
| GET | `/chat` | OpenAI GPT-4 chat |
| POST | `/upload` | PDF file upload and processing |

## Running Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
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
├── tests/               # Test files
└── .clinerules/         # AI assistant configuration
```

## Development

### Coding Conventions

- Use `async/await` for I/O operations
- Use Pydantic schemas for request/response validation
- Dependencies injected via FastAPI's `Depends()`
- Settings managed via pydantic-settings with `.env` file
- Logging via loguru

## License

This project is part of the "Building Generative AI Services" educational resource.