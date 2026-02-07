# Vagmi Backend

Educational platform backend built with FastAPI, providing AI-powered content management and student assessment features.

## Features

### Core Functionality
- **Academic Hierarchy Management**: Boards, Classes, Subjects, and Chapters
- **Multi-language Support**: English, Kannada, Hindi, Tamil, Telugu, Malayalam, and more
- **LLM Resource Management**: Textbooks, Notes, Images, and Q&A patterns
- **AI-Powered Activities**: Automatic topic and question generation
- **Activity Sessions**: Student play sessions with immediate AI feedback
- **User Management**: Authentication and role-based access control

### AI Features
- **Topic Generation**: Automatically extract key topics from chapter content
- **Activity Generation**: Generate MCQ and descriptive questions
- **Immediate Feedback**: AI evaluation of descriptive answers with scoring
- **Language-Aware**: Generates content in the appropriate medium language
- **Kannada Text Conversion**: Automatic conversion of legacy ASCII Kannada to Unicode

### Resource Management
- **PDF Processing**: Upload and vectorize textbooks, notes, and Q&A patterns
- **Image Management**: Upload and compress images with searchable metadata
- **Vector Search**: Semantic search using PGVector and Google embeddings
- **Chunk Management**: Intelligent text chunking with overlap removal

## Tech Stack

- **Framework**: FastAPI
- **Database**: PostgreSQL with PGVector extension
- **ORM**: SQLModel
- **AI/ML**: LangChain, Google Generative AI (Gemini)
- **Storage**: DigitalOcean Spaces
- **Authentication**: JWT tokens
- **PDF Processing**: PyMuPDF, pdfplumber
- **Text Processing**: py-mini-racer (Kannada conversion)

## Installation

### Prerequisites
- Python 3.11+
- PostgreSQL with PGVector extension
- Google Cloud API key (for Gemini)
- DigitalOcean Spaces credentials

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd vagmi-backend
```

2. **Install dependencies using uv**
```bash
uv sync
```

3. **Set up environment variables**
Create a `.env` file:
```env
POSTGRES_URL=postgresql+psycopg://user:password@host:port/database
GOOGLE_API_KEY=your_google_api_key
DO_SPACES_KEY=your_do_spaces_key
DO_SPACES_SECRET=your_do_spaces_secret
DO_SPACES_BUCKET=your_bucket_name
DO_SPACES_REGION=your_region
SECRET_KEY=your_jwt_secret_key
```

4. **Run database migrations**
```bash
alembic upgrade head
```

5. **Start the server**
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Key Endpoints

### Activities
- `POST /api/v1/activities/session/start` - Start activity session
- `POST /api/v1/activities/session/{session_id}/submit` - Submit answer
- `POST /api/v1/activities/topics/generate` - AI generate topics
- `POST /api/v1/activities/generate` - AI generate activities

### LLM Resources
- `POST /api/v1/llm-resources/textbook` - Upload textbook PDF
- `POST /api/v1/llm-resources/llm-note` - Upload LLM note PDF
- `POST /api/v1/llm-resources/qa-pattern` - Upload Q&A pattern PDF
- `POST /api/v1/llm-resources/image` - Upload image

### Academic Hierarchy
- `GET /api/v1/hierarchy` - Get full hierarchy
- `POST /api/v1/chapters` - Create chapter
- `PUT /api/v1/chapters/{id}` - Update chapter

## Activity Workflow

1. **Topic Generation**: Upload textbook → AI extracts key topics
2. **Activity Generation**: Select topics → AI generates questions (MCQ/Descriptive)
3. **Publishing**: Review and publish activities
4. **Student Play**: Students answer questions
5. **Immediate Feedback**: Descriptive answers get instant AI evaluation

## Kannada Support

The system automatically detects and converts legacy ASCII-encoded Kannada text to proper Unicode:

```python
from app.utils.kannada_converter import convert_kannada_text

# Automatically detects encoding and converts if needed
unicode_text = convert_kannada_text(legacy_text)
```

Works seamlessly with PDF uploads - all Kannada content is automatically converted.

## Database Schema

Key models:
- `Board`, `Class`, `Subject`, `Medium`, `Chapter` - Academic hierarchy
- `Topic` - Chapter topics
- `ActivityGroup`, `ChapterActivity` - Questions and activities
- `ActivityPlaySession`, `ActivityAnswer` - Student sessions
- `LLMTextbook`, `LLMNote`, `QAPattern`, `LLMImage` - Learning resources

## Development

### Running Tests
```bash
pytest
```

### Database Migrations
```bash
# Create new migration
alembic revision -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

### Code Style
The project uses standard Python formatting. Run linters before committing:
```bash
ruff check .
```

## Architecture

### Vector Store
- Uses PGVector for semantic search
- Separate collections for textbooks, notes, Q&A, and images
- Chunks stored with metadata for filtering

### AI Pipeline
1. Text extraction from PDFs
2. Kannada text conversion (if needed)
3. Chunking with overlap
4. Embedding generation (Google Gemini)
5. Storage in vector database

### Activity Generation
1. Retrieve relevant chapter content
2. Generate topics using LLM
3. Create activities based on selected topics
4. Store with proper metadata

## Contributing

1. Create a feature branch
2. Make changes
3. Run tests
4. Create pull request

## License

Proprietary - All rights reserved
