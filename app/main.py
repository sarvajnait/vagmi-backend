from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from app.api.v1.api import api_router
from app.core.config import settings
from app.core.langgraph.graph import EducationPlatform
from app.models import *
from app.schemas import *
from app.services.database import create_db_and_tables
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URL = settings.POSTGRES_URL
GOOGLE_API_KEY = settings.GOOGLE_API_KEY

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

COLLECTION_NAME = "education_documents"


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield
    logger.info("Shutting down application")


app = FastAPI(lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    verbose=False,
    temperature=0.5,
    streaming=True,
)


# Initialize platform
platform = EducationPlatform()

app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }


@app.get("/")
def root():
    return {
        "message": "Welcome to VAGMI",
    }
