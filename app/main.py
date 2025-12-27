from fastapi import FastAPI
from loguru import logger
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from app.api.v1.api import api_router
from app.core.config import settings
from app.models import *
from app.schemas import *
from fastapi.middleware.cors import CORSMiddleware
from app.services.database import engine

# Load environment variables
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await engine.dispose()
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
