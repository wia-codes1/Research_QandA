from fastapi import FastAPI
from app.api.pdf_router import pdf_router
from app.api.query_router import query_router
from pydantic_settings import BaseSettings

# Pydantic Settings to load environment variables
class Settings(BaseSettings):
    HUGGINGFACEHUB_API_TOKEN: str  # Define the environment variable as a class attribute

    class Config:
        env_file = ".env"  # Automatically load environment variables from .env file

# Instantiate settings
settings = Settings()

# Access the API key (this will raise an error if not set)
hf_api_key = settings.HUGGINGFACEHUB_API_TOKEN
print(f"✅ HF API key loaded: {hf_api_key}")

# Initialize FastAPI app
app = FastAPI()

# Root route to show basic functionality
@app.get("/")
def read_root():
    return {"message": "Welcome to the Retrieval-Augmented Generation (RAG) system!"}

# Health check route to verify the API is up and running
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "The system is up and running."}

# Include routers for PDF processing and query processing
app.include_router(pdf_router, prefix="/pdf", tags=["PDF Processing"])
app.include_router(query_router, prefix="/query", tags=["Query Processing"])
