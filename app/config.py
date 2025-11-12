from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    HUGGINGFACEHUB_API_TOKEN: str  # Define the environment variable for Hugging Face API token

    class Config:
        env_file = ".env"  # Automatically load environment variables from .env file
