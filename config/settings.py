from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with validation"""
    
    # API Keys
    google_api_key: str = Field(..., alias="GOOGLE_API_KEY")
    pinecone_api_key: str = Field(..., alias="PINECONE_API_KEY")
    serpapi_key: str = Field(..., alias="SERPAPI_KEY")
    
    # Pinecone Configuration
    pinecone_index_name: str = Field(default="medical-pdf-index")
    pinecone_environment: str = Field(default="us-east-1")
    
    # Model Configuration
    gemini_model: str = Field(default="gemini-2.5-flash-lite")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    
    # Agent Configuration
    max_iterations: int = Field(default=15)
    max_memory_turns: int = Field(default=10)
    top_k_results: int = Field(default=10)
    llm_temperature: float = Field(default=0.0)
    max_output_tokens: int = Field(default=4096)
    
    # Medical Keywords for Guard
    medical_keywords: List[str] = Field(default=[
        'symptom', 'disease', 'treatment', 'diagnosis', 'medical', 'health',
        'patient', 'doctor', 'hospital', 'clinic', 'medication', 'drug',
        'pain', 'fever', 'infection', 'cancer', 'diabetes', 'surgery',
        'prescription', 'therapy', 'clinical', 'pharmaceutical', 'radiology',
        'x-ray', 'mri', 'ct scan', 'blood', 'test', 'laboratory', 'anatomy',
        'pathology', 'cardiology', 'neurology', 'oncology'
    ])
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton instance
settings = Settings()