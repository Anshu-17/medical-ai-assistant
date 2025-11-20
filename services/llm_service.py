from typing import Any, List
import google.generativeai as genai
from loguru import logger
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import settings
from core.exceptions import InitializationError


class LLMService:
    """Handles all LLM operations"""
    
    def __init__(self):
        """Initialize LLM service"""
        try:
            logger.info("Initializing LLMService...")
            
            genai.configure(api_key=settings.google_api_key)
            self.gemini_vision = genai.GenerativeModel(settings.gemini_model)
            self.langchain_llm = ChatGoogleGenerativeAI(
                model=settings.gemini_model,
                temperature=settings.llm_temperature,
                max_output_tokens=settings.max_output_tokens,
                google_api_key=settings.google_api_key  
            )
            
            logger.success("LLMService initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLMService: {e}")
            raise InitializationError(f"Could not initialize LLMService: {e}")
    
    def generate_text(self, prompt: str) -> str:
        try:
            response = self.gemini_vision.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise
    
    def analyze_image(self, prompt: str, image: Any) -> str:
        try:
            response = self.gemini_vision.generate_content([prompt, image])
            return response.text
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            raise
    
    def get_langchain_llm(self) -> ChatGoogleGenerativeAI:
        return self.langchain_llm