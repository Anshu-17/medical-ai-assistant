import sys
from loguru import logger
from dotenv import load_dotenv
from config.settings import settings
from services.knowledge_base import KnowledgeBaseService
from services.image_handler import ImageHandlerService
from services.llm_service import LLMService
from agents.medical_agent import MedicalAgent
from ui.streamlit_app import run_app
import streamlit as st

st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="wide", 
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_application():
    """Initialize all application components - CACHED VERSION"""
    try:
        logger.info("Starting Medical AI Assistant...")
        
        # Load environment variables
        load_dotenv()
        logger.info("Environment variables loaded")
        
        # Initialize services
        logger.info("Initializing services...")
        knowledge_service = KnowledgeBaseService()
        image_handler = ImageHandlerService()
        llm_service = LLMService()
        
        # Initialize agent
        logger.info("Initializing agent...")
        agent = MedicalAgent(
            knowledge_service=knowledge_service,
            image_handler=image_handler,
            llm_service=llm_service,
            enable_guard=True
        )
        
        logger.success("Application initialized successfully")
        
        return agent
        
    except Exception as e:
        logger.error(f"Application initialization failed: {e}")
        sys.exit(1)


def main():
    """Main entry point"""
    # Configure logger
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Initialize application (CACHED - only runs once!)
    agent = initialize_application()
    
    # Run Streamlit UI
    logger.info("Launching Streamlit UI...")
    run_app(agent)


if __name__ == "__main__":
    main()