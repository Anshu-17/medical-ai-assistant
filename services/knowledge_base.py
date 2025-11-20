"""
Knowledge Base Service - Single Responsibility: Vector DB operations
"""

from typing import List, Dict, Any
from loguru import logger
# UPDATED IMPORT - Works with pinecone-client 2.x
try:
    from pinecone import Pinecone  # v3+
except ImportError:
    import pinecone  # v2.x
    Pinecone = None

from sentence_transformers import SentenceTransformer

from config.settings import settings
from core.exceptions import ToolExecutionError, InitializationError


class KnowledgeBaseService:
    """Handles all knowledge base operations"""
    
    def __init__(self):
        """Initialize knowledge base connection"""
        try:
            logger.info("Initializing Knowledge Base Service...")
            
            # Initialize Pinecone (compatible with v2.x and v3.x)
            if Pinecone:  # v3+
                self.pc = Pinecone(api_key=settings.pinecone_api_key)
                self.index = self.pc.Index(settings.pinecone_index_name)
            else:  # v2.x
                import pinecone
                pinecone.init(
                    api_key=settings.pinecone_api_key,
                    environment=settings.pinecone_environment
                )
                self.index = pinecone.Index(settings.pinecone_index_name)
            
            # Initialize embedding model
            self.embed_model = SentenceTransformer(settings.embedding_model)
            
            logger.success("Knowledge Base Service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Knowledge Base: {e}")
            raise InitializationError(f"Knowledge Base init failed: {e}")
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search knowledge base
        
        Args:
            query: Search query
            top_k: Number of results (defaults to settings)
            
        Returns:
            List of search results with metadata
        """
        try:
            top_k = top_k or settings.top_k_results
            
            # Generate embedding
            query_embedding = self.embed_model.encode([query]).tolist()[0]
            
            # Search Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            if not results.get("matches"):
                return []
            
            # Format results
            formatted_results = []
            for match in results["matches"]:
                formatted_results.append({
                    "score": match.get("score", 0.0),
                    "text": match["metadata"].get("text", ""),
                    "source": match["metadata"].get("source", "unknown"),
                    "metadata": match.get("metadata", {})
                })
            
            logger.info(f"Found {len(formatted_results)} results for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}")
            raise ToolExecutionError(f"Search failed: {e}")
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text"""
        try:
            return self.embed_model.encode([text]).tolist()[0]
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise ToolExecutionError(f"Embedding failed: {e}")