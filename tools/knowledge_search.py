from langchain_core.tools import tool
from loguru import logger

from tools.base import BaseMedicalTool
from services.knowledge_base import KnowledgeBaseService


class KnowledgeSearchTool(BaseMedicalTool):
    """Search medical knowledge base"""
    
    def __init__(self, knowledge_service: KnowledgeBaseService):
        """
        Initialize knowledge search tool
        
        Args:
            knowledge_service: Knowledge base service instance
        """
        super().__init__(
            name="search_medical_knowledge",
            description="""Search medical knowledge base for clinical information.
            
            Use for: Medical definitions, pathophysiology, treatment protocols,
            clinical guidelines, drug information.
            
            Args:
                query: Medical question or topic
                
            Returns:
                Relevant medical literature with relevance scores
            """
        )
        self.knowledge_service = knowledge_service
    
    def execute(self, query: str) -> str:
        """Search knowledge base"""
        try:
            results = self.knowledge_service.search(query)
            
            if not results:
                return "No relevant information found in knowledge base"
            
            # Format results
            context_parts = []
            for i, result in enumerate(results, 1):
                score = result['score']
                text = result['text']
                source = result['source']
                
                context_parts.append(
                    f"Result {i} (Relevance: {score:.2f})\n"
                    f"Source: {source}\n{text}\n"
                )
            
            header = f"Found {len(results)} results\n\n"
            return header + "\n---\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Knowledge search error: {e}")
            return f"Search error: {str(e)}"


# Global instance - will be set by agent
_knowledge_tool_instance = None


def set_knowledge_tool(tool: KnowledgeSearchTool):
    """Set the global knowledge tool instance"""
    global _knowledge_tool_instance
    _knowledge_tool_instance = tool


# LangChain tool wrapper
@tool
def search_medical_knowledge(query: str) -> str:
    """Search medical knowledge base for clinical information.
    
    Use this tool to search for medical definitions, disease information,
    treatment protocols, clinical guidelines, and drug information.
    
    Args:
        query: Medical question or topic to search for
    
    Returns:
        Relevant medical literature with citations
    """
    if _knowledge_tool_instance is None:
        return "Knowledge base tool not initialized"
    return _knowledge_tool_instance.execute(query)