from langchain_core.tools import tool
from serpapi import GoogleSearch
from loguru import logger
import os

from tools.base import BaseMedicalTool


class WebSearchTool(BaseMedicalTool):
    """Search web for current medical information"""
    
    def __init__(self):
        """Initialize web search tool"""
        super().__init__(
            name="search_web_medical",
            description="""Search web for current medical information.
            
            Use for: Latest clinical trials, FDA approvals, updated guidelines,
            outbreak information, breaking medical news.
            
            Args:
                query: Search query
                
            Returns:
                Web search results with titles, snippets, URLs
            """
        )
    
    def execute(self, query: str) -> str:
        """Execute web search"""
        try:
            search = GoogleSearch({
                "q": f"{query} medical",
                "api_key": os.getenv("SERPAPI_KEY"),
                "num": 5
            })
            results = search.get_dict()
            
            if "organic_results" not in results or not results["organic_results"]:
                return "No web results found"
            
            # Format results
            snippets = ["WEB SEARCH RESULTS:\n"]
            
            for i, result in enumerate(results["organic_results"][:5], 1):
                title = result.get("title", "No title")
                snippet = result.get("snippet", "No description")
                link = result.get("link", "")
                
                snippets.append(f"{i}. **{title}**\n   {snippet}\n   🔗 {link}\n")
            
            return "\n".join(snippets)
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Web search error: {str(e)}"


# LangChain tool wrapper
@tool
def search_web_medical(query: str) -> str:
    """Search web for current medical information."""
    tool_instance = WebSearchTool()
    return tool_instance.execute(query)