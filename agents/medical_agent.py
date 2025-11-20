"""
Medical Agent Orchestrator - Dependency Inversion Principle
"""

from typing import Dict, Any, List
from loguru import logger
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

from config.settings import settings
from core.models import QueryResult
from core.exceptions import GuardRejectionError
from services.knowledge_base import KnowledgeBaseService
from services.image_handler import ImageHandlerService
from services.llm_service import LLMService
from services.guard_service import GuardService
from agents.memory import ConversationMemory

# Import tool classes and functions
from tools.knowledge_search import (
    KnowledgeSearchTool, 
    search_medical_knowledge,
    set_knowledge_tool  # ADD THIS
)
from tools.web_search import search_web_medical
from tools.image_analysis import (
    ImageAnalysisTool, 
    analyze_medical_image,
    set_image_tool  # ADD THIS
)
from tools.medical_calculator import calculate_medical_metric


# Agent Prompt Template
MEDICAL_REACT_PROMPT = PromptTemplate.from_template("""You are an expert medical AI assistant with specialized tools.
Goal: Provide accurate, evidence-based medical information.

{conversation_context}

AVAILABLE TOOLS:
{tools}

TOOL NAMES: {tool_names}

FORMAT:
Question: the medical question
Thought: analyze information needs and tool selection
Action: tool name from [{tool_names}]
Action Input: tool input
Observation: tool result
... (repeat as needed)
Thought: sufficient information gathered
Final Answer: complete evidence-based answer with citations

CONTEXT USAGE:
- Reference previous conversation when user uses "previous", "that", "it"
- Maintain continuity across turns
- Clarify based on earlier discussion

DECISION GUIDELINES:
1. **Standard medical questions** → search_medical_knowledge first
2. **Recent information (2024-2025)** → search_web_medical directly
3. **Medical images** → analyze_medical_image (image must be uploaded first)
4. **Calculations** → calculate_medical_metric
5. **Multi-tool strategies** → Combine sources when needed
6. **Always cite sources** in Final Answer

IMPORTANT - IMAGE ANALYSIS:
- If user asks about an image, X-ray, scan, or radiological analysis
- ALWAYS use analyze_medical_image tool
- The tool will automatically use the uploaded image
- Action Input should be the analysis query (e.g., "Check for fractures")
- DO NOT include JSON or image_source in Action Input for analyze_medical_image

EXAMPLES:

Example 1 - Knowledge Base:
Question: What is type 2 diabetes pathophysiology?
Thought: Standard medical knowledge in knowledge base
Action: search_medical_knowledge
Action Input: type 2 diabetes pathophysiology insulin resistance
Observation: [Medical textbook results]
Thought: Comprehensive information obtained
Final Answer: Type 2 diabetes is characterized by... [Source: Medical Knowledge Base]

Example 2 - Image Analysis:
Question: Analyze this chest X-ray for pneumonia
Thought: User wants image analysis, will use analyze_medical_image tool
Action: analyze_medical_image
Action Input: Analyze for signs of pneumonia, infiltrates, or consolidation
Observation: [Analysis report from Gemini]
Thought: Analysis complete with detailed findings
Final Answer: Based on the radiological analysis... [Source: AI Image Analysis]

Example 3 - Follow-up Question:
Previous context: Discussed pneumonia X-ray findings
Question: What treatment do you recommend?
Thought: User asking about pneumonia treatment based on previous image analysis
Action: search_medical_knowledge
Action Input: pneumonia treatment protocols antibiotics
Observation: [Treatment guidelines]
Thought: Have treatment information for pneumonia discussed earlier
Final Answer: For the pneumonia identified in the X-ray, recommended treatments include... [Source: Medical Knowledge Base]

Begin!

Question: {input}
Thought: {agent_scratchpad}""")


class MedicalAgent:
    """
    Main medical agent orchestrator
    Follows Dependency Inversion - depends on abstractions (services)
    """
    
    def __init__(
        self,
        knowledge_service: KnowledgeBaseService,
        image_handler: ImageHandlerService,
        llm_service: LLMService,
        enable_guard: bool = True
    ):
        """
        Initialize medical agent
        
        Args:
            knowledge_service: Knowledge base service
            image_handler: Image handler service
            llm_service: LLM service
            enable_guard: Enable query validation
        """
        logger.info("Initializing Medical Agent...")
        
        # Store services
        self.knowledge_service = knowledge_service
        self.image_handler = image_handler
        self.llm_service = llm_service
        
        # Initialize tool instances
        self.knowledge_tool = KnowledgeSearchTool(knowledge_service)
        self.image_tool = ImageAnalysisTool(image_handler, llm_service)
        
        # SET GLOBAL INSTANCES - THIS IS THE FIX! ✅
        set_knowledge_tool(self.knowledge_tool)
        set_image_tool(self.image_tool)
        
        # LangChain tool functions
        self.tools = [
            search_medical_knowledge,
            search_web_medical,
            analyze_medical_image,
            calculate_medical_metric
        ]
        
        # Initialize guard
        self.enable_guard = enable_guard
        if enable_guard:
            self.guard_service = GuardService(llm_service)
        
        # Initialize memory
        self.memory = ConversationMemory()
        
        # Create agent
        self.agent = create_react_agent(
            llm=llm_service.get_langchain_llm(),
            tools=self.tools,
            prompt=MEDICAL_REACT_PROMPT
        )
        
        # Create executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=settings.max_iterations,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        
        logger.success("Medical Agent initialized successfully")
    
    def query(self, question: str, skip_guard: bool = False) -> QueryResult:
        """
        Process user query
        
        Args:
            question: User's medical question
            skip_guard: Skip guard validation
            
        Returns:
            Query result with response and metadata
        """
        logger.info(f"Processing query: {question[:50]}...")
        
        # Guard validation
        if self.enable_guard and not skip_guard:
            is_valid, reason = self.guard_service.is_medical_query(question)
            
            if not is_valid:
                logger.warning(f"Query rejected: {reason}")
                return QueryResult(
                    response=self.guard_service.get_rejection_message(reason),
                    success=False,
                    rejected=True,
                    rejection_reason=reason
                )
        
        # Get conversation context
        context = self.memory.get_recent_context(n=3)
        image_context = self._get_image_context()
        
        try:
            # Execute agent
            result = self.agent_executor.invoke({
                "input": question,
                "image_context": image_context,
                "conversation_context": context
            })
            
            # Extract tool usage
            tool_usage = {}
            for step in result.get("intermediate_steps", []):
                action, _ = step
                tool_name = action.tool
                tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
            
            response_text = result["output"]
            tools_used = list(tool_usage.keys())
            
            # Save to memory
            self.memory.add_exchange(question, response_text, tools_used)
            
            logger.success(f"Query processed. Tools used: {tools_used}")
            
            return QueryResult(
                response=response_text,
                success=True,
                rejected=False,
                tools_used=tools_used,
                tool_call_counts=tool_usage
            )
            
        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            logger.error(error_message)
            
            # Save error to memory
            self.memory.add_exchange(question, error_message, [])
            
            return QueryResult(
                response=error_message,
                success=False,
                rejected=False,
                metadata={"error": str(e)}
            )
    def _get_image_context(self) -> str:
        if self.image_handler.has_pending_image():
            uploaded_keys = list(self.image_handler.uploaded_images.keys())
            if uploaded_keys:
                filename = uploaded_keys[-1]
                img_info = self.image_handler.uploaded_images[filename]
                metadata = img_info.get('metadata')
                
                return f"""IMAGE_STATUS: Image available
    IMAGE_FILE: {filename}
    IMAGE_SIZE: {metadata.size if metadata else 'Unknown'}
    NOTE: User has uploaded a medical image. You can analyze it using analyze_medical_image tool."""
            else:
                return "IMAGE_STATUS: Image available (filename unknown)"
        else:
            return "IMAGE_STATUS: No image uploaded"
        
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.memory.export_history()
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.memory.clear_history()
        logger.info("Conversation cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "memory_summary": self.memory.get_summary(),
            "total_messages": len(self.memory.get_all_messages()),
            "has_pending_image": self.image_handler.has_pending_image()
        }