from typing import List, Dict, Any
from datetime import datetime
from loguru import logger

from core.models import Message, MessageRole
from config.settings import settings


class ConversationMemory:
    """Manages conversation history with configurable retention"""
    
    def __init__(self, max_history: int = None):
        """
        Initialize conversation memory
        
        Args:
            max_history: Maximum conversation turns to keep
        """
        self.max_history = max_history or settings.max_memory_turns
        self.conversation_history: List[Message] = []
        logger.info(f"Memory initialized (max: {self.max_history} turns)")
    
    def add_exchange(
        self,
        user_query: str,
        assistant_response: str,
        tools_used: List[str] = None
    ):
        """
        Add conversation exchange
        
        Args:
            user_query: User's question
            assistant_response: Assistant's response
            tools_used: List of tools used
        """
        # Add user message
        user_msg = Message(
            role=MessageRole.USER,
            content=user_query
        )
        self.conversation_history.append(user_msg)
        
        # Add assistant message
        assistant_msg = Message(
            role=MessageRole.ASSISTANT,
            content=assistant_response,
            tools_used=tools_used or []
        )
        self.conversation_history.append(assistant_msg)
        
        # Trim if exceeds max
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-(self.max_history * 2):]
        
        logger.debug(f"Added exchange. Total messages: {len(self.conversation_history)}")
    
    def get_recent_context(self, n: int = 3) -> str:
        """
        Get N most recent conversation turns
        
        Args:
            n: Number of recent turns
            
        Returns:
            Formatted conversation context
        """
        if not self.conversation_history:
            return "No recent conversation."
        
        # Get last N*2 messages (N user + N assistant)
        recent_messages = self.conversation_history[-(n * 2):]
        
        context_parts = ["RECENT CONVERSATION CONTEXT:\n"]
        
        i = 0
        while i < len(recent_messages):
            if i + 1 < len(recent_messages):
                user_msg = recent_messages[i]
                assistant_msg = recent_messages[i + 1]
                
                context_parts.append(
                    f"User: {user_msg.content}\n"
                    f"Assistant: {assistant_msg.content[:150]}...\n"
                )
                i += 2
            else:
                i += 1
        
        return "\n".join(context_parts)
    
    def get_all_messages(self) -> List[Message]:
        """Get all conversation messages"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear all conversation history"""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")
    
    def get_summary(self) -> str:
        if not self.conversation_history:
            return "No conversation history"
    
        total_turns = len(self.conversation_history) // 2
    
        tools_used = set()
        for msg in self.conversation_history:
            if msg.role == MessageRole.ASSISTANT:
                # Message object has tools_used as a list attribute
                if hasattr(msg, 'tools_used') and msg.tools_used:
                    tools_used.update(msg.tools_used)
    
        return f"Turns: {total_turns}, Tools used: {', '.join(tools_used) or 'None'}"
    
    def export_history(self) -> List[Dict[str, Any]]:
        """Export conversation history as JSON-serializable dict"""
        return [
            {
                "role": msg.role.value,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "tools_used": msg.tools_used,
                "metadata": msg.metadata
            }
            for msg in self.conversation_history
        ]