from abc import ABC, abstractmethod
from typing import Any
from loguru import logger


class BaseMedicalTool(ABC):
    """Abstract base class for medical tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        logger.info(f"Tool initialized: {name}")
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> str:
        """Execute the tool's main functionality"""
        pass
    
    def __call__(self, *args, **kwargs) -> str:
        """Make tool callable"""
        try:
            logger.info(f"Executing tool: {self.name}")
            result = self.execute(*args, **kwargs)
            logger.success(f"Tool {self.name} completed")
            return result
        except Exception as e:
            logger.error(f"Tool {self.name} failed: {e}")
            return f"Tool execution failed: {str(e)}"