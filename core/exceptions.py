"""
Custom exceptions for the application
"""


class MedicalAIException(Exception):
    """Base exception for Medical AI application"""
    pass


class InitializationError(MedicalAIException):
    """Raised when initialization fails"""
    pass


class ToolExecutionError(MedicalAIException):
    """Raised when tool execution fails"""
    pass


class GuardRejectionError(MedicalAIException):
    """Raised when query is rejected by guard"""
    pass


class ImageProcessingError(MedicalAIException):
    """Raised when image processing fails"""
    pass