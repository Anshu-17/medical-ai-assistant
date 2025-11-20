from typing import Tuple
from loguru import logger

from config.settings import settings
from core.exceptions import GuardRejectionError


class GuardService:
    """Validates medical relevance of queries"""
    
    def __init__(self, llm_service):
        self.llm_service = llm_service
        self.medical_keywords = settings.medical_keywords
        logger.info("Guard Service initialized")
    
    def is_medical_query(self, query: str) -> Tuple[bool, str]:
        query_lower = query.lower()
        
        # Layer 1: Keyword matching
        if any(kw in query_lower for kw in self.medical_keywords):
            return True, "Medical keywords detected"
        
        # Layer 2: Medical context
        medical_verbs = ['analyze', 'diagnosis', 'examine', 'assess', 'evaluate', 
                        'check', 'look at', 'review', 'interpret']
        if any(verb in query_lower for verb in medical_verbs):
            return True, "Medical context detected"
        
        # Layer 3: LLM validation
        try:
            validation_prompt = f"""Is this query medical/healthcare related?

Query: "{query}"

Respond ONLY with "YES" or "NO" followed by brief reason.

Examples:
- "What are flu symptoms?" → YES - disease symptoms
- "Analyze this X-ray" → YES - medical image
- "What's the weather?" → NO - not medical

Response:"""

            response = self.llm_service.generate_text(validation_prompt).strip()
            
            if response.upper().startswith("YES"):
                return True, "LLM validated as medical"
            else:
                reason = response.split("-", 1)[1].strip() if "-" in response else "Not medical"
                return False, reason
                
        except Exception as e:
            logger.warning(f"LLM validation failed: {e}")
            return True, "Validation inconclusive - allowing"
    
    def get_rejection_message(self, reason: str) -> str:
        """Generate user-friendly rejection message"""
        return f"""I apologize, but I'm a specialized medical AI assistant.

Reason: {reason}

I can help with:
Medical conditions and symptoms
Treatment options and medications
Medical image analysis
Clinical guidelines
Health calculations
Please ask a medical or healthcare question!"""