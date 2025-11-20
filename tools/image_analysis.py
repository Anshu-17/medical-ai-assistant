from langchain_core.tools import tool
from loguru import logger
from datetime import datetime

from tools.base import BaseMedicalTool
from services.image_handler import ImageHandlerService
from services.llm_service import LLMService


class ImageAnalysisTool(BaseMedicalTool):
    """Analyze medical images using vision AI"""
    
    def __init__(self, image_handler: ImageHandlerService, llm_service: LLMService):
        """
        Initialize image analysis tool
        
        Args:
            image_handler: Image handler service
            llm_service: LLM service for vision
        """
        super().__init__(
            name="analyze_medical_image",
            description="""Analyze uploaded medical image using Gemini vision.
            
            Image must be uploaded first. Tool automatically uses most recent image.
            
            Use for: X-ray, CT, MRI analysis, detecting abnormalities,
            tumors, lesions, radiological assessments.
            
            Args:
                query: Analysis request (e.g., "Check for fractures")
                
            Returns:
                Detailed radiological analysis report
            """
        )
        self.image_handler = image_handler
        self.llm_service = llm_service
    
    def execute(self, query: str) -> str:
        """Analyze medical image - FIXED VERSION"""
        try:
            logger.info(f"Image analysis requested: {query[:50]}...")
            
            # FIXED: Try multiple ways to get the image
            img = None
            filename = None
            
            # Method 1: Check for pending image (most recent upload)
            image_data = self.image_handler.get_pending_image()
            if image_data:
                img, filename = image_data
                logger.info(f" Found pending image: {filename}")
            
            # Method 2: Get most recent uploaded image
            if img is None:
                img = self.image_handler.get_uploaded_image()
                if img:
                    # Try to get filename from uploaded_images dict
                    if self.image_handler.uploaded_images:
                        filename = list(self.image_handler.uploaded_images.keys())[-1]
                        logger.info(f"Found uploaded image: {filename}")
                    else:
                        filename = "uploaded_image"
                        logger.info(" Found image (no filename)")
            
            # Method 3: Check if there are any images at all
            if img is None and self.image_handler.uploaded_images:
                # Get the last uploaded image
                last_key = list(self.image_handler.uploaded_images.keys())[-1]
                img = self.image_handler.uploaded_images[last_key]["image"]
                filename = last_key
                logger.info(f" Retrieved from uploaded_images: {filename}")
            
            # If still no image found
            if img is None:
                logger.warning(" No image found in handler")
                logger.info(f"Handler state: pending={self.image_handler.has_pending_image()}, "
                          f"uploaded_count={len(self.image_handler.uploaded_images)}")
                return """No medical image found. 

Please upload an image first:
1. Click "Browse files" in the Upload Image section (right sidebar)
2. Select your medical image (X-ray, CT, MRI, etc.)
3. Wait for " Image uploaded!" confirmation
4. Then ask me to analyze it again

The image must be uploaded before I can analyze it."""
            
            # We have an image! Proceed with analysis
            logger.info(f"Analyzing image: {filename}")
            
            # Construct analysis prompt
            prompt = f"""You are an expert radiologist with 30+ years of experience.

Analyze this medical image and provide a detailed, structured report.

Query: {query}

Structure your analysis as follows:

1. IMAGE TYPE & QUALITY
   - Imaging modality identification
   - Technical quality assessment

2. ANATOMICAL STRUCTURES
   - Visible anatomical structures
   - Positioning and orientation

3. KEY FINDINGS
   - Normal findings
   - Abnormal findings (if any)

4. PATHOLOGICAL FEATURES
   - Detailed description of any pathology
   - Location, size, characteristics

5. SEVERITY ASSESSMENT
   - Mild / Moderate / Severe (if applicable)

6. DIFFERENTIAL DIAGNOSIS
   - Most likely diagnosis
   - Alternative diagnoses to consider

7. RECOMMENDATIONS
   - Additional imaging needed
   - Clinical correlation suggested
   - Follow-up recommendations

Use precise medical terminology. Be thorough and evidence-based."""

            # Get analysis from LLM
            logger.info("Sending to Gemini Vision...")
            analysis = self.llm_service.analyze_image(prompt, img)
            logger.success("Analysis completed!")
            
            # Format output
            output = f""" MEDICAL IMAGE ANALYSIS REPORT
{'='*70}

 Query: {query}
 Image: {filename}
 Analysis Model: Gemini Vision (Clinical Grade)
 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*70}
 RADIOLOGICAL ANALYSIS:
{'='*70}

{analysis}

{'='*70}
  MEDICAL DISCLAIMER:
This AI-generated analysis is for educational and informational purposes 
only. It should NOT be used as a substitute for professional medical 
advice, diagnosis, or treatment. Always consult qualified healthcare 
professionals for medical decisions.
{'='*70}
"""
            
            logger.info("Image analysis report generated successfully")
            return output
            
        except Exception as e:
            error_msg = f"Image analysis error: {str(e)}"
            logger.error(error_msg)
            return f"""Analysis failed: {str(e)}

This could be due to:
- Image format not supported
- Image file corrupted
- API connectivity issues
- Model access problems

Please try:
1. Re-uploading the image
2. Using a different image format (PNG, JPG)
3. Checking your internet connection"""


# Global instance - will be set by agent
_image_tool_instance = None


def set_image_tool(tool: ImageAnalysisTool):
    """Set the global image tool instance"""
    global _image_tool_instance
    _image_tool_instance = tool


# LangChain tool wrapper
@tool
def analyze_medical_image(query: str) -> str:
    """Analyze uploaded medical image using AI vision.
    
    Use this tool to analyze X-rays, CT scans, MRIs, and other medical images.
    The image must be uploaded first through the UI.
    
    Args:
        query: What you want to analyze (e.g., "Check for fractures in this X-ray")
    
    Returns:
        Detailed radiological analysis report
    """
    if _image_tool_instance is None:
        logger.error("Image tool instance is None")
        return "Image analysis tool not initialized"
    
    logger.info("Calling image tool execute method")
    return _image_tool_instance.execute(query)