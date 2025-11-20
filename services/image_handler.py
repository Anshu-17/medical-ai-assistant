from typing import Optional, Tuple, Dict
from datetime import datetime
from PIL import Image
import io
import requests
from loguru import logger
from core.models import ImageMetadata
from core.exceptions import ImageProcessingError


class ImageHandlerService:
    """Manages image uploads and storage"""
    
    def __init__(self):
        """Initialize image handler"""
        self.uploaded_images: Dict[str, Dict] = {}
        self.pending_image: Optional[Image.Image] = None
        self.pending_filename: Optional[str] = None
        logger.info("Image Handler Service initialized")
    
    def load_image(self, image_source: any) -> Image.Image:
        try:
            if isinstance(image_source, Image.Image):
                return image_source.convert("RGB")    
            elif isinstance(image_source, str):
                if image_source.startswith(('http://', 'https://')):
                    # Load from URL
                    response = requests.get(
                        image_source,
                        headers={"User-Agent": "MedicalAI/2.0"},
                        timeout=15
                    )
                    response.raise_for_status()
                    return Image.open(io.BytesIO(response.content)).convert("RGB")
                else:
                    # Load from file path
                    return Image.open(image_source).convert("RGB")
            elif hasattr(image_source, 'read'):
                # File-like object
                return Image.open(image_source).convert("RGB")
            raise ValueError(f"Unsupported image source type: {type(image_source)}") 
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise ImageProcessingError(f"Image load failed: {e}")
    
    def store_image(self, image: Image.Image, filename: str) -> ImageMetadata:
        try:
            metadata = ImageMetadata(
                filename=filename,
                size=image.size,
                format=image.format or "Unknown"
            )
            
            self.uploaded_images[filename] = {
                "image": image,
                "metadata": metadata
            } 
            # Set as pending
            self.pending_image = image
            self.pending_filename = filename
            
            logger.info(f"Image stored: {filename}")
            logger.info(f"Pending: {self.has_pending_image()}")
            logger.info(f"Total uploaded: {len(self.uploaded_images)}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to store image: {e}")
            raise ImageProcessingError(f"Image storage failed: {e}")
    
    def get_pending_image(self) -> Optional[Tuple[Image.Image, str]]:
        """Get and clear pending image"""
        if self.pending_image:
            img = self.pending_image
            filename = self.pending_filename
            self.pending_image = None
            self.pending_filename = None
            return (img, filename)
        return None
    
    def has_pending_image(self) -> bool:
        """Check if there's a pending image"""
        return self.pending_image is not None
    
    def get_uploaded_image(self, filename: Optional[str] = None) -> Optional[Image.Image]:
        """Get uploaded image by filename or most recent"""
        if not self.uploaded_images:
            return None
        
        if filename is None:
            filename = list(self.uploaded_images.keys())[-1]       
        return self.uploaded_images.get(filename, {}).get("image")
    
    def clear_all(self):
        """Clear all images"""
        self.uploaded_images.clear()
        self.pending_image = None
        self.pending_filename = None
        logger.info("All images cleared")