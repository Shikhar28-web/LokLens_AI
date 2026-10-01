import pytest
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.services.ocr.ocr_engine import extract_text, preprocess_for_ocr

@pytest.fixture
def text_image_path(tmp_path):
    """Creates a dummy image with text for testing OCR."""
    img_path = str(tmp_path / "test_ocr_img.png")
    
    # Create a white image
    img = Image.new('RGB', (400, 150), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw some clear text
    # Tesseract should be able to read standard basic fonts even without a ttf loaded, 
    # but PIL default font is tiny. Let's just draw some basic shapes if font isn't available,
    # OR we can just try to use a default font and scale it up.
    # Actually, a better way for a unit test to guarantee OCR success without external fonts 
    # is to mock pytesseract in the test, or just test the preprocessing.
    # Since tesseract might not be installed on the CI/CD machine, we should test 
    # the behavior gracefully.
    
    # Let's write text
    draw.text((20, 50), "Testing OCR Pipeline", fill=(0,0,0))
    img.save(img_path)
    
    return img_path

def test_preprocess_for_ocr(text_image_path):
    processed = preprocess_for_ocr(text_image_path)
    assert processed is not None
    assert len(processed.shape) == 2 # Should be grayscale/binary (2D array)

def test_extract_text_graceful(text_image_path):
    # This test will attempt OCR. If Tesseract is not installed, it should return gracefully
    # with {"text": None, "confidence": 0.0} instead of crashing.
    result = extract_text(text_image_path)
    
    assert "text" in result
    assert "confidence" in result
    assert isinstance(result["confidence"], float)
    
    # If Tesseract IS installed and it successfully read the text
    if result["text"] is not None:
        assert isinstance(result["text"], str)
