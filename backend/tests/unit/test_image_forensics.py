import pytest
import os
import cv2
import numpy as np
from PIL import Image

from app.services.image_forensics.ela_analyzer import perform_ela
from app.services.image_forensics.noise_analyzer import analyze_noise
from app.services.image_forensics.frequency_analyzer import analyze_frequency
from app.services.image_forensics.copy_move_detector import detect_copy_move
from app.services.image_forensics.ai_score_estimator import estimate_ai_likelihood

@pytest.fixture
def dummy_image_path(tmp_path):
    """Creates a simple dummy image for testing."""
    img_path = str(tmp_path / "test_img.jpg")
    
    # Create a 100x100 white image
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (20, 20), (80, 80), (0, 0, 0), -1)
    
    # Add some noise to make it realistic
    noise = np.random.randint(0, 50, (100, 100, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return img_path

def test_ela_analyzer(dummy_image_path):
    score = perform_ela(dummy_image_path)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_noise_analyzer(dummy_image_path):
    score = analyze_noise(dummy_image_path)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_frequency_analyzer(dummy_image_path):
    score = analyze_frequency(dummy_image_path)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_copy_move_detector(dummy_image_path):
    score = detect_copy_move(dummy_image_path)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_ai_score_estimator():
    # Mock some scores
    scores = {
        "ela": 0.8,
        "noise": 0.2,
        "frequency": 0.1,
        "copy_move": 0.0
    }
    result = estimate_ai_likelihood(scores)
    
    assert "ai_likelihood_score" in result
    assert "ai_likelihood_label" in result
    assert isinstance(result["ai_likelihood_score"], float)
    assert isinstance(result["ai_likelihood_label"], str)
    
    # Based on the logic, high ELA should trigger "likely_manipulated" or similar
    # The exact string may vary, but we can check if it returns a non-empty string
    assert len(result["ai_likelihood_label"]) > 0
