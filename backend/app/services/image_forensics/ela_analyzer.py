import os
from PIL import Image, ImageChops, ImageEnhance
import numpy as np

def perform_ela(image_path: str, quality: int = 90) -> float:
    """
    Performs Error Level Analysis (ELA) on an image.
    Returns a score between 0.0 and 1.0 representing the anomaly level.
    High score (>0.6) suggests manipulation.
    """
    try:
        original = Image.open(image_path).convert('RGB')
        
        # Save at known quality
        temp_path = image_path + ".temp_ela.jpg"
        original.save(temp_path, 'JPEG', quality=quality)
        
        resaved = Image.open(temp_path)
        
        # Compute absolute difference
        diff = ImageChops.difference(original, resaved)
        
        # Calculate max difference to scale the image
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        
        if max_diff == 0:
            max_diff = 1
            
        # Enhance difference to make it visible (for stats)
        scale = 255.0 / max_diff
        diff = ImageEnhance.Brightness(diff).enhance(scale)
        
        # Compute standard deviation of the difference
        # A high standard deviation means some parts compressed differently than others (anomaly)
        diff_array = np.array(diff)
        std_dev = np.std(diff_array)
        
        # Cleanup
        resaved.close()
        os.remove(temp_path)
        
        # Normalize score (heuristic mapping)
        # Typical authentic images have std_dev < 30. Manipulated can be > 60.
        score = min(max((std_dev - 10) / 70.0, 0.0), 1.0)
        return float(score)
        
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"ELA failed: {e}")
        return 0.0
