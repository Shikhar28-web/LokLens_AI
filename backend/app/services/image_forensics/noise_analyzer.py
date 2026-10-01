import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def analyze_noise(image_path: str) -> float:
    """
    Analyzes local noise variance to detect splices.
    Returns a score between 0.0 and 1.0.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
            
        # Apply median blur to isolate structural features
        blurred = cv2.medianBlur(img, 3)
        
        # The difference represents the noise
        noise = cv2.absdiff(img, blurred)
        
        # Calculate local variance using block-based approach
        # Split image into 8x8 blocks
        h, w = noise.shape
        block_size = 16
        
        variances = []
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                block = noise[y:y+block_size, x:x+block_size]
                if block.shape[0] == block_size and block.shape[1] == block_size:
                    variances.append(np.var(block))
                    
        if not variances:
            return 0.0
            
        # If variance of variances is very high, it means some blocks have completely 
        # different noise patterns than others (common in copy/paste splices)
        var_of_var = float(np.var(variances))
        
        # Normalize (heuristic threshold mapping)
        score = min(max((var_of_var - 50) / 300.0, 0.0), 1.0)
        return score
        
    except Exception as e:
        logger.error(f"Noise analysis failed: {e}")
        return 0.0
