import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def detect_copy_move(image_path: str) -> float:
    """
    Detects copy-move forgery (cloning) using ORB feature matching.
    Returns a score between 0.0 and 1.0.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
            
        # Downscale to speed up processing
        h, w = img.shape
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            
        # Use ORB for feature detection (fast and free vs SIFT/SURF)
        orb = cv2.ORB_create(nfeatures=1000)
        keypoints, descriptors = orb.detectAndCompute(img, None)
        
        if descriptors is None or len(descriptors) < 50:
            return 0.0
            
        # Match features against themselves
        # Use Brute-Force Matcher with Hamming distance for ORB
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(descriptors, descriptors, k=4)
        
        # Filter matches (exclude self-matches and very close spatial points)
        good_matches = []
        for m_list in matches:
            for m in m_list:
                # m.queryIdx and m.trainIdx
                if m.queryIdx == m.trainIdx:
                    continue
                    
                pt1 = keypoints[m.queryIdx].pt
                pt2 = keypoints[m.trainIdx].pt
                
                # Check spatial distance (if points are too close, it's just normal local similarity)
                dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                
                # If they are spatially distant but feature-wise very similar (low hamming distance)
                if dist > 50 and m.distance < 30:
                    good_matches.append(m)
                    
        # Score based on number of clustered anomalous matches
        # A few matches happen randomly, but >15 highly similar distant points indicates cloning.
        score = min(max((len(good_matches) - 5) / 20.0, 0.0), 1.0)
        return float(score)
        
    except Exception as e:
        logger.error(f"Copy-move detection failed: {e}")
        return 0.0
