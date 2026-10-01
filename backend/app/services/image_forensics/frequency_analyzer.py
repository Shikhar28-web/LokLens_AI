import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def analyze_frequency(image_path: str) -> float:
    """
    Performs FFT frequency anomaly detection.
    AI generated images (like Midjourney, GANs) often have unnatural frequency distributions.
    Returns a score between 0.0 and 1.0.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
            
        # Compute 2D Fourier Transform
        f_transform = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f_transform)
        
        # Calculate magnitude spectrum
        magnitude = np.abs(f_shift)
        
        # Calculate radially averaged power spectrum
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        y, x = np.indices((h, w))
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        r = r.astype(int)
        
        # Sum magnitudes over radius
        tbin = np.bincount(r.ravel(), magnitude.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / nr
        
        # Authentic images usually have a 1/f^2 dropoff. 
        # Deepfakes/GANs often have a spike at high frequencies (checkerboard artifacts).
        # We calculate the ratio of high-freq to mid-freq energy.
        if len(radial_profile) < 50:
            return 0.0
            
        mid_freq_energy = np.sum(radial_profile[20:min(len(radial_profile)//2, 100)])
        high_freq_energy = np.sum(radial_profile[min(len(radial_profile)//2, 100):])
        
        if mid_freq_energy == 0:
            return 0.0
            
        ratio = high_freq_energy / mid_freq_energy
        
        # Normalize (heuristic threshold mapping)
        score = min(max((ratio - 0.05) / 0.15, 0.0), 1.0)
        return float(score)
        
    except Exception as e:
        logger.error(f"Frequency analysis failed: {e}")
        return 0.0
