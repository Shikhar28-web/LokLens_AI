"""
LokLens AI — Manual Image Forensics & AI Likelihood Tester
Usage:
    python evaluate_image.py <path_to_image>
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.services.image_forensics.ela_analyzer import perform_ela
from app.services.image_forensics.frequency_analyzer import analyze_frequency
from app.services.image_forensics.noise_analyzer import analyze_noise
from app.services.image_forensics.copy_move_detector import detect_copy_move
from app.services.image_forensics.ai_score_estimator import estimate_ai_likelihood
from app.services.image_forensics.metadata_extractor import extract_image_metadata

def test_image(image_path: str):
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return

    print("=" * 65)
    print(f" LokLens AI — Image Forensic Authenticity Analysis")
    print(f" File: {os.path.basename(image_path)}")
    print("=" * 65)

    # 1. Metadata
    meta = extract_image_metadata(image_path)
    print(f"Resolution         : {meta.get('width', 0)} x {meta.get('height', 0)} px")
    print(f"File Size          : {meta.get('file_size_bytes', 0) / 1024:.1f} KB")
    print(f"pHash (64-bit DCT) : {meta.get('phash', 'N/A')}")
    exif = meta.get('exif_json', {})
    exif_boost = meta.get('exif_authenticity_boost', 0.0)
    if exif:
        print(f"Camera / Model     : {exif.get('Make', '')} {exif.get('Model', '')}")
        print(f"Date Captured      : {exif.get('DateTimeOriginal', 'N/A')}")
        if exif_boost < 0:
            print(f"EXIF Camera Boost  : {exif_boost:+.2f}  (real camera detected → lowers AI score)")
    else:
        print(f"EXIF Metadata      : None (Stripped / Social Media Image)")

    print("-" * 65)
    print(" Running Mathematical Forensic Operators...")
    
    # 2. Forensics
    ela = perform_ela(image_path)
    freq = analyze_frequency(image_path)
    noise = analyze_noise(image_path)
    copy_move = detect_copy_move(image_path)

    scores = {
        "ela": ela,
        "frequency": freq,
        "noise": noise,
        "copy_move": copy_move,
        "exif_authenticity_boost": meta.get("exif_authenticity_boost", 0.0)
    }
    
    ai_verdict = estimate_ai_likelihood(scores)

    print(f" - Error Level Analysis (ELA)       : {ela:.3f}  (>0.60 indicates JPEG splicing)")
    print(f" - 2D-FFT Frequency Anomaly         : {freq:.3f}  (>0.60 indicates GAN/Diffusion artifacts)")
    print(f" - Local Noise Variance             : {noise:.3f}  (>0.60 indicates cut-and-paste compositing)")
    print(f" - Copy-Move Clone Matches          : {copy_move:.3f}  (>0.40 indicates cloned regions)")
    print("-" * 65)
    
    label = ai_verdict.get("ai_likelihood_label", "UNKNOWN")
    composite_score = ai_verdict.get("ai_likelihood_score")
    
    score_str = f"{composite_score:.2f}" if composite_score is not None else "N/A (Infographic Guard)"
    print(f" Composite AI / Tampering Score    : {score_str}")
    print(f" Final Forensic Verdict            : >>> {label} <<<")
    print("=" * 65)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        # Default test image in project
        target = os.path.join(os.path.dirname(__file__), "real_news_test.png")
        if not os.path.exists(target):
            target = r"D:\Projects\Political_lens\Mumbai Monsoon Flood News Bulletin.png"

    test_image(target)

