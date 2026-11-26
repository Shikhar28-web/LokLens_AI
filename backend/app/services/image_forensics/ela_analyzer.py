import os
from PIL import Image, ImageChops, ImageEnhance
import numpy as np

def perform_ela(image_path: str, quality: int = 90) -> float:
    """
    Performs multi-quality Error Level Analysis (ELA) on an image.
    Runs ELA at quality levels 70, 80, and 90 and returns the MAXIMUM
    standard deviation across all three — making it robust against
    pre-existing re-compression (e.g. WhatsApp/social media images).
    Returns a score between 0.0 and 1.0 representing the anomaly level.
    High score (>0.6) suggests manipulation.
    """
    try:
        original = Image.open(image_path).convert('RGB')

        def _ela_std_at_quality(img: Image.Image, q: int) -> float:
            temp_path = image_path + f".temp_ela_{q}.jpg"
            try:
                img.save(temp_path, 'JPEG', quality=q)
                resaved = Image.open(temp_path)

                diff = ImageChops.difference(img, resaved)
                extrema = diff.getextrema()
                max_diff = max([ex[1] for ex in extrema])
                if max_diff == 0:
                    max_diff = 1

                scale = 255.0 / max_diff
                diff = ImageEnhance.Brightness(diff).enhance(scale)
                diff_array = np.array(diff)
                std_dev = float(np.std(diff_array))
                resaved.close()
                return std_dev
            except Exception:
                return 0.0
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

        # Multi-quality ELA — take the maximum std_dev for robustness
        std_devs = [_ela_std_at_quality(original, q) for q in [70, 80, 90]]
        std_dev = max(std_devs)

        # Normalize score (heuristic mapping)
        # Typical authentic images have std_dev < 30. Manipulated can be > 60.
        score = min(max((std_dev - 10) / 70.0, 0.0), 1.0)
        return float(score)

    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"ELA failed: {e}")
        return 0.0
