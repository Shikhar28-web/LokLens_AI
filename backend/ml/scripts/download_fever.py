"""
Quick script to download and validate FEVER dataset, then run baseline training.
Run from project root: .\venv\Scripts\python.exe ml\scripts\download_and_train_fever.py
"""
import sys
import logging
sys.path.insert(0, '.')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('models/fever_claim_verifier/download.log', mode='w'),
    ]
)

from pathlib import Path
Path('models/fever_claim_verifier').mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

logger.info("=== Step 1: Download FEVER NLI dataset from HuggingFace ===")
from ml.datasets.fever_loader import load_all_splits, validate_dataset

train, val, test = load_all_splits(use_nli=True)

for split_name, ds in [('train', train), ('val', val), ('test', test)]:
    s = validate_dataset(ds, 'FEVER-NLI-' + split_name)
    logger.info("  %s: total=%d, labels=%s, issues=%s",
                split_name, s['total'], s['label_distribution'], s['issues'])

logger.info("=== FEVER download and validation complete ===")
logger.info("Dataset sizes: train=%d, val=%d, test=%d", len(train), len(val), len(test))

