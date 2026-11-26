"""
FakeNewsNet Dataset Loader
==========================
Loads the FakeNewsNet news article classification dataset.
Default source: Kaggle pre-processed snapshot (no Twitter/X API key required).
Alternative: local CSV files if already collected via the official scraper.

Labels: REAL (0) | FAKE (1)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

LABEL2ID = {"REAL": 0, "FAKE": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def _load_config() -> Dict[str, Any]:
    cfg_path = Path(__file__).parent.parent / "config" / "datasets.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def load_from_kaggle(output_dir: Optional[str] = None) -> str:
    """
    Download FakeNewsNet from Kaggle using the kaggle CLI.
    Requires ~/.kaggle/kaggle.json with your Kaggle API credentials.

    Returns:
        Path to the extracted dataset directory.
    """
    import subprocess
    cfg = _load_config()
    fnn_cfg = cfg["datasets"]["fakenewsnet"]
    out = output_dir or fnn_cfg["local_dir"]
    os.makedirs(out, exist_ok=True)

    dataset_id = fnn_cfg["kaggle_dataset"]
    logger.info("Downloading FakeNewsNet from Kaggle: %s → %s", dataset_id, out)

    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_id, "-p", out, "--unzip"],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        logger.error("Kaggle download failed:\n%s\n%s", result.stdout, result.stderr)
        raise RuntimeError(
            f"Kaggle download failed. Make sure kaggle CLI is installed and "
            f"~/.kaggle/kaggle.json exists.\nSTDERR: {result.stderr}"
        )
    logger.info("Kaggle download complete: %s", out)
    return out


def load_from_csv(
    fake_csv: Optional[str] = None,
    real_csv: Optional[str] = None,
) -> Any:
    """
    Load FakeNewsNet from pre-existing local CSV files.
    Expected columns: 'title', 'text', 'label' (or separate fake/real CSVs).
    """
    import pandas as pd

    cfg = _load_config()
    fnn_cfg = cfg["datasets"]["fakenewsnet"]
    fake_path = fake_csv or fnn_cfg.get("local_csv_fake")
    real_path = real_csv or fnn_cfg.get("local_csv_real")

    frames = []
    if fake_path and os.path.exists(fake_path):
        df_fake = pd.read_csv(fake_path)
        df_fake["label"] = 1
        frames.append(df_fake)
    if real_path and os.path.exists(real_path):
        df_real = pd.read_csv(real_path)
        df_real["label"] = 0
        frames.append(df_real)

    if not frames:
        raise FileNotFoundError(
            f"No CSV files found at: fake={fake_path}, real={real_path}"
        )
    df = pd.concat(frames, ignore_index=True)
    logger.info("Loaded FakeNewsNet from local CSVs: %d rows", len(df))
    return df


def load_dataset_auto() -> Any:
    """
    Auto-detect data source and load FakeNewsNet & political news into a pandas DataFrame.
    Priority:
      1. Explicit FakeNewsNet content CSVs (PolitiFact / BuzzFeed fake & real)
      2. Augmentation with ISOT political news corpus if present (clean, high-volume)
      3. Fallback CSVs / Kaggle download if needed
    """
    import pandas as pd
    cfg = _load_config()
    fnn_cfg = cfg["datasets"]["fakenewsnet"]
    local_dir = fnn_cfg["local_dir"]

    frames = []

    # 1. Look for known FakeNewsNet content pairs
    if os.path.exists(local_dir):
        for csv_path in sorted(Path(local_dir).rglob("*_news_content.csv")):
            try:
                fname = csv_path.name.lower()
                df = pd.read_csv(csv_path, low_memory=False)
                if "fake" in fname:
                    df["label"] = 1  # FAKE
                elif "real" in fname:
                    df["label"] = 0  # REAL
                frames.append(df)
                logger.info("Loaded content CSV %s with %d rows", fname, len(df))
            except Exception as e:
                logger.warning("Error reading %s: %s", csv_path, e)

    # 2. Augment with ISOT political news if available for rich political coverage
    isot_dir = Path("data/datasets/isot")
    if (isot_dir / "Fake.csv").exists() and (isot_dir / "True.csv").exists():
        try:
            df_fake_isot = pd.read_csv(isot_dir / "Fake.csv", low_memory=False)
            df_true_isot = pd.read_csv(isot_dir / "True.csv", low_memory=False)
            
            pol_fake = df_fake_isot[df_fake_isot["subject"].isin(["politics", "Government News"])].copy()
            pol_true = df_true_isot[df_true_isot["subject"] == "politicsNews"].copy()
            
            # Sample balanced political news (up to 1,200 of each)
            n_sample = min(1200, len(pol_fake), len(pol_true))
            sample_fake = pol_fake.sample(n=n_sample, random_state=42).copy()
            sample_true = pol_true.sample(n=n_sample, random_state=42).copy()
            sample_fake["label"] = 1
            sample_true["label"] = 0
            
            frames.append(sample_fake)
            frames.append(sample_true)
            logger.info("Augmented with %d real and %d fake ISOT political news articles.", n_sample, n_sample)
        except Exception as e:
            logger.warning("Failed to load ISOT political news augmentation: %s", e)

    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = _normalize_columns(df)
        logger.info("Loaded %d FakeNewsNet articles with full text content.", len(df))
        return df

    # Fallback to FakeNewsNet.csv or any CSV
    if os.path.exists(local_dir):
        csv_candidates = list(Path(local_dir).rglob("*.csv"))
        if csv_candidates:
            frames = []
            for csv_path in csv_candidates:
                try:
                    df = pd.read_csv(csv_path, low_memory=False)
                    if "real" in df.columns and "label" not in df.columns:
                        # 1=real -> 0=REAL, 0=fake -> 1=FAKE
                        df["label"] = df["real"].apply(lambda r: 0 if r == 1 else 1)
                    frames.append(df)
                except Exception as e:
                    logger.warning("Failed to read %s: %s", csv_path, e)
            if frames:
                df = pd.concat(frames, ignore_index=True)
                df = _normalize_columns(df)
                return df

    # --- Fallback: Kaggle download ---
    logger.info("No local CSVs found. Attempting Kaggle download ...")
    out = load_from_kaggle(local_dir)
    csv_files = list(Path(out).rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found after Kaggle download. "
            "Please check the Kaggle dataset ID or manually place CSV files in: " + local_dir
        )
    frames = [pd.read_csv(f, low_memory=False) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)
    df = _normalize_columns(df)
    logger.info("FakeNewsNet loaded (Kaggle): %d rows", len(df))
    return df


def _normalize_columns(df: Any) -> Any:
    """Standardize column names, remove duplicates, and ensure balanced valid labels."""
    import pandas as pd

    rename_map = {}
    cols_lower = {c.lower(): c for c in df.columns}

    # Text column
    for cand in ["text", "content", "article", "body"]:
        if cand in cols_lower:
            rename_map[cols_lower[cand]] = "text"
            break
    # Title
    for cand in ["title", "headline"]:
        if cand in cols_lower:
            rename_map[cols_lower[cand]] = "title"
            break
    # Label
    for cand in ["label", "class", "fake", "is_fake"]:
        if cand in cols_lower and cols_lower[cand] not in rename_map:
            rename_map[cols_lower[cand]] = "label"
            break

    df = df.rename(columns=rename_map)

    # Ensure label column is numeric (0=REAL, 1=FAKE)
    if "label" in df.columns:
        label_map = {"real": 0, "fake": 1, "0": 0, "1": 1, 0: 0, 1: 1}
        df["label"] = df["label"].apply(
            lambda x: label_map.get(str(x).lower().strip(), -1)
        )
        df = df[df["label"].isin([0, 1])].reset_index(drop=True)

    # Ensure title and text are clean strings
    if "title" in df.columns:
        df["title"] = df["title"].fillna("").astype(str).str.strip()
    if "text" in df.columns:
        df["text"] = df["text"].fillna("").astype(str).str.strip()
        # Fall back to title if text is empty
        if "title" in df.columns:
            empty_mask = df["text"].str.len() == 0
            df.loc[empty_mask, "text"] = df.loc[empty_mask, "title"]

    # Drop rows that have neither title nor text
    has_text = (df["text"].str.len() > 10) | (df.get("title", pd.Series([""] * len(df))).str.len() > 5)
    df = df[has_text].reset_index(drop=True)

    # CRITICAL: Prevent data leakage and conflicting labels
    # If identical articles appear with different labels, remove duplicates
    if "title" in df.columns:
        df = df.drop_duplicates(subset=["title", "label"], keep="first")
        # If any title still appears under both labels, drop all occurrences to eliminate noise
        title_counts = df.groupby("title")["label"].nunique()
        conflict_titles = title_counts[title_counts > 1].index
        if len(conflict_titles) > 0:
            logger.warning("Dropping %d titles with conflicting real/fake labels.", len(conflict_titles))
            df = df[~df["title"].isin(conflict_titles)].reset_index(drop=True)

    # Balance the dataset evenly between REAL and FAKE
    real_df = df[df["label"] == 0]
    fake_df = df[df["label"] == 1]
    min_count = min(len(real_df), len(fake_df))
    if min_count > 0:
        real_df = real_df.sample(n=min_count, random_state=42)
        fake_df = fake_df.sample(n=min_count, random_state=42)
        df = pd.concat([real_df, fake_df], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    return df


def train_val_test_split(
    df: Any,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Any, Any, Any]:
    """
    Stratified split of FakeNewsNet DataFrame.
    Returns (train_df, val_df, test_df).
    """
    from sklearn.model_selection import train_test_split

    if "label" not in df.columns:
        raise ValueError("DataFrame must have a 'label' column.")

    df = df.dropna(subset=["label"]).reset_index(drop=True)
    temp_ratio = val_ratio + test_ratio

    train_df, temp_df = train_test_split(
        df, test_size=temp_ratio, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio / temp_ratio,
        stratify=temp_df["label"],
        random_state=seed,
    )
    logger.info(
        "FakeNewsNet split: train=%d, val=%d, test=%d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def validate_dataset(df: Any) -> Dict[str, Any]:
    """Run integrity checks on the loaded DataFrame."""
    from collections import Counter
    issues = []

    if len(df) == 0:
        return {"total": 0, "issues": ["Dataset is empty!"]}

    for col in ["text", "label"]:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")

    if "text" in df.columns:
        empty_text = df["text"].isna().sum() + (df["text"] == "").sum()
        if empty_text > 0:
            issues.append(f"{empty_text} rows with empty 'text' field.")

    label_dist = {}
    if "label" in df.columns:
        label_dist = dict(Counter(df["label"].tolist()))

    summary = {
        "total": len(df),
        "columns": list(df.columns),
        "label_distribution": label_dist,
        "issues": issues,
    }
    logger.info("[FakeNewsNet] Validation: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== FakeNewsNet Dataset Loader Test ===")
    df = load_dataset_auto()
    summary = validate_dataset(df)
    print(summary)
    train, val, test = train_val_test_split(df)
    print(f"Splits: train={len(train)}, val={len(val)}, test={len(test)}")

