"""
Data loading, cleaning, and formatting utilities for the Nemotron competition.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Optional


DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def load_competition_data(split: str = "train") -> pd.DataFrame:
    """Load competition data from raw directory.
    
    Args:
        split: 'train' or 'test'
    
    Returns:
        DataFrame with competition problems
    """
    # Try common file formats
    for ext in [".csv", ".json", ".jsonl", ".parquet"]:
        path = RAW_DIR / f"{split}{ext}"
        if path.exists():
            if ext == ".csv":
                return pd.read_csv(path)
            elif ext == ".json":
                return pd.read_json(path)
            elif ext == ".jsonl":
                return pd.read_json(path, lines=True)
            elif ext == ".parquet":
                return pd.read_parquet(path)

    # Try finding any file with the split name
    candidates = list(RAW_DIR.glob(f"*{split}*"))
    if candidates:
        path = candidates[0]
        if path.suffix == ".csv":
            return pd.read_csv(path)
        elif path.suffix == ".json":
            return pd.read_json(path)
        elif path.suffix == ".jsonl":
            return pd.read_json(path, lines=True)
        elif path.suffix == ".parquet":
            return pd.read_parquet(path)

    raise FileNotFoundError(
        f"No {split} data found in {RAW_DIR}. "
        f"Run: kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge"
    )


def format_for_chat(problem: str, system_prompt: Optional[str] = None) -> list[dict]:
    """Format a problem into chat messages for the model.
    
    Args:
        problem: The problem text
        system_prompt: Optional system prompt
    
    Returns:
        List of chat message dicts
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": problem})
    return messages


def explore_data(df: pd.DataFrame) -> dict:
    """Generate summary statistics for the competition dataset."""
    info = {
        "num_rows": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample": df.head(3).to_dict(orient="records"),
    }

    # Check for common column names
    for col in ["category", "type", "subject", "difficulty", "topic"]:
        if col in df.columns:
            info[f"{col}_distribution"] = df[col].value_counts().to_dict()

    return info
