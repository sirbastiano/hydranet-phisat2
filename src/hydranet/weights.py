"""Download and manage HydraNet model weights from HuggingFace."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from huggingface_hub import hf_hub_download

# HuggingFace repository
HF_REPO = "sirbastiano94/hydranet"

# Path to the model catalog CSV
_CATALOG_PATH = Path(__file__).parent / "model_weights.csv"


def _load_catalog() -> pd.DataFrame:
    """Load the model weights catalog from CSV."""
    if not _CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Model catalog not found at {_CATALOG_PATH}. "
            "Please run the notebook to generate it."
        )
    return pd.read_csv(_CATALOG_PATH)


def get_model_weights(
    training: str,
    model: str,
    task: str,
    n_shots: int,
    download_dir: Optional[str] = None,
    verbose: bool = True,
    latest_only: bool = True,
) -> list[str]:
    """
    Download model weights from HuggingFace based on specifications.
    
    Args:
        training: Training type (e.g., 'finetuning', 'linear_probing')
        model: Model name (e.g., 'student', 'teacher')
        task: Task name (e.g., 'anomaly_detection', 'worldfloods')
        n_shots: Number of shots (e.g., 50, 100, 500, 1000, 5000)
        download_dir: Directory to save weights (default: HuggingFace cache)
        verbose: Print download progress (default: True)
        latest_only: Download only the most recent checkpoint (default: True)
    
    Returns:
        List of local file paths to downloaded weights (single file if latest_only=True)
    
    Example:
        >>> weights = get_model_weights(
        ...     training='finetuning',
        ...     model='student',
        ...     task='anomaly_detection',
        ...     n_shots=1000,
        ...     download_dir='./weights'
        ... )
        Found latest checkpoint from 20260108
        ...
    """
    # Load catalog
    df = _load_catalog()
    
    # Query the DataFrame
    results = df[
        (df['training'] == training)
        & (df['model'] == model)
        & (df['task'] == task)
        & (df['n_shots'] == n_shots)
    ]
    
    if len(results) == 0:
        msg = f"No weights found for: {training}/{model}/{task}/nshot{n_shots}"
        if verbose:
            print(msg)
        raise ValueError(msg)
    
    # Filter for latest checkpoint if requested
    if latest_only and 'datetime' in results.columns:
        # Sort by datetime descending and take the first (latest)
        results = results.sort_values(by='datetime', ascending=False).head(1)  # type: ignore
        if verbose and len(results) > 0:
            print(f"Found latest checkpoint from {results.iloc[0]['datetime']}")
    else:
        if verbose:
            print(f"Found {len(results)} checkpoint(s)")
    
    # Download files
    local_paths = []
    for _, row in results.iterrows():
        if verbose:
            print(f"  Downloading: {row['filename']}")
        
        local_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=row['file_path'],
            repo_type="dataset",
            local_dir=download_dir,
        )
        local_paths.append(local_path)
        
        if verbose:
            print(f"  Saved to: {local_path}")
    
    return local_paths


def list_available_weights() -> pd.DataFrame:
    """
    List all available model weights with their specifications.
    
    Returns:
        DataFrame with columns: training, model, task, n_shots, file_path, filename
    
    Example:
        >>> df = list_available_weights()
        >>> print(df.groupby(['training', 'model', 'task', 'n_shots']).size())
    """
    return _load_catalog()


def get_available_combinations() -> pd.DataFrame:
    """
    Get all unique combinations of training/model/task/n_shots.
    
    Returns:
        DataFrame with columns: training, model, task, n_shots, count
    
    Example:
        >>> combos = get_available_combinations()
        >>> print(combos)
    """
    df = _load_catalog()
    return (
        df.groupby(['training', 'model', 'task', 'n_shots'])
        .size()
        .reset_index(name='count')
    )
