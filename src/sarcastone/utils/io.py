"""Input/Output utility functions for SarcasTone."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pandas as pd


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file safely with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Save data to JSON file with UTF-8 encoding."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def load_numpy_feature(path: Union[str, Path]) -> np.ndarray:
    """Load a single .npy feature file."""
    return np.load(path)


def save_numpy_feature(array: np.ndarray, path: Union[str, Path]) -> None:
    """Save a numpy array to .npy file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, array)

