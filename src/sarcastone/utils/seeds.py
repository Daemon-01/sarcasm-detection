"""Reproducibility utilities and random seed configuration.

Enforces deterministic execution in accordance with AGENTS.md Section 7 & 9:
- Multi-seed experimental protocol (seeds: 42, 123, 2026).
- Deterministic seeding for Python random, NumPy, and PyTorch (if available).
"""

from __future__ import annotations

import logging
import os
import random
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# Official 3-seed protocol specified in AGENTS.md Section 7
EXPERIMENT_SEEDS: List[int] = [42, 123, 2026]


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed integer.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    logger.debug("Random seed set to %d", seed)

