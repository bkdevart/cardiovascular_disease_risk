"""
Shared utility functions for the CVD prediction pipeline.
"""

import os
import matplotlib.pyplot as plt
from src.config import FIG_DPI


def save_plot(fig, directory, filename):
    """Save a matplotlib figure and close it."""
    path = os.path.join(directory, filename)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def print_section(title):
    """Print a formatted section header."""
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_step(step):
    """Print a formatted step indicator."""
    print(f"\n── {step} {'─' * max(0, 60 - len(step))}")
