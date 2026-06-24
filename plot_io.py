"""Shared matplotlib figure-saving helper (SVG + PNG)."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save_svg_png(fig, out_prefix):
    """Saves ``fig`` to ``out_prefix.svg`` and ``out_prefix.png`` and closes it.

    The parent directory of ``out_prefix`` is created if needed.

    Args:
        fig: A matplotlib ``Figure``.
        out_prefix: Path prefix (no extension); ``.svg`` / ``.png`` are appended.

    Returns:
        A ``(svg_path, png_path)`` tuple of the written files.
    """
    parent = os.path.dirname(out_prefix)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(f"{out_prefix}.svg", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"{out_prefix}.svg", f"{out_prefix}.png"
