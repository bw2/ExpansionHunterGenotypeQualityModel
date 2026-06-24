"""Training diagnostics for the EH genotype-quality models.

These plots read the *outputs* of the model modules (the ``warm_start`` monitor
``history`` from ``model_q`` / ``model_direction``) and never the model modules
themselves, so they are independently testable (SPEC sec ``diagnostics.py``).

One diagnostic lives here:

- ``plot_loss_vs_iteration`` -- train vs calibration loss as boosting proceeds,
  marking the minimum-calib iteration (the early-stop point the monitor chose).

Determinism: pure apart from writing the figure files; no module-level mutable
state. Plots are rendered with the non-interactive ``Agg`` backend so the module
works headless, and every figure is saved as both ``.svg`` and ``.png``.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_io import _save_svg_png


def plot_loss_vs_iteration(history, out_prefix, title="loss vs boosting iteration"):
    """Plots train and calib loss against boosting iteration; marks min-calib.

    ``history`` is the monitor log produced by the model modules' early-stopping
    loop: a list of ``{"n_iter", "train_loss", "calib_loss"}`` dicts (one per
    monitor step). The iteration with the smallest calib loss is the early-stop
    point and is highlighted with a marker and a dashed vertical line.

    Args:
        history: List of ``{"n_iter", "train_loss", "calib_loss"}`` dicts.
        out_prefix: Path prefix for the saved ``.svg`` / ``.png`` figures.
        title: Plot title.

    Returns:
        The ``n_iter`` at which calib loss is minimal (the chosen stop iteration).

    Raises:
        ValueError: If ``history`` is empty.
    """
    if not history:
        raise ValueError("history is empty; nothing to plot")

    n_iter = np.array([step["n_iter"] for step in history], dtype=float)
    train_loss = np.array([step["train_loss"] for step in history], dtype=float)
    calib_loss = np.array([step["calib_loss"] for step in history], dtype=float)
    best = int(np.argmin(calib_loss))
    best_n_iter = int(n_iter[best])

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(n_iter, train_loss, "-o", color="#1f77b4", label="train loss", markersize=4)
    ax.plot(n_iter, calib_loss, "-o", color="#d62728", label="calib loss", markersize=4)
    ax.axvline(best_n_iter, linestyle="--", color="gray", linewidth=1)
    ax.scatter([best_n_iter], [calib_loss[best]], s=80, facecolors="none",
               edgecolors="#d62728", zorder=5,
               label=f"min calib @ n_iter={best_n_iter}")
    ax.set_xlabel("boosting iterations (max_iter)")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_svg_png(fig, out_prefix)
    return best_n_iter
