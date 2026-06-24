"""Unit tests for ``diagnostics`` using hand-made inputs.

No model modules are imported: the loss history is a constructed list of dicts.
Figures are written under a per-test temporary directory.

Run with:  python3 -m unittest diagnostics_tests -v
"""

import os
import tempfile
import unittest

import diagnostics as D


def _assert_svg_png(test, out_prefix):
    """Asserts both ``out_prefix.svg`` and ``out_prefix.png`` exist and are non-empty."""
    for ext in ("svg", "png"):
        path = f"{out_prefix}.{ext}"
        test.assertTrue(os.path.exists(path), f"missing {path}")
        test.assertGreater(os.path.getsize(path), 0, f"empty {path}")


class PlotLossVsIterationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_returns_argmin_calib_n_iter(self):
        # Calib loss is U-shaped with its minimum at n_iter=75 (index 2).
        history = [
            {"n_iter": 25, "train_loss": 1.0, "calib_loss": 0.9},
            {"n_iter": 50, "train_loss": 0.8, "calib_loss": 0.6},
            {"n_iter": 75, "train_loss": 0.6, "calib_loss": 0.5},
            {"n_iter": 100, "train_loss": 0.4, "calib_loss": 0.55},
            {"n_iter": 125, "train_loss": 0.3, "calib_loss": 0.7},
        ]
        out = os.path.join(self.tmp.name, "loss_curve")
        self.assertEqual(D.plot_loss_vs_iteration(history, out, title="q median"), 75)
        _assert_svg_png(self, out)

    def test_single_step_history(self):
        out = os.path.join(self.tmp.name, "loss_single")
        self.assertEqual(
            D.plot_loss_vs_iteration(
                [{"n_iter": 25, "train_loss": 1.0, "calib_loss": 0.9}], out), 25)
        _assert_svg_png(self, out)

    def test_empty_history_raises(self):
        with self.assertRaises(ValueError):
            D.plot_loss_vs_iteration([], os.path.join(self.tmp.name, "empty"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
