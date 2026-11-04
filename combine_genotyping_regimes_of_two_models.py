"""Builds a genotype quality model from the quick (fast-path) regime of one model and the full-genotyper regimes of
another.

``train.py`` fits each genotyping regime's heads independently of the other regimes', so a model whose regimes come from
two training runs is as valid as one from a single run. ``train_model.sh`` uses this to give each regime its own
training recipe: the full-genotyper regimes train on the variant-locus tables, the quick regime on all loci. The two
models must declare the same feature names and format version. The output is written the way ``train.py`` writes
models (gzip with mtime 0 and no file name), so the same inputs give the same bytes.

Usage:
    python3 combine_genotyping_regimes_of_two_models.py \\
        --model-to-take-quick-regime-from QUICK_MODEL.json.gz \\
        --model-to-take-full-genotyper-regimes-from FULL_MODEL.json.gz \\
        --out MODEL.json.gz

Coding rules: no type hints, Google docstrings, ``print()``.
"""

import argparse
import gzip
import json

MODEL_KEYS = {"format_version", "feature_names", "genotyping_regimes"}
FULL_GENOTYPER_REGIMES = ("full_spanning", "full_nonspanning")


def load_model(path):
    """Returns the model JSON at ``path`` (gzipped) as a dict."""
    with gzip.open(path, "rb") as f:
        return json.loads(f.read())


def combine_genotyping_regimes(quick_regime_model, full_genotyper_regimes_model):
    """Returns ``full_genotyper_regimes_model`` with its quick regime replaced by ``quick_regime_model``'s.

    Args:
        quick_regime_model: Model dict whose ``genotyping_regimes["quick"]`` is kept.
        full_genotyper_regimes_model: Model dict whose other keys and full-genotyper regimes are kept.

    Returns:
        The combined model dict, with the keys and regimes in ``full_genotyper_regimes_model``'s order.

    Raises:
        ValueError: if the models differ in ``format_version`` or ``feature_names``, or either has keys other than
            ``MODEL_KEYS``.
    """
    for model in (quick_regime_model, full_genotyper_regimes_model):
        if set(model) != MODEL_KEYS:
            raise ValueError("unexpected model keys %s (expected %s)" % (sorted(model), sorted(MODEL_KEYS)))
    for key in ("format_version", "feature_names"):
        if quick_regime_model[key] != full_genotyper_regimes_model[key]:
            raise ValueError("the two models differ in %s" % key)
    return dict(full_genotyper_regimes_model,
                genotyping_regimes=dict(full_genotyper_regimes_model["genotyping_regimes"],
                                        quick=quick_regime_model["genotyping_regimes"]["quick"]))


def write_model(model, path):
    """Writes ``model`` to ``path`` as compact JSON, gzipped at level 9 with mtime 0 and no file name."""
    blob = json.dumps(model, separators=(",", ":"), allow_nan=False).encode()
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=raw, mtime=0) as f:
            f.write(blob)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-to-take-quick-regime-from", required=True)
    parser.add_argument("--model-to-take-full-genotyper-regimes-from", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    quick_regime_model = load_model(args.model_to_take_quick_regime_from)
    full_genotyper_regimes_model = load_model(args.model_to_take_full_genotyper_regimes_from)
    try:
        combined = combine_genotyping_regimes(quick_regime_model, full_genotyper_regimes_model)
    except ValueError as e:
        parser.error(str(e))
    write_model(combined, args.out)
    written = load_model(args.out)
    assert written["genotyping_regimes"]["quick"] == quick_regime_model["genotyping_regimes"]["quick"]
    for regime in FULL_GENOTYPER_REGIMES:
        assert written["genotyping_regimes"][regime] == full_genotyper_regimes_model["genotyping_regimes"][regime]
    print("wrote %s: quick regime from %s, full-genotyper regimes from %s" % (
        args.out, args.model_to_take_quick_regime_from, args.model_to_take_full_genotyper_regimes_from))


if __name__ == "__main__":
    main()
