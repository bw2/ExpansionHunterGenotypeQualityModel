import gzip
import os
import tempfile
import unittest

import combine_genotyping_regimes_of_two_models as combine


def _model(tag, feature_names=("a", "b"), format_version=2):
    return {"format_version": format_version, "feature_names": list(feature_names),
            "genotyping_regimes": {"quick": {"from": tag}, "full_spanning": {"from": tag},
                                   "full_nonspanning": {"from": tag}}}


class CombineGenotypingRegimesTest(unittest.TestCase):

    def test_takes_quick_from_one_model_and_everything_else_from_the_other(self):
        combined = combine.combine_genotyping_regimes(_model("quick source"), _model("full source"))
        self.assertEqual(combined["genotyping_regimes"], {"quick": {"from": "quick source"},
                                                          "full_spanning": {"from": "full source"},
                                                          "full_nonspanning": {"from": "full source"}})
        self.assertEqual(list(combined), ["format_version", "feature_names", "genotyping_regimes"])
        self.assertEqual(list(combined["genotyping_regimes"]), ["quick", "full_spanning", "full_nonspanning"])

    def test_leaves_the_input_models_unchanged(self):
        quick_regime_model, full_genotyper_regimes_model = _model("quick source"), _model("full source")
        combine.combine_genotyping_regimes(quick_regime_model, full_genotyper_regimes_model)
        self.assertEqual(full_genotyper_regimes_model, _model("full source"))
        self.assertEqual(quick_regime_model, _model("quick source"))

    def test_refuses_models_with_different_feature_names_or_format_versions(self):
        with self.assertRaisesRegex(ValueError, "feature_names"):
            combine.combine_genotyping_regimes(_model("q", feature_names=("a", "c")), _model("f"))
        with self.assertRaisesRegex(ValueError, "format_version"):
            combine.combine_genotyping_regimes(_model("q", format_version=3), _model("f"))

    def test_refuses_a_model_with_unexpected_keys(self):
        model_with_extra_key = dict(_model("q"), extra=1)
        with self.assertRaisesRegex(ValueError, "unexpected model keys"):
            combine.combine_genotyping_regimes(model_with_extra_key, _model("f"))
        with self.assertRaisesRegex(ValueError, "unexpected model keys"):
            combine.combine_genotyping_regimes(_model("q"), model_with_extra_key)

    def test_written_model_round_trips_and_its_bytes_do_not_depend_on_when_or_where_it_was_written(self):
        combined = combine.combine_genotyping_regimes(_model("quick source"), _model("full source"))
        with tempfile.TemporaryDirectory() as tmp:
            first, second = os.path.join(tmp, "first.json.gz"), os.path.join(tmp, "second.json.gz")
            combine.write_model(combined, first)
            combine.write_model(combined, second)
            self.assertEqual(combine.load_model(first), combined)
            with open(first, "rb") as f1, open(second, "rb") as f2:
                first_bytes = f1.read()
                self.assertEqual(first_bytes, f2.read())
            self.assertEqual(first_bytes[4:8], b"\x00\x00\x00\x00")  # gzip header mtime
            self.assertEqual(first_bytes[3] & 0x08, 0)  # no file name in the gzip header
            with gzip.open(first, "rb") as f:
                text = f.read()
            self.assertNotIn(b", ", text)  # compact separators
            self.assertNotIn(b": ", text)


if __name__ == "__main__":
    unittest.main()
