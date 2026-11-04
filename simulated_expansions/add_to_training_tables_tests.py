import os
import tempfile
import unittest

import dataset
from simulated_expansions import add_to_training_tables as add


def touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


class LinkPartsTest(unittest.TestCase):
    def test_links_base_and_spiked_parts_in_place_of_an_earlier_runs_links(self):
        with tempfile.TemporaryDirectory() as d:
            base, spiked, out = (os.path.join(d, name) for name in ("base", "spiked", "out"))
            touch(os.path.join(base, dataset.SOURCE_SUBDIR, "HG01106.parquet"))
            touch(os.path.join(spiked, "HG01106_spiked_0.parquet"))
            touch(os.path.join(spiked, "HG01106_spiked_1.parquet"))
            self.assertEqual(add.link_parts(base, [spiked], out), (1, 2))
            os.remove(os.path.join(spiked, "HG01106_spiked_1.parquet"))
            self.assertEqual(add.link_parts(base, [spiked], out), (1, 1))
            linked = sorted(os.listdir(os.path.join(out, dataset.SOURCE_SUBDIR)))
        self.assertEqual(linked, ["HG01106.parquet", "HG01106_spiked_0.parquet"])

    def test_links_the_parts_of_several_spiked_folders_and_refuses_a_name_two_share(self):
        with tempfile.TemporaryDirectory() as d:
            base, first, second, out = (os.path.join(d, name) for name in ("base", "first", "second", "out"))
            touch(os.path.join(base, dataset.SOURCE_SUBDIR, "HG01106.parquet"))
            touch(os.path.join(first, "HG01106_spiked_0.parquet"))
            touch(os.path.join(second, "HG01106_spiked_100000.parquet"))
            self.assertEqual(add.link_parts(base, [first, second], out), (1, 2))
            touch(os.path.join(second, "HG01106_spiked_0.parquet"))
            with self.assertRaises(RuntimeError):
                add.link_parts(base, [first, second], out)

    def test_refuses_to_link_into_the_base_folder(self):
        with tempfile.TemporaryDirectory() as d:
            touch(os.path.join(d, dataset.SOURCE_SUBDIR, "HG01106.parquet"))
            touch(os.path.join(d, "spiked", "HG01106_spiked_0.parquet"))
            with self.assertRaises(RuntimeError):
                add.link_parts(d, [os.path.join(d, "spiked")], d)
            self.assertTrue(os.path.isfile(os.path.join(d, dataset.SOURCE_SUBDIR, "HG01106.parquet")))


if __name__ == "__main__":
    unittest.main()
