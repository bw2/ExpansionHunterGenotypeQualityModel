"""Assembles training tables that include the spiked samples' rows, in a data folder of their own.

The new folder's ``real_quick/`` links to every per-source parquet of ``--base-data-dir`` and to every spiked sample's
parquet (``spike_in_expansions.py tables``), and its branch parquets are assembled from them with
``dataset.assemble_branch`` for the training populations asked for, dropping the rows whose truth the person's own
short reads contradict as the base tables do (none of the spiked samples' rows: their loci leave out those where the
person's reads contradict the truth of the allele they keep, and the simulated allele's truth is known). Train on it
with ``train.py --data-dir OUT_DATA_DIR``; ``train._calibration_people`` holds out the same people as without the spiked
samples.

Usage:
    python3 simulated_expansions/add_to_training_tables.py --base-data-dir DIR --out-data-dir DIR
"""

import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import dataset  # noqa: E402


def link_parts(base_data_dir, spiked_tables_dirs, out_data_dir):
    """Links the base per-source parquets and the spiked samples' parquets of every folder of ``spiked_tables_dirs``
    into ``out_data_dir/real_quick/``, in place of every link an earlier run left there (so a part that is gone is no
    longer trained on); returns the number of each."""
    out = os.path.join(out_data_dir, dataset.SOURCE_SUBDIR)
    if os.path.realpath(out) == os.path.realpath(os.path.join(base_data_dir, dataset.SOURCE_SUBDIR)):
        raise RuntimeError("--out-data-dir must not be --base-data-dir: linking would replace the base parquets")
    os.makedirs(out, exist_ok=True)
    base_parts = sorted(glob.glob(os.path.join(base_data_dir, dataset.SOURCE_SUBDIR, "*.parquet")))
    spiked_parts = sorted(part for spiked_tables_dir in spiked_tables_dirs
                          for part in glob.glob(os.path.join(spiked_tables_dir, "*.parquet")))
    if not base_parts or not spiked_parts:
        raise RuntimeError("found %d base and %d spiked parquets" % (len(base_parts), len(spiked_parts)))
    names = [os.path.basename(p) for p in base_parts + spiked_parts]
    clashes = {name for name in names if names.count(name) > 1}
    if clashes:
        raise RuntimeError("parquets of the base and spiked folders share names: %s" % sorted(clashes))
    for name in os.listdir(out):
        if os.path.islink(os.path.join(out, name)):
            os.remove(os.path.join(out, name))
    for part in base_parts + spiked_parts:
        os.symlink(os.path.realpath(part), os.path.join(out, os.path.basename(part)))
    return len(base_parts), len(spiked_parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-data-dir", required=True)
    parser.add_argument("--spiked-tables-dir", action="append",
                        help="a folder of spiked samples' per-source parquets (spike_in_expansions.py tables); "
                             "repeatable; default: work/tables")
    parser.add_argument("--out-data-dir", required=True)
    parser.add_argument("--training-population", choices=sorted(dataset.BRANCH_PARQUET_SUBDIR_BY_TRAINING_POPULATION),
                        action="append",
                        help="which branch parquets to assemble, as for dataset.py: variant_loci and/or all_loci "
                             "(which needs base parquets built with dataset.py --keep-hom-ref-loci); repeatable; "
                             "default variant_loci")
    args = parser.parse_args()
    n_base, n_spiked = link_parts(args.base_data_dir, args.spiked_tables_dir or [os.path.join(HERE, "work", "tables")],
                                  args.out_data_dir)
    print("linked %d base and %d spiked per-source parquets into %s" % (n_base, n_spiked, args.out_data_dir))
    for population in args.training_population or ["variant_loci"]:
        for branch in ("quick", "full"):
            dataset.assemble_branch(args.out_data_dir, branch, dataset.SOURCE_SUBDIR, training_population=population)


if __name__ == "__main__":
    main()
