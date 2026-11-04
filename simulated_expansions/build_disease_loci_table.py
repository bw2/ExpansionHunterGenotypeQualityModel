"""Builds the table of known disease loci to simulate expansions at, and an ExpansionHunter catalog of them.

Each known pathogenic locus of the str_analysis package (``str_analysis/data/known_pathogenic_str_loci_metadata_hg38_hg19
.tsv``) is matched to an overlapping locus of a training run's ExpansionHunter JSON with a motif of the same length,
preferring one whose motif matches (any rotation, either strand, N matching any base), then the one overlapping it most.
The JSON is used rather than the catalog file because its records carry each locus's ReferenceRegion and
RepeatUnit as the training runs genotyped them, and the catalog file may have changed since. A locus with no such match
is left out and listed.

The expanded motif is the locus's pathogenic motif where it differs from the reference one (e.g. AAGGG at RFC1), else
the training catalog's motif, written on the reference's + strand (``plus_strand_pathogenic_motif``).

Each locus also gets the smallest and largest pathogenic alleles reported there, in bp, from STRchive (the curated
database of pathogenic tandem repeats, https://github.com/dashnowlab/STRchive;
``overlapping_strchive_ids_and_pathogenic_allele_range_bp``), which set the sizes simulated there: the largest caps
them, as polyalanine expansions such as PABPN1's or ARX's, and several polyglutamine ones such as AR's, stay well under
300 bp in patients, and loci with pathogenic alleles under 150 bp get sizes from that range.

Usage:
    python3 simulated_expansions/build_disease_loci_table.py [--strchive-loci-json PATH_OR_URL] TRAINING_EH_JSON OUT_DIR

Writes ``OUT_DIR/disease_loci.tsv`` and ``OUT_DIR/disease_loci.catalog.json``.
"""

import argparse
import gzip
import json
import os
import re
import sys
import urllib.request

import ijson
import pandas as pd
import str_analysis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulated_expansions.expanded_allele_reads import reverse_complement  # noqa: E402

KNOWN_LOCI_TSV = os.path.join(os.path.dirname(str_analysis.__file__), "data",
                              "known_pathogenic_str_loci_metadata_hg38_hg19.tsv")
LOCATION_COLUMN = "TR Location (hg38, 1-based)       NOTE: IGV is also 1-based"
# STRchive's locus records at a fixed version of its repository, so the table can be rebuilt the same way.
STRCHIVE_LOCI_JSON = ("https://raw.githubusercontent.com/dashnowlab/STRchive/"
                      "e964fa88df3b2189ee478cbd6eaad15bb28e3a0a/data/STRchive-loci.json")


def rotations(motif):
    return [motif[i:] + motif[:i] for i in range(len(motif))]


def motif_family(motif):
    """The alphabetically first rotation of the motif or of its reverse complement, which all motifs of one repeat
    share (CAG, AGC, CTG and GCT all give AGC). N is read as a letter."""
    return min(rotations(motif.upper()) + rotations(reverse_complement(motif.upper())))


def mismatches(a, b):
    """The number of positions where two equal-length motifs differ, N matching any base."""
    return sum(x != y and "N" not in (x, y) for x, y in zip(a.upper(), b.upper()))


def distance_between_motifs(a, b):
    """The fewest mismatches between ``b`` and any rotation of ``a`` (equal lengths), on the same strand."""
    return min(mismatches(rotation, b) for rotation in rotations(a))


def motifs_match(a, b):
    """True when a rotation of ``a`` or of its reverse complement equals ``b`` (N matching any base)."""
    return len(a) == len(b) and min(distance_between_motifs(a, b),
                                    distance_between_motifs(reverse_complement(a), b)) == 0


def plus_strand_pathogenic_motif(pathogenic_motif, plus_strand_reference_motif):
    """Returns the pathogenic motif on the reference's + strand: the motif or its reverse complement, whichever is
    closer to the + strand reference motif. The metadata writes some pathogenic motifs in the gene's orientation (DAB1's
    ATTTC sits in an AAAAT repeat on the + strand, where it reads GAAAT)."""
    turned = reverse_complement(pathogenic_motif)
    if (distance_between_motifs(turned, plus_strand_reference_motif)
            < distance_between_motifs(pathogenic_motif, plus_strand_reference_motif)):
        return turned
    return pathogenic_motif


def known_disease_loci(tsv_path=KNOWN_LOCI_TSV):
    """Returns the known pathogenic loci as a DataFrame: disease_locus_id, gene, chrom, start_0based, end,
    metadata_motif, metadata_pathogenic_motif, normal_max, pathogenic_min and inheritance."""
    table = pd.read_csv(tsv_path, sep="\t")
    location = table[LOCATION_COLUMN].str.extract(r"^(chr[^:]+):(\d+)-(\d+)$")
    # ANALYSIS_OK[imputation]: a missing or blank pathogenic motif becomes "", which the reference motif replaces below.
    pathogenic_motif = table["Pathogenic Motif"].fillna("").str.split().str[0].fillna("")
    return pd.DataFrame({
        "disease_locus_id": table["LocusId"], "gene": table["Gene"], "chrom": location[0],
        "start_0based": location[1].astype(int) - 1, "end": location[2].astype(int),
        "metadata_motif": table["RU"].str.upper(),
        # ANALYSIS_OK[imputation]: an empty pathogenic motif means the expansion is of the reference motif.
        "metadata_pathogenic_motif": pathogenic_motif.where(pathogenic_motif != "", table["RU"]).str.upper(),
        "normal_max": pd.to_numeric(table["NormalMax"], errors="coerce"),
        "pathogenic_min": pd.to_numeric(table["PathogenicMin"], errors="coerce"),
        "inheritance": table["InheritanceMode"]})


def training_loci_overlapping(eh_json_path, intervals):
    """Streams a training run's EH JSON and returns its single-repeat loci that overlap any of ``intervals``.

    Args:
        eh_json_path: The EH JSON (.json.gz).
        intervals: ``{chrom: [(start_0based, end), ...]}``.

    Returns:
        A list of dicts: locus_id, chrom, start_0based, end, motif.
    """
    found = []
    with gzip.open(eh_json_path, "rb") as f:
        for locus_id, locus in ijson.kvitems(f, "LocusResults", use_float=True):
            variants = list((locus.get("Variants") or {}).values())
            if len(variants) != 1:
                continue
            chrom, start, end = re.match(r"^(chr[^:]+):(\d+)-(\d+)$", variants[0]["ReferenceRegion"]).groups()
            start, end = int(start), int(end)
            if any(start < e and s < end for s, e in intervals.get(chrom, ())):
                found.append({"locus_id": locus_id, "chrom": chrom, "start_0based": start, "end": end,
                              "motif": variants[0]["RepeatUnit"]})
    return found


def match_training_locus(disease_locus, training_loci):
    """The training locus overlapping ``disease_locus`` with a motif of the same length, preferring one whose motif
    matches the metadata's (``motifs_match``), then the one overlapping it most; None if there is none."""
    def preference(locus):
        overlap = min(locus["end"], disease_locus["end"]) - max(locus["start_0based"], disease_locus["start_0based"])
        return motifs_match(disease_locus["metadata_motif"], locus["motif"]), overlap

    candidates = [locus for locus in training_loci if locus["chrom"] == disease_locus["chrom"]
                  and len(locus["motif"]) == len(disease_locus["metadata_motif"]) and preference(locus)[1] > 0]
    return max(candidates, key=preference, default=None)


def load_strchive_loci(path_or_url):
    """STRchive's locus records (a list of dicts), from a local copy or a URL of ``STRchive-loci.json``."""
    if path_or_url.startswith("https://"):
        with urllib.request.urlopen(path_or_url) as response:
            return json.load(response)
    with open(path_or_url) as f:
        return json.load(f)


def overlapping_strchive_ids_and_pathogenic_allele_range_bp(locus, strchive_loci):
    """``(STRchive ids, smallest bp, largest bp)``: the STRchive loci overlapping ``locus`` (a dict with chrom,
    start_0based and end on hg38), and the smallest and largest pathogenic alleles they report (``pathogenic_min`` and
    ``pathogenic_max`` copies of their motif), in bp, each NaN where none reports one."""
    matches = [entry for entry in strchive_loci if entry.get("chrom") == locus["chrom"]
               and entry.get("start_hg38") is not None and entry["start_hg38"] < locus["end"]
               and locus["start_0based"] < entry["stop_hg38"]]
    smallest = [entry["pathogenic_min"] * entry["motif_len"] for entry in matches if entry.get("pathogenic_min")]
    largest = [entry["pathogenic_max"] * entry["motif_len"] for entry in matches if entry.get("pathogenic_max")]
    return (",".join(sorted(entry["id"] for entry in matches)), min(smallest) if smallest else float("nan"),
            max(largest) if largest else float("nan"))


def build_table(disease_loci, training_loci, strchive_loci):
    """Returns ``(table of matched loci, list of unmatched disease locus ids)``. Each locus also gets the smallest and
    largest pathogenic alleles STRchive reports there (``overlapping_strchive_ids_and_pathogenic_allele_range_bp``)."""
    rows, unmatched = [], []
    for disease_locus in disease_loci.to_dict("records"):
        locus = match_training_locus(disease_locus, training_loci)
        if locus is None:
            unmatched.append(disease_locus["disease_locus_id"])
            continue
        pathogenic_motif = plus_strand_pathogenic_motif(disease_locus["metadata_pathogenic_motif"], locus["motif"])
        strchive_ids, smallest_pathogenic_bp, largest_pathogenic_bp = (
            overlapping_strchive_ids_and_pathogenic_allele_range_bp(locus, strchive_loci))
        rows.append({**{k: disease_locus[k] for k in ("disease_locus_id", "gene", "normal_max", "pathogenic_min",
                                                       "inheritance")},
                     "locus_id": locus["locus_id"], "chrom": locus["chrom"],
                     "start_0based": locus["start_0based"], "end": locus["end"],
                     "reference_motif": locus["motif"],
                     "expanded_motif": locus["motif"] if motifs_match(pathogenic_motif, locus["motif"])
                     else pathogenic_motif,
                     "strchive_ids": strchive_ids, "largest_reported_pathogenic_bp": largest_pathogenic_bp,
                     "smallest_reported_pathogenic_bp": smallest_pathogenic_bp})
    return pd.DataFrame(rows), unmatched


def catalog_entries(table):
    """ExpansionHunter catalog entries for the table's loci, as the training runs defined them."""
    return [{"LocusId": row["locus_id"], "LocusStructure": "(%s)*" % row["reference_motif"],
             "ReferenceRegion": "%s:%d-%d" % (row["chrom"], row["start_0based"], row["end"]),
             "VariantType": "Repeat"} for row in table.to_dict("records")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strchive-loci-json", default=STRCHIVE_LOCI_JSON,
                        help="STRchive's STRchive-loci.json, a local copy or an https URL (default: a fixed version)")
    parser.add_argument("training_eh_json")
    parser.add_argument("out_dir")
    args = parser.parse_args()
    disease_loci = known_disease_loci()
    intervals = {}
    for row in disease_loci.to_dict("records"):
        intervals.setdefault(row["chrom"], []).append((row["start_0based"], row["end"]))
    training_loci = training_loci_overlapping(args.training_eh_json, intervals)
    table, unmatched = build_table(disease_loci, training_loci, load_strchive_loci(args.strchive_loci_json))
    os.makedirs(args.out_dir, exist_ok=True)
    table.to_csv(os.path.join(args.out_dir, "disease_loci.tsv"), sep="\t", index=False)
    with open(os.path.join(args.out_dir, "disease_loci.catalog.json"), "w") as f:
        json.dump(catalog_entries(table), f, indent=1)
    print("%d of %d known disease loci matched a training locus; unmatched: %s"
          % (len(table), len(disease_loci), ", ".join(unmatched) or "none"))
    print(table[["disease_locus_id", "locus_id", "reference_motif", "expanded_motif", "pathogenic_min",
                 "smallest_reported_pathogenic_bp", "largest_reported_pathogenic_bp"]].to_string(index=False))
    strchive_pathogenic_min = table["smallest_reported_pathogenic_bp"] / table["reference_motif"].str.len()
    print("\nLoci where STRchive's smallest pathogenic allele is not the metadata's pathogenic_min (copies):")
    print(table.assign(strchive_pathogenic_min=strchive_pathogenic_min)[
        strchive_pathogenic_min != table["pathogenic_min"]][
        ["disease_locus_id", "gene", "pathogenic_min", "strchive_pathogenic_min"]].to_string(index=False))


if __name__ == "__main__":
    main()
