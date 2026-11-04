# Simulated expansions for training

Healthy genomes have almost no alleles longer than ExpansionHunter can measure. So in the training tables, a long call
without spanning reads at a short, pure reference repeat is nearly always an over-call, and the model learns to say
"too long" there. It says the same about real pathogenic expansions, which ExpansionHunter under-calls once they outgrow
the read pairs. This folder makes training rows for such alleles: it adds simulated reads of an expanded allele to a real
person's reads at known disease loci and genotypes them the way the training runs did.

Everything but the expanded allele is the person's own reads, so the evidence around it (coverage, reads mis-mapped from
other loci, base qualities, mates) is real. The simulated reads take their fragment lengths and base-quality strings
from the same person's reads, and are aligned to the whole genome, so in-repeat read pairs land wherever the aligner
puts a real expansion's. Inside the repeat, real reads have more low-quality bases (more the richer the motif is in G
and C, and the further into the read), more mismatches, and gained or lost bases (whole motif copies and partial ones).
The simulated reads get these there as measured in the real reads of long repeats of 6 pilot people
(`spike_in_expansions.PILOT_PEOPLE`, the people `calibrate_simulated_reads.py` uses by default, all among the 45 spiked
ones) in `repeat_read_errors.tsv`, from `calibrate_simulated_reads.py measure-read-errors` (see below).

## Steps

The steps need the repository's own requirements plus this folder's (`pip3 install -r requirements.txt -r
simulated_expansions/requirements.txt`), and samtools and bwa on the machine.

`make_spiked_samples.sh` makes the spiked samples the shipped model was trained on, from the 45 people in
`spike_in_expansions.SPIKED_PEOPLE`:

```bash
REFERENCE_FASTA=HG38_FA BILLING_PROJECT=PROJECT GCS_DIR=gs://... simulated_expansions/make_spiked_samples.sh
```

It runs `spike_in_expansions.py`'s steps in this order; naming steps runs only those (e.g.
`simulated_expansions/make_spiked_samples.sh genotype tables`):

| step | what it makes | where |
|---|---|---|
| `make-minicrams` | each person's reads at the disease loci ("minicrams"), from their CRAMs (`short_read_paths.tsv` at the repository root) | `work_spiked_samples/minicrams/` |
| `check-parity` | on Hail Batch: checks that the un-spiked minicrams give exactly the training runs' rows at those loci | |
| `spike` | the spiked samples: simulated reads of expanded alleles aligned with bwa and merged into the person's (see below) | `work_spiked_samples/spiked/` |
| `genotype` | on Hail Batch: their ExpansionHunter calls | `work_spiked_samples/spiked/` |
| `tables` | their per-source parquets | `work_spiked_samples/tables/` |
| `downsample` | a copy of each spiked sample at 15-30x and a matched control of the person's own reads (see "Lower coverage") | `work_lower_coverage_copies_of_spiked_samples/spiked/` |
| `genotype-lower-coverage`, `tables-lower-coverage` | the same two steps for those | `work_lower_coverage_copies_of_spiked_samples/tables/` |

Where the work folders already hold spiked samples, such as the shipped ones, `spike` and `downsample` refuse to replace
them: to make them again next to the shipped ones, set `SPIKED_SAMPLES_WORK_DIR` and `LOWER_COVERAGE_WORK_DIR` to new
folders, compare the new tables with the shipped ones, and then point `train_model.sh`'s `SPIKED_TABLES_DIRS` (and
`DATA_DIR`) at them; `REPLACE_EXISTING_SPIKED_SAMPLES=1` replaces them in place instead, after moving the folder's
`spiked/` aside (to `spiked.replaced_<time>/`) so no earlier ExpansionHunter output is used for the new reads.

The two `tables/` folders are `train_model.sh`'s `SPIKED_TABLES_DIRS`. It adds them to the training people's tables
(`add_to_training_tables.py`) in a data folder of its own, which links to the base folder's per-source parquets, so it
is rebuilt by `train_model.sh` (or `add_to_training_tables.py`), never by `dataset.py` (which refuses it).

For the shipped samples, `spike` took about half an hour and `downsample` about 20 minutes locally, and the two
genotyping steps cost about $11 and $21 on Hail Batch. A genotyping step that failed part way is finished with
`ONLY_MISSING_OUTPUTS=1` (only the samples without an output are genotyped).

The steps read four committed tables: `disease_loci.tsv` and `disease_loci.catalog.json` (rebuilt with
`python3 simulated_expansions/build_disease_loci_table.py TRAINING_EH_JSON simulated_expansions` from one of the
training runs' ExpansionHunter JSONs and a fixed version of STRchive's locus data), `pathogenic_allele_structures.tsv`
(from the literature, below) and `repeat_read_errors.tsv` (measured by `calibrate_simulated_reads.py`, below). `spike`
also reads each person's sex and the loci where their reads contradict the truth from the training people's per-source
parquets (`TRAINING_TABLES_DIR`, `TABLES` below), and needs the bwa index at `BWA_INDEX_PREFIX` (default
`work/bwa_index/Homo_sapiens_assembly38.fasta.64`), which no step makes: download the six files
`gs://gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.fasta.64.{alt,amb,ann,bwt,pac,sa}` there first
(the script stops before running any step when `spike` is to run without them). The script's header lists every
setting.

`check-parity` must report no differences, and the script stops if it does: a difference means a spiked minicram would
not be genotyped the way the training reads were. The shipped samples' check covered 14 of the 45 people (the 6 pilot
people and 8 more), with no difference at any of the 60 loci; the script checks all 45, so a difference for one of the
other 31 is new information rather than a failure to reproduce, and leaving the step out (naming the other steps) runs
the steps the shipped samples went through. Genotyping runs on Hail Batch in the training runs' ExpansionHunter
image with their settings
(optimized-streaming mode, `--max-depth 150`, motif composition output for all loci, their UCSC hg38 reference), because
a macOS build of the same ExpansionHunter commit gives slightly different read depths, strand bias and indel sizes at the
loci the full genotyper calls, which is where every simulated expansion goes.

## What gets spiked

- **Loci** (`disease_loci.tsv`, `disease_loci.catalog.json`): the str_analysis package's known pathogenic loci, each
  matched to an overlapping training catalog locus with a motif of the same length, preferring the same motif (any
  rotation, either strand, N matching any base), then the largest overlap. The expanded
  motif is the pathogenic one where it differs from the reference (e.g. AAGGG at RFC1, GAAAT at DAB1, on the + strand).
- **Where** (`spikeable_loci`): for each person, the loci inside their DipCall high-confidence regions. Carriers only
  where both truth alleles are equal, so the allele the simulated one pairs with is known (at a male's chrX and chrY
  loci the simulated allele replaces the only one). Patients, whose two simulated alleles replace both of the person's,
  wherever the person's two alleles are both under 150 bp, whatever the genotype: then nearly all of the person's reads
  of them lie near the locus and are removed, while a longer allele's reads with a mate elsewhere would stay (at RFC1
  many people carry several hundred AAAAG or AAAGG copies). Neither is made where the person's own short reads
  contradict the truth (`TABLES` below).
- **Sizes** (`simulated_sizes_bp`): those of `SIZES_BP`, 150 bp to 24 kb, no longer than the largest pathogenic
  allele reported at the locus (`largest_reported_pathogenic_bp` in `disease_loci.tsv`, from STRchive), as no patient
  has one that long. Past 3 kb (4.5, 6, 9, 12, 18 and 24 kb) they reach the alleles of several kb that most patients
  carry at loci such as RFC1 (pathogenic from 2 kb), C9orf72, CNBP, DMPK, NOP56 and ATXN10. Every locus also gets the
  smallest pathogenic allele reported there (`smallest_reported_pathogenic_bp`) and the largest, where the sizes above
  don't already reach past it, and, where the smallest is under 150 bp, one halfway between it and the largest or
  150 bp, whichever is shorter; each rounded to whole motif copies (`reported_pathogenic_sizes_bp`). So patients' own
  reported sizes are simulated even where they fall between two of the sizes above (TNRC6A's and YEATS2's single
  reported sizes, 5.5 and 5 kb) or below all of them (the polyalanine loci, such as ARX, HOXA13, PABPN1 and PHOX2B,
  whose expansions stay under 150 bp, and the polyglutamine ones whose patients' alleles start under it, such as HTT's
  and AR's). That is about 2 more sizes per locus, 128 over the 60. Only sizes longer than the person's own allele are
  kept.
  VWA1, with no reported pathogenic sizes, gets none. STRchive counts a locus's copies over its own repeat region, which
  at PABPN1 and RUNX2 takes in 13 and 9 bp of alanine codons next to the training catalog's locus; the simulated alleles
  keep those bases in their flanks, so their alanine tracts are 3 to 4 codons longer than the patients'.
- **Carriers and patients** (`simulated_genotypes`): each size once as a carrier's simulated allele, which pairs with
  the person's own allele, and at the loci of autosomal recessive diseases (`AR` in `disease_loci.tsv`: CSTB, EIF4A3,
  FXN, GLS, PRDM12, RFC1, VWA1 and XYLT1) each size once more as the longer of a patient's two simulated alleles, which
  replace both of the person's. (An X-linked recessive disease's patients are mostly males, whose one allele the
  carriers at a male's chrX loci already simulate.) The patient's shorter allele is drawn from the locus's sizes no
  longer than the longer one, and no shorter than the smallest pathogenic allele where the longer one is pathogenic
  (so RFC1's patients with a 3 to 12 kb allele have a second one of 3 kb or more), with a seed fixed by the person, the
  locus and the size. RFC1's patients carry AAGGG on both alleles.
- **Structure** (`pathogenic_allele_structures.tsv`, `drawn_allele_structure`): at loci where patients' expanded
  alleles are not a pure tract of the pathogenic motif, each simulated allele takes one of the structures published
  for patients (blocks of other motifs or interruptions, laid out on the + strand), drawn by the share of patients'
  alleles that carry it (seeded by the allele), and otherwise a pure tract. The table lists each structure with the
  patient counts, assays and citations it rests on, mostly long-read studies of patients: GGA tracts in about 10% of
  FXN alleles, the 3' TCTG block of most CNBP alleles, the (AAAGG)(AAGGG)(AAAGG) allele at RFC1, ATTCC blocks at
  ATXN10, the TTTTA tracts with a TTTCA insertion that every patient of SAMD12 and MARCHF6 (familial adult myoclonic
  epilepsy) carries, CCG interruptions at the 3' end of a few DMPK alleles, and the 5' AGG that most FMR1 full
  mutations keep. At the loci whose patients carry an inserted motif between blocks of the reference one, no patient
  carries a pure tract of the inserted motif, so every simulated allele there is laid out as the published structure
  (DAB1, BEAN1, STARD7, TNRC6A, RAPGEF2 and YEATS2). Their reported pathogenic sizes count only the insertion, so the
  simulated size is the inserted block's and the reference blocks are added to it (`added_count` in the table,
  `structure_blocks`): a DAB1 allele of 150 to 375 bp of the inserted motif comes out about 0.8 to 1.2 kb in all, as
  patients' are. Block sizes are drawn from the published ranges, as copies (log-uniformly, with the published
  median where there is one; a `count` block takes less than half the allele, while an `added_count` block is added
  to it) or as shares of the allele; every block motif has the locus motif's length, so the allele's truth count is
  its length divided by that, the planned size plus any added blocks, and its truth purity counts each block's match
  to the reference motif. Each block's motif sets the loss of
  the fragments over it (`expanded_allele_reads.fragment_kept`), and the motif with the most copies sets the reads'
  errors inside the repeat.
  Loci left pure: those whose patients' alleles are pure in the literature (ATXN1, ATXN2, HTT), C9orf72 (the few
  interruptions long reads find are mostly put down to sequencing error) and NOP56 (only error-prone long reads of one
  family). Where only part of a structure is published, the table says so in its notes: no patient's reference-block
  length is reported at RAPGEF2 or YEATS2, so those blocks are the reference allele's own copies, a lower bound, and
  TNRC6A's 3' block is drawn from the range the same study measured in healthy people.
- **Samples**: a spiked sample holds one genotype at each of several loci (up to one per group below, so a dozen or
  more in a person's first samples and fewer in later ones), but never two loci within 4 kb of each other, whose
  simulated alleles would cover the same reads (in `disease_loci.tsv` only ARX's and HOXA13's loci are that close), nor
  two loci of one repeat (CAG and CTG are one repeat read from either strand; `motif_family`), because bwa puts one
  expansion's in-repeat reads on the other's repeat (every CAG expansion's on TCF4's CTG repeat), which no genome with
  one expansion has. Such loci take turns (`plan_spiked_samples`), so a person gets as many samples as the group of
  such loci with the most genotypes has.
- **Coverage**: each simulated allele takes the place of one of the person's alleles at the locus (one of two, or the
  only one at a male's chrX and chrY): it gets read pairs at that share of the rate the person's fragments start in the
  flanks, and the person's read pairs that lie within 2 kb of the locus are thinned by the simulated alleles' share
  (half for a carrier, all of them for a patient or a male's chrX and chrY). Pairs with a mate elsewhere or unmapped
  are kept: they are mostly reads of other parts of the genome, which simulating the locus's alleles does not bring
  back (`read_pairs_within_windows`). Both of a patient's alleles lose fragments over the repeat with the same
  exponent (below), as they come from one library.
- **Coverage over the repeat**: real reads cover long repeats less than their flanks, the more so the more extreme the
  motif's GC share (an AT-only or a GC-rich repeat keeps a small share of the fragments that lie in it). A simulated
  fragment with a share f of its bases in the repeat is kept with probability c^(f e): c per GC bin as measured
  (`repeat_read_errors.tsv`), e drawn per spiked locus (shared by a patient's two alleles) from
  `FRAGMENT_KEEP_EXPONENT_RANGE` (1 to 3, in the manifest). c is
  fitted at alleles mostly under 300 bp, where e = 1 matches ExpansionHunter's real and simulated rows; the calibration
  alleles over 300 bp lost more (e of about 2-3), and other library preps can lose more or fewer.
- **Truth**: the person's own allele and the simulated one, or a patient's two simulated ones, in the truth-table
  format the training tables are built from, so `dataset.extract_rows_and_join_truth` labels the rows as it does the
  real ones. The manifest (`work_spiked_samples/spiked/manifest.tsv`) lists each sample's loci with the number of simulated alleles
  (`simulated_alleles`), the longer one's size (`copies`, `size_bp`), structure (`simulated_allele_structure`, and its
  blocks as `MOTIF:copies` in + strand order in `simulated_allele_blocks`) and truth purity
  (`simulated_allele_purity`), and the same of a patient's shorter one (`shorter_simulated_allele_*`).
- **Lower coverage** (`downsample`, `downsample_spiked_samples`): the training people's genomes are all 38-50x, while
  clinical genomes are often 15-30x. For every spiked sample of another work folder, a copy downsampled to a coverage
  drawn from `DOWNSAMPLED_COVERAGE_RANGE` (15-30x; the person's coverage is twice their read length times their
  fragment start rate near the loci), keeping both mates of a pair or neither (`samtools view -s`), and a matched
  control: the person's own reads downsampled by the same share with the same seed, so it keeps the same pairs of
  theirs, labeled with their own truth at the same loci. Without the controls the model could learn that low coverage
  at these loci means an expansion. The copies and controls are named with `DOWNSAMPLED_COPY_NUMBER_OFFSET` and
  `MATCHED_CONTROL_NUMBER_OFFSET` added to the sample's number, and their manifest rows are the source sample's with
  `source_label`, `downsampled_coverage`, `downsample_fraction` and `matched_control` (a control has
  `simulated_alleles` 0). Only these new samples need genotyping; `add_to_training_tables.py` takes both work folders'
  tables. Read length and library prep are left as they are: trimming every simulated read to 100 bp would change
  them against real rows of the training people that are all 150 bp.

Spiked samples are named `<person>_spiked_<k>`. `train.py` never draws them as calibration people
(`dataset.is_spiked_sample`), refuses a table where a person it holds out has spiked samples (their reads would then
also train the model), and treats a spiked sample as its person (`dataset.person_of_sample`) where it holds out the
always-trained people's rows on one autosome and where it splits people for the out-of-fold gate. So no person's reads
sit on both sides of a split, and with the spiked people (`SPIKED_PEOPLE`), who are chosen outside the people it holds
out, calibration stays on real people's rows.

## How the simulated reads compare with real ones

`calibrate_simulated_reads.py` measures what the simulation leaves out, on alleles of known sequence rather than on the
confirmed expansions the simulated alleles are judged against (`compare_simulated_and_real_expansions.py`). At the
pilot people's own long alleles (at least 150 bp in the truth tables, 2-6 bp motifs, both alleles at least 0.9 pure;
about 270 loci per person), it replaces the person's read pairs that lie near the locus with simulated pairs of the
person's two alleles, whose sequences the truth tables hold, interruptions included. Then it genotypes the real and the
simulated version side by side and compares each allele's row between them. Before simulating, it measures the errors
of the real reads inside these repeats, by realigning them to the person's two alleles, into `repeat_read_errors.tsv`
(committed, and used by the spike step too). Two values can't be read off real reads alone: how many fragments over a
repeat are lost, and the rate of insertions and deletions to simulate (realigned, some simulated ones near a read's end
come out as mismatches or clipped bases). Run again after `simulate`, it compares the real and simulated reads and
corrects the values the simulation used by how far apart they came out, so `measure-read-errors` and `simulate`
alternate until they agree:

```bash
python3 simulated_expansions/calibrate_simulated_reads.py select-loci --training-tables-dir TABLES
python3 simulated_expansions/calibrate_simulated_reads.py make-minicrams --reference-fasta HG38_FA --billing-project PROJECT
python3 simulated_expansions/calibrate_simulated_reads.py measure-read-errors --reference-fasta HG38_FA --bwa BWA
python3 simulated_expansions/calibrate_simulated_reads.py simulate --reference-fasta HG38_FA --bwa BWA
python3 simulated_expansions/calibrate_simulated_reads.py measure-read-errors --reference-fasta HG38_FA --bwa BWA
# ... simulate and measure-read-errors again until the simulated reads' errors and coverage match the real ones
python3 simulated_expansions/calibrate_simulated_reads.py simulate --reference-fasta HG38_FA --bwa BWA
python3 simulated_expansions/calibrate_simulated_reads.py genotype --billing-project PROJECT --gcs-dir gs://...
python3 simulated_expansions/calibrate_simulated_reads.py compare
```

Everything goes to `work/calibration/`, except `measure-read-errors`'s table, which overwrites the committed
`repeat_read_errors.tsv` that the spike step reads (`--repeat-read-errors` writes it elsewhere), and each person's truth
download, in `work/truth/`.

## Files

| file | role |
|---|---|
| `make_spiked_samples.sh` | runs every step that makes the spiked samples the shipped model was trained on |
| `spike_in_expansions.py` | the steps themselves |
| `expanded_allele_reads.py` | the expanded allele's sequence and its simulated read pairs |
| `build_disease_loci_table.py` | the disease loci and their ExpansionHunter catalog (`disease_loci.tsv`, `disease_loci.catalog.json`) |
| `add_to_training_tables.py` | adds the spiked samples' tables to the training tables (`train_model.sh`'s first stage) |
| `compare_simulated_and_real_expansions.py` | how genotype quality models score the simulated alleles next to real confirmed expansions, and how far apart their model inputs are |
| `calibrate_simulated_reads.py` | how simulated reads of a long repeat differ from real ones, at the pilot people's own long alleles |
| `repeat_read_errors.tsv` | the errors of real reads inside long repeats, which the simulated reads get there |
| `disease_loci.tsv`, `disease_loci.catalog.json` | the loci that get simulated alleles, with their reported pathogenic sizes, and their ExpansionHunter catalog |
| `pathogenic_allele_structures.tsv` | published structures of patients' expanded alleles (interruptions, blocks of other motifs) that the simulated alleles take, with their sources |
| `work_spiked_samples/`, `work_lower_coverage_copies_of_spiked_samples/` | `make_spiked_samples.sh`'s minicrams, truth downloads, spiked reads, ExpansionHunter output and tables (not committed) |
| `work/` | the calibration's `work/calibration/` and truth downloads (`work/truth/`), and the default place of the bwa index (not committed) |

`TABLES` (`TRAINING_TABLES_DIR` in `make_spiked_samples.sh`, by default the shipped model's
`data.eh_e90307c_with_homopolymer_handoff_fix_and_new_truth/real_quick/`) is the folder of the training people's
per-source parquets (`<person>.parquet`, built from the training runs and marked by
`check_truth_genotypes_against_short_reads.py --mark`): their calls give each person's sex, and their marks leave out
the loci where the person's reads contradict the truth (the check judged only the loci of the table where a call
disagrees with the truth; loci missing from it, mostly hom-ref, were not checked). `HG38_FA` is a local, indexed copy of
the reference the training runs genotyped against (UCSC hg38, `TRAINING_REFERENCE_FASTA` in `spike_in_expansions.py`).
`spike` needs bwa (`--bwa`, a build for the machine's CPU) and samtools, and the bwa index of the hg38 the people's
reads were aligned to (`--bwa-index`, the prefix of `Homo_sapiens_assembly38.fasta.64.*` from the Broad's public hg38
references, above).
