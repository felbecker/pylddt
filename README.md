# pylddt

Python scripts to calculate DALI Z score and LDDT from a test Multiple Sequence Alignment (MSA) and PDB files with structures. 
Sequences are matched to structures by string matching, the sequence of a PDB file is taken from CA ATOM records.
Structures which do not match the MSA are ignored. Sequences in the MSA which do not match a structure are reported, then ignored.

## Usage 

Scripts are in `src/*.py`. Type `scriptname.py -h` to get usage information.

### Parallel scorers

`daliz.py`, `lddt_foldmason.py` and `lddt_mu.py` each have a numpy-vectorized
twin with the same CLI and the same output, plus a `--threads` option:
`daliz_par.py`, `lddt_foldmason_par.py`, `lddt_mu_par.py`. They share
`src/ms_np.py` and reuse `src/ms.py` unchanged for parsing and for matching
sequences to structures. On a 115-structure AFDB cluster with 16 threads:

| scorer | serial | 16 threads |
|---|---|---|
| `daliz` | 302 s | 3.0 s |
| `lddt_mu` | 155 s | 2.4 s |
| `lddt_foldmason` | 64 s | 2.4 s |

`lddt_ref_par.py` has no serial twin. It is a reference-style pair-wise LDDT:
the shorter sequence of each pair is the reference, neighbours come from the
reference structure only (an unaligned neighbour fails every threshold), and
the pair score is normalised by the reference length instead of the number of
aligned columns. `lddt_mu` therefore behaves like precision and `lddt_ref` like
recall; `tests/test_lddt_ref.py` pins down how the two differ.

`lddt_mu_par.py --lengthnorm` also prints `LDDT_rce_L_mean` (and, with
`--pairs`, a 4th column). This is LDDT-mu with the pair sum divided by the
length of the shorter sequence instead of the number of aligned columns, i.e.
coverage times LDDT-mu. Unlike `lddt_ref`, neighbours still come from the
aligned columns only, so correctly unaligned residues lower the coverage but
not the scores of their aligned neighbours. Without the flag the output is
unchanged.

`lddt_mu_par.py --coverage` also prints `COV_mean` (and, with `--pairs`, a
last column): the alignment coverage, i.e. the mean over pairs i < j of the
number of columns where both have residues divided by the length of the
shorter sequence. It ignores structure, so `LDDT_rce_L = coverage * LDDT_rce`
per pair.

`lddt_mu_par.py --min_plddt T` (default 70; 0 disables) also prints both
scores restricted to confident residues (`LDDT_rce_conf_mean`,
`LDDT_rce_L_conf_mean`). A column is scored, and counts as a neighbour, only if
the residues of both sequences have pLDDT >= T; the length-normalised version
divides by the smaller number of confident residues. pLDDT is read from the
B-factor column only for files whose header names AlphaFold; in other files,
e.g. experimental structures, every residue counts as confident. With
`--min_plddt 0` and without `--lengthnorm` and `--coverage`, the output is
identical to
`lddt_mu.py`; `tests/test_lddt_plddt.py` pins down the masked scores.

snakeMSA's `score_msa` rule reports `lddt_mu_par.py --symmetry both
--lengthnorm --coverage --min_plddt 70` (as `lddt_muscle`, `lddt_muscle_L`,
`lddt_muscle_conf`, `lddt_muscle_L_conf`, `coverage`) plus the default LDDT of
the FoldMason binary (`lddt_fm`); the other scorers here are kept for manual
use.

## Tests

`pytest` (needs `numpy`). The suite checks that each parallel scorer reproduces
its serial original byte for byte, across option sets, families and thread
counts:

<pre>
pytest tests -q                 # everything, about 90 s
pytest tests -q -m "not slow"   # skip the larger AFDB family, about 20 s
</pre>

Test structures live in `tests/data/&lt;collection&gt;/&lt;family&gt;/`, one `msa.afa`
plus one PDB per sequence. The `homstrad` families come from HOMSTRAD; the
`afdb` family is a 16-structure subsample of an AlphaFold DB cluster, stripped
to CA `ATOM` records (the scorers only ever read CA).

Note that the fixtures under `data/` and the expected output under `output/`
are not runnable: the PDB files their `.files` lists point at were never
committed.

## To reproduce FoldMason LDDT anomalies

<pre>

cd pylddt/bin
gunzip *.gz

cd pylddt/scorpion_toxin
./run_msa2lddt.bash
./run_reseek.bash

cd pylddt/scorpion_toxin3
./run_msa2lddt.bash
./run_reseek.bash
</pre>

## References

Mariani, Valerio, et al. "LDDT: a local superposition-free score for comparing protein structures and models using distance difference tests." <i>Bioinformatics</i> 29.21 (2013): 2722-2728.

Holm, Liisa, and Chris Sander. "Protein structure comparison by alignment of distance matrices." <i>Journal of molecular biology</i> 233.1 (1993): 123-138.

Gilchrist CL, Mirdita M, Steinegger M. Multiple Protein Structure Alignment at Scale with FoldMason. bioRxiv. 2024 Aug 1:2024-08.


![FoldMason LDDT anomalies](https://github.com/rcedgar/pylddt/raw/main/results/FoldMason_LDDT_anomaly_figure.png)

<b>Anomalous accounting for gaps with the FoldMason implemention of LDDT</b>.


(a) Three structures used to construct test
cases: (i) scorpion toxin 1sco chain A, (ii) 2iy5 A/a an all-alpha domain at the start of 2iy5 chain A,
and (iii) 4dl1 A/b, an all-beta domain at the start of 4ld1 chain A.


(b) Incorrect alignment of four copies
of 1sco A where the first amino acid is changed to ensure that sequences are distinct, e.g. 1sco/D has
first letter D.

(c) Correct alignment corresponding to (b). 

(d) 1sco A+2iy5 A/a aligned to
1sco A+4ld1 A/b; these structure pairs were manually concatenated to simulate multi-domain proteins
with one common domain. Domains 2iy5 A/a and 4ld1 A/b are not homologous and have no sequence
or structural similarity, therefore (d) is incorrect; these domains should be considered independent
inserts per alignment (e). 

Note that LDDT-fm (FoldMason) assigns 1.0, the highest possible score, to both (b) and (c);
also, LDDT-fm assigns a very low score to correct alignment (e) and a higher score to incorrect
alignment (d). LDDT-mu (Muscle) and Z assign higher scores to the correct alignments.
