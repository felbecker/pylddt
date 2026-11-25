#!/usr/bin/python3

# Parallel, numpy-vectorized equivalent of lddt_mu.py.
#
# Same CLI and same stdout format, plus --threads. The pair-wise LDDT of
# ms.MSTAScorer.lddt_score is transcribed to numpy and the pair loop of
# ms.MSTAScorer.calc_lddt_scores is spread over a process pool. PDB/MSA parsing
# and sequence-to-structure matching are reused unchanged from ms.py.
#
# Two things make this much cheaper than lddt_mu.py on deep MSAs:
#  - set_dist_mxs() is not called. lddt_mu.py calls it, but self.dist_mxs is only
#    ever read by calc_col_score(), which lddt_mu.py never runs, so it allocates
#    (and discards) one L*L Python-float matrix per structure. On a 903-sequence
#    cluster that is 19 s and 3 GB of resident memory for nothing.
#  - lddt_score() rebuilt both L*L distance matrices from scratch for every pair
#    in pure Python. Here each pair works directly on the coordinates of its own
#    common columns, so peak memory stays flat in the number of sequences.
#
# With --lengthnorm, a second, length-normalised score is reported as well:
#
#   LDDT_mu_ij   = 1/|A_ij| * sum_{c in A_ij} s(i, j, c)
#   LDDT_mu,L_ij = 1/L_s    * sum_{c in A_ij} s(i, j, c)  = cov_ij * LDDT_mu_ij
#
# A_ij = columns where both have residues, s(i, j, c) the usual per-column
# LDDT-mu score, L_s = min(L_i, L_j) the ungapped length of the shorter
# sequence and cov_ij = |A_ij|/L_s. Both come from the same column sum, so the
# second score is free. LDDT-mu behaves like precision (of what is aligned, how
# much is structurally consistent) and cannot detect under-alignment;
# LDDT-mu,L charges 1/L_s for every residue of the shorter sequence that is
# left unaligned.
#
# With --coverage, the alignment coverage is reported as well (COV_mean):
#
#   cov_ij = |A_ij| / L_s
#
# averaged over all pairs i < j, so LDDT_mu,L_ij = cov_ij * LDDT_mu_ij. It is
# the fraction of the shorter sequence that the MSA aligns to the other one,
# regardless of whether the aligned residues are structurally consistent.
#
# With --min_plddt t > 0 (default 70), the same scores are also reported on the
# residues whose structure is reliable (LDDT_rce_conf_mean, and with
# --lengthnorm LDDT_rce_L_conf_mean). A residue is confident if its pLDDT,
# which AlphaFold writes to the B-factor column, is >= t. Only files that
# declare themselves AlphaFold predictions (a TITLE/HEADER/REMARK line naming
# ALPHAFOLD) are read that way; in any other file, e.g. an experimental
# structure, every residue counts as confident, since its B-factor is not a
# pLDDT. Per pair:
#
#   K_ij              columns of A_ij where the residues of both i and j are
#                     confident; only these are scored, and only these count as
#                     neighbours
#   LDDT_mu_conf_ij   = 1/|K_ij| * sum_{c in K_ij} s_K(i, j, c)
#   LDDT_mu,L_conf_ij = 1/C_s * sum_{c in K_ij} s_K(i, j, c)
#
# s_K is the per-column score with the neighbourhood restricted to K_ij and
# C_s = min(C_i, C_j) the smaller number of confident residues. C_s does not
# depend on the alignment, so leaving a confident residue unaligned, or aligning
# it to a low-confidence partner, costs 1/C_s; unaligned low-confidence residues
# cost nothing. A pair in which one sequence has no confident residue is
# excluded from the _conf means (shown as nan with --pairs; a mean over no
# pairs is printed as -1).
#
# With --min_plddt 0 and without --lengthnorm and --coverage, stdout is
# identical to
# lddt_mu.py.

import sys
import argparse
import numpy as np
from ms import *
import ms_np

Usage = \
(
'Calculate LDDT for a sequence alignment by comparison with '
'reference structures as average of pair-wise LDDT (parallel).'
)

AP = argparse.ArgumentParser(description = Usage)
AP.add_argument("--msa", required=True, help="Multiple Sequence Alignment (FASTA format)")
AP.add_argument("--pdbfiles", required=True, help="Text file with one PDB pathname per line")
AP.add_argument("--radius", required=False, type=float, default=15.0, help="LDDT inclusion radius (default 15)")
AP.add_argument("--dists", required=False, default="0.5,1,2,4", help="LDDT distance thresholds, comma-separated (default 0.5,1,2,4)")
AP.add_argument("--symmetry", required=False, choices=[ "first", "both", "either" ], default="first", help="Set R0 according to both / either / first (default first)")
AP.add_argument("--pairs", action="store_true", help="Report pair-wise LDDTs (default don't show)")
AP.add_argument("--lengthnorm", action="store_true", help="Also report LDDT normalised by the length of the shorter sequence (LDDT_rce_L_mean; with --pairs as a 4th column)")
AP.add_argument("--coverage", action="store_true", help="Also report the alignment coverage, the mean over pairs of aligned columns / length of the shorter sequence (COV_mean; with --pairs as the last column)")
AP.add_argument("--min_plddt", required=False, type=float, default=70.0, help="Also report LDDT on residues with pLDDT >= this in AlphaFold models (LDDT_rce_conf_mean, LDDT_rce_L_conf_mean; with --pairs as extra columns); 0 disables (default 70)")
AP.add_argument("--threads", required=False, type=int, default=1, help="Number of worker processes (default 1)")

# Set by main() before the pool is forked, so workers inherit them copy-on-write.
COORDS = None      # list of (L, 3) float64 arrays, one per matched sequence
COL2POS = None     # (nr_seqs, nr_cols) int32, -1 where the row has a gap
R0 = None
THRESHOLDS = None
SYMMETRY = None
CONF = None        # list of (L,) bool arrays, True where the residue is confident

def is_alphafold(fn):
	"""True if the PDB header names AlphaFold, i.e. B-factors are pLDDTs."""
	for line in open(fn):
		if line.startswith(("ATOM", "HETATM", "MODEL")):
			return False
		if line.startswith(("TITLE", "HEADER", "REMARK")) and "ALPHAFOLD" in line.upper():
			return True
	return False

def read_bfactors(scorer, fn):
	"""B-factors of the residues ms.MSTAScorer.read_pdb keeps, in its order.

	The line filters are those of read_pdb, so the result lines up with the
	coordinates and the sequence.
	"""
	bs = []
	for line in open(fn):
		if line.startswith("ENDMDL") or line.startswith("TER "):
			break
		if line[12:16].strip() != "CA":
			continue
		if line[16] != ' ' and line[16] != 'A':
			continue
		if line[17:20] not in scorer.three2one:
			continue
		try:
			float(line[30:38]); float(line[38:46]); float(line[46:54])
		except ValueError:
			continue
		try:
			bs.append(float(line[60:66]))
		except ValueError:
			bs.append(100.0)
	return np.array(bs, dtype=np.float64)

def build_conf(scorer, min_plddt):
	"""Per matched sequence, which residues are confident; and a stderr note."""
	conf = []
	nr_af = 0
	nr_res_af = 0
	nr_low = 0
	for i in range(len(scorer.msa_idxs)):
		fn = scorer.pdb_fns[scorer.pdb_idxs[i]]
		L = len(scorer.fn2data[fn][0])
		if min_plddt > 0 and is_alphafold(fn):
			b = read_bfactors(scorer, fn)
			assert b.size == L, fn
			c = b >= min_plddt
			nr_af += 1
			nr_res_af += L
			nr_low += int((~c).sum())
		else:
			c = np.ones(L, dtype=bool)
		conf.append(c)
	note = ("pLDDT mask (>= %g): %d/%d structures are AlphaFold models, "
			"%d/%d of their residues below the cutoff\n"
			% (min_plddt, nr_af, len(conf), nr_low, nr_res_af))
	return conf, note

def col_score_sum(sel, diff):
	"""Sum of the per-column LDDT-mu scores for a neighbour mask sel."""
	# one "considered" is counted per threshold, not per column pair
	nr_considered = sel.sum(1)*len(THRESHOLDS)
	nr_preserved = np.zeros(sel.shape[0], dtype=np.int64)
	for t in THRESHOLDS:
		nr_preserved += (sel & (diff <= t)).sum(1)
	# columns with no neighbours score 0 but still count in the denominator
	return np.where(nr_considered > 0,
					nr_preserved/np.maximum(nr_considered, 1), 0.0).sum()

def lddt_pair(p1, p2, C1, C2, K1, K2):
	"""(LDDT, length-normalised LDDT, both on confident residues, coverage) of
	one sequence pair.

	The first is equivalent to ms.MSTAScorer.lddt_score. The second divides the
	same column sum by the length of the shorter sequence instead of the number
	of common columns. len(C) is the ungapped row length: match_seqs_to_pdbs
	requires the PDB sequence to equal the ungapped MSA row. The last two are
	nan if one sequence has no confident residue (K1/K2 are the masks). The
	coverage is the number of common columns over the shorter length.
	"""
	nr_conf = min(K1.sum(), K2.sum())
	nan = float("nan")
	cols, d1, d2 = ms_np.common_dists(p1, p2, C1, C2)
	n = cols.size
	L_s = min(len(C1), len(C2))
	if n == 0:
		return (0.0, 0.0) + ((0.0, 0.0) if nr_conf else (nan, nan)) + (0.0,)
	cov = n/L_s
	sel = ms_np.symmetry_mask(d1, d2, R0, SYMMETRY)
	diff = np.abs(d1 - d2)
	total = col_score_sum(sel, diff)
	mu, mu_L = total/n, total/L_s
	if nr_conf == 0:
		return mu, mu_L, nan, nan, cov
	keep = K1[p1[cols]] & K2[p2[cols]]
	k = int(keep.sum())
	if k == n:
		total_K = total
	elif k == 0:
		return mu, mu_L, 0.0, 0.0, cov
	else:
		total_K = col_score_sum(sel[np.ix_(keep, keep)], diff[np.ix_(keep, keep)])
	return mu, mu_L, total_K/k, total_K/nr_conf, cov

def score_row(i):
	"""lddt_pair of sequence i against every j > i, one row per j."""
	p1 = COL2POS[i]
	C1 = COORDS[i]
	scores = np.empty((len(COORDS) - i - 1, 5), dtype=np.float64)
	for k, j in enumerate(range(i + 1, len(COORDS))):
		scores[k] = lddt_pair(p1, COL2POS[j], C1, COORDS[j], CONF[i], CONF[j])
	return i, scores

def main():
	global COORDS, COL2POS, R0, THRESHOLDS, SYMMETRY, CONF

	Args = AP.parse_args()
	msa_fn = Args.msa
	paths_fn = Args.pdbfiles

	scorer = create_scorer(Args)

	scorer.read_pdbs(paths_fn)
	scorer.read_msa(msa_fn)
	scorer.match_seqs_to_pdbs()

	nr_matched = len(scorer.msa_idxs)
	if nr_matched != scorer.nr_seqs:
		sys.stderr.write("\n===ERROR=== %d / %d MSA sequences matched to structures\n" %
						 (scorer.nr_matched, scorer.nr_seqs))
		sys.exit(1)

	R0 = scorer.R0
	SYMMETRY = scorer.symmetry
	THRESHOLDS = scorer.thresholds

	# before build_arrays, which drops scorer.fn2data
	CONF, note = build_conf(scorer, Args.min_plddt)
	if Args.min_plddt > 0:
		sys.stderr.write(note)
	COORDS, COL2POS = ms_np.build_arrays(scorer)

	rows = ms_np.run_rows(score_row, nr_matched, Args.threads)

	conf = Args.min_plddt > 0
	# which of the five lddt_pair values are reported, in output order
	shown = [ 0 ] + ([ 1 ] if Args.lengthnorm else []) \
		+ (([ 2 ] + ([ 3 ] if Args.lengthnorm else [])) if conf else []) \
		+ ([ 4 ] if Args.coverage else [])
	names = [ "LDDT_rce_mean", "LDDT_rce_L_mean",
			  "LDDT_rce_conf_mean", "LDDT_rce_L_conf_mean", "COV_mean" ]

	totals = [ 0.0 ]*5
	nr_pairs = 0
	nr_pairs_conf = 0
	for i in range(nr_matched - 1):
		labeli = scorer.labels[scorer.msa_idxs[i]]
		for k, j in enumerate(range(i + 1, nr_matched)):
			scores = rows[i][k]
			totals[0] += scores[0]
			totals[1] += scores[1]
			totals[4] += scores[4]
			nr_pairs += 1
			if not np.isnan(scores[2]):
				totals[2] += scores[2]
				totals[3] += scores[3]
				nr_pairs_conf += 1
			if Args.pairs:
				labelj = scorer.labels[scorer.msa_idxs[j]]
				print("\t".join([ labeli, labelj ] + [ "%.4f" % scores[m] for m in shown ]))

	for m in shown:
		n = nr_pairs_conf if m in (2, 3) else nr_pairs
		mean = totals[m]/n if n else -1.0
		print("%s\t%.4f\t%s" % (names[m], mean, msa_fn))

if __name__ == "__main__":
	main()
