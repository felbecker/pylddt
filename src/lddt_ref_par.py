#!/usr/bin/python3

# Reference-style pair-wise LDDT, parallel and numpy-vectorized.
#
# Same CLI shape and stdout format as lddt_mu_par.py (minus --symmetry). The
# difference is the score: for every pair (i, j) the shorter sequence s is the
# reference (ties broken by label), o is the other one.
#
#   N_s(c)  = { c' != c : s has a residue at c', d_s(c, c') <= R0 }
#             neighbours come from the reference structure only
#
#   l(c)    = 1/(|N_s(c)| |T|) * sum_{c' in N_s(c)} sum_{t in T}
#               [ o has a residue at c'  and  |d_s(c, c') - d_o(c, c')| <= t ]
#             for a column c where both have residues; a neighbour that o does
#             not align fails every threshold
#
#   LDDT_ij = 1/L_s * sum_{c in A_ij} l(c)
#             A_ij = columns where both have residues; L_s = ungapped length of
#             s, so unaligned residues of s contribute 0
#
#   LDDT    = mean of LDDT_ij over all pairs i < j
#
# lddt_mu_par.py normalises by |A_ij| and takes neighbours from the common
# columns only, so it behaves like precision (of what was aligned, how much is
# structurally consistent). This score behaves like recall (how much of the
# reference is reproduced). Note that it cannot tell a residue that is correctly
# left unaligned from a missed one.
#
# PDB/MSA parsing and sequence-to-structure matching come from ms.py, the array
# setup and the process pool from ms_np.py.

import sys
import argparse
import numpy as np
from ms import *
import ms_np

Usage = \
(
'Calculate reference-style LDDT for a sequence alignment by comparison with '
'reference structures as average of pair-wise LDDT, using the shorter sequence '
'of each pair as reference (parallel).'
)

AP = argparse.ArgumentParser(description = Usage)
AP.add_argument("--msa", required=True, help="Multiple Sequence Alignment (FASTA format)")
AP.add_argument("--pdbfiles", required=True, help="Text file with one PDB pathname per line")
AP.add_argument("--radius", required=False, type=float, default=15.0, help="LDDT inclusion radius (default 15)")
AP.add_argument("--dists", required=False, default="0.5,1,2,4", help="LDDT distance thresholds, comma-separated (default 0.5,1,2,4)")
AP.add_argument("--pairs", action="store_true", help="Report pair-wise LDDTs (default don't show)")
AP.add_argument("--threads", required=False, type=int, default=1, help="Number of worker processes (default 1)")

# Set by main() before the pool is forked, so workers inherit them copy-on-write.
COORDS = None      # list of (L, 3) float64 arrays, one per matched sequence
COL2POS = None     # (nr_seqs, nr_cols) int32, -1 where the row has a gap
LABELS = None      # labels in matched order, for the reference tie break
R0 = None
THRESHOLDS = None

def pairwise_dists(A, B):
	return np.sqrt(((A[:, None, :] - B[None, :, :])**2).sum(-1))

def lddt_ref_pair(i, j):
	"""Reference-style LDDT of sequences i and j."""
	Li = len(COORDS[i])
	Lj = len(COORDS[j])
	s, o = (i, j) if (Li, LABELS[i]) <= (Lj, LABELS[j]) else (j, i)
	Cs = COORDS[s]
	Co = COORDS[o]
	L_s = len(Cs)
	if L_s == 0:
		return 0.0

	# Reference residues are 0 .. L_s-1 in column order. q[k] is the residue of
	# o aligned to reference residue k, -1 if none.
	ps = COL2POS[s]
	q = COL2POS[o][ps >= 0]
	r = np.flatnonzero(q >= 0)
	n = r.size
	if n == 0:
		return 0.0

	# Neighbours of the aligned reference residues among all reference residues.
	Ds = pairwise_dists(Cs[r], Cs)
	nbr = Ds <= R0
	nbr[np.arange(n), r] = False
	nr_nbr = nbr.sum(1)

	# Only neighbours that o aligns can be preserved; the others are failures
	# by never being counted in the numerator.
	Do = pairwise_dists(Co[q[r]], Co[q[r]])
	diff = np.abs(Ds[:, r] - Do)
	nbr_al = nbr[:, r]
	nr_preserved = np.zeros(n, dtype=np.int64)
	for t in THRESHOLDS:
		# <= as in lddt_mu (ms.MSTAScorer.lddt_score), so the two scores agree
		nr_preserved += (nbr_al & (diff <= t)).sum(1)

	nr_considered = nr_nbr*len(THRESHOLDS)
	res_scores = np.where(nr_considered > 0,
						  nr_preserved/np.maximum(nr_considered, 1), 0.0)
	return res_scores.sum()/L_s

def score_row(i):
	"""LDDT of sequence i against every j > i."""
	scores = np.empty(len(COORDS) - i - 1, dtype=np.float64)
	for k, j in enumerate(range(i + 1, len(COORDS))):
		scores[k] = lddt_ref_pair(i, j)
	return i, scores

def main():
	global COORDS, COL2POS, LABELS, R0, THRESHOLDS

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
	THRESHOLDS = scorer.thresholds
	LABELS = [ scorer.labels[scorer.msa_idxs[i]] for i in range(nr_matched) ]

	COORDS, COL2POS = ms_np.build_arrays(scorer)

	rows = ms_np.run_rows(score_row, nr_matched, Args.threads)

	total_LDDT = 0.0
	nr_pairs = 0
	for i in range(nr_matched - 1):
		for k, j in enumerate(range(i + 1, nr_matched)):
			score = rows[i][k]
			total_LDDT += score
			nr_pairs += 1
			if Args.pairs:
				print("%s\t%s\t%.4f" % (LABELS[i], LABELS[j], score))

	print("LDDT_ref_mean\t%.4f\t%s" % (total_LDDT/nr_pairs, msa_fn))

if __name__ == "__main__":
	main()
