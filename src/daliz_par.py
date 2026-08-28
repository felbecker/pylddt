#!/usr/bin/python3

# Parallel, numpy-vectorized equivalent of daliz.py.
#
# Same CLI and same stdout format, plus --threads. The pair-wise DALI score of
# ms.MSTAScorer.dali_score is transcribed to numpy and the pair loop of
# ms.MSTAScorer.calc_dali_scores is spread over a process pool; the Z scores and
# the means are then computed in the parent exactly as daliz.py does them.
#
# Unlike the two LDDT scorers, daliz tolerates MSA rows with no matching
# structure, so everything here indexes through msa_idxs/pdb_idxs rather than
# assuming msa_idxs[i] == i.

import sys
import argparse
import numpy as np
from ms import *
import ms_np

Usage = \
(
"Calculate DALI Z score from a test Multiple Sequence Alignment (MSA) and PDB files with structures. "
"Reports Z for each pair of sequences, and the mean Z over all pairs. "
"Sequences are matched to structures by string matching, the sequence of a PDB file is taken from CA ATOM records. "
"Structures which do not match the MSA are ignored. "
"(parallel) "
)

AP = argparse.ArgumentParser(description = Usage)
AP.add_argument("--msa", required=True, help="Multiple Sequence Alignment (FASTA format)")
AP.add_argument("--pdbfiles", required=True, help="Text file with one PDB pathname per line")
AP.add_argument("--radius", required=False, type=float, default=None, help="Inclusion radius as for LDDT (default none)")
AP.add_argument("--symmetry", required=False, choices=[ "first", "both", "either" ], default="first",
                help="Which distance(s) considered for radius (default first)")
AP.add_argument("--horizon", required=False, type=float, default=20.0, help="DALI horizon (default 20)")
AP.add_argument("--diagwt", required=False, type=float, default=0.2, help="DALI diagonal weight (default 0.2)")
AP.add_argument("--threads", required=False, type=int, default=1, help="Number of worker processes (default 1)")

# Set by main() before the pool is forked, so workers inherit them copy-on-write.
COORDS = None      # list of (L, 3) float64 arrays, one per matched sequence
COL2POS = None     # (nr_matched, nr_cols) int32, -1 where the row has a gap
R0 = None
SYMMETRY = None
D = None           # DALI horizon
D0 = None          # DALI diagonal weight

def dali_pair(p1, p2, C1, C2):
	"""DALI score of one sequence pair. Equivalent to ms.MSTAScorer.dali_score.

	The serial version sums over every ordered (coli, colj) with colj != coli,
	so each unordered column pair is counted twice; the full off-diagonal matrix
	below does the same.
	"""
	cols, d1, d2 = ms_np.common_dists(p1, p2, C1, C2)
	n = cols.size
	if n == 0:
		return 0.0
	sel = ms_np.symmetry_mask(d1, d2, R0, SYMMETRY)
	diff = np.abs(d1 - d2)
	mean = (d1 + d2)/2
	# ms.DALI_weight: x = 1/D^2, w = exp(-x*y*y)
	x = 1.0/(D*D)
	w = np.exp(-x*mean*mean)
	# ms.DALI_dpscorefun divides by mean only on the mean != 0 branch
	ratio = np.divide(diff, mean, out=np.zeros_like(diff), where=(mean != 0))
	term = np.where(mean == 0, w*D0, w*(D0 - ratio))
	term = np.where(mean > 100, 0.0, term)
	# row-wise first, so the reduction stays as close to the serial coli/colj
	# order as vectorizing allows; any residual difference is far below the
	# %.0f / %.3f this is printed at
	score = np.where(sel, term, 0.0).sum(1).sum()
	# ms.dali_score adds the diagonal term after the loop
	return float(score) + n*D0

def score_row(i):
	"""DALI score of sequence i against every j > i."""
	p1 = COL2POS[i]
	C1 = COORDS[i]
	scores = np.empty(len(COORDS) - i - 1, dtype=np.float64)
	for k, j in enumerate(range(i + 1, len(COORDS))):
		scores[k] = dali_pair(p1, COL2POS[j], C1, COORDS[j])
	return i, scores

def main():
	global COORDS, COL2POS, R0, SYMMETRY, D, D0

	Args = AP.parse_args()

	scorer = create_scorer(Args)

	scorer.read_pdbs(Args.pdbfiles)
	scorer.read_msa(Args.msa)
	scorer.match_seqs_to_pdbs()
	for label in scorer.not_matched_labels:
		print("MSA_label_not_matched=%s" % label)

	R0 = scorer.R0
	SYMMETRY = scorer.symmetry
	D = scorer.D
	D0 = scorer.d0

	nr_matched = scorer.nr_matched
	COORDS, COL2POS = ms_np.build_arrays(scorer)

	rows = ms_np.run_rows(score_row, nr_matched, Args.threads)

	total_score = 0.0
	total_Z = 0.0
	nr_pairs = 0
	for i in range(nr_matched - 1):
		labeli = scorer.labels[scorer.msa_idxs[i]]
		# read_pdb appends one coordinate per residue, so this is len(pdb_seq)
		Li = len(COORDS[i])
		for k, j in enumerate(range(i + 1, nr_matched)):
			labelj = scorer.labels[scorer.msa_idxs[j]]
			Lj = len(COORDS[j])
			score = rows[i][k]
			Z = scorer.dali_Z_from_score_and_lengths(score, Li, Lj)
			total_score += score
			total_Z += Z
			nr_pairs += 1
			s = "label1=" + labeli
			s += "\tlabel2=" + labelj
			s += "\tscore=%.0f" % score
			s += "\tZ=%.3f" % Z
			print(s)

	print("MSA=%s\tmean_DALI_score=%.0f\tmean_DALI_Z=%.3f" %
	      (Args.msa, total_score/nr_pairs, total_Z/nr_pairs))

if __name__ == "__main__":
	main()
