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
AP.add_argument("--threads", required=False, type=int, default=1, help="Number of worker processes (default 1)")

# Set by main() before the pool is forked, so workers inherit them copy-on-write.
COORDS = None      # list of (L, 3) float64 arrays, one per matched sequence
COL2POS = None     # (nr_seqs, nr_cols) int32, -1 where the row has a gap
R0 = None
THRESHOLDS = None
SYMMETRY = None

def lddt_pair(p1, p2, C1, C2):
	"""LDDT of one sequence pair. Equivalent to ms.MSTAScorer.lddt_score."""
	cols, d1, d2 = ms_np.common_dists(p1, p2, C1, C2)
	n = cols.size
	if n == 0:
		return 0.0
	sel = ms_np.symmetry_mask(d1, d2, R0, SYMMETRY)
	diff = np.abs(d1 - d2)
	# one "considered" is counted per threshold, not per column pair
	nr_considered = sel.sum(1)*len(THRESHOLDS)
	nr_preserved = np.zeros(n, dtype=np.int64)
	for t in THRESHOLDS:
		nr_preserved += (sel & (diff <= t)).sum(1)
	col_scores = np.where(nr_considered > 0,
						  nr_preserved/np.maximum(nr_considered, 1), 0.0)
	# columns with no neighbours score 0 but still count in the denominator
	return col_scores.sum()/n

def score_row(i):
	"""LDDT of sequence i against every j > i."""
	p1 = COL2POS[i]
	C1 = COORDS[i]
	scores = np.empty(len(COORDS) - i - 1, dtype=np.float64)
	for k, j in enumerate(range(i + 1, len(COORDS))):
		scores[k] = lddt_pair(p1, COL2POS[j], C1, COORDS[j])
	return i, scores

def main():
	global COORDS, COL2POS, R0, THRESHOLDS, SYMMETRY

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

	COORDS, COL2POS = ms_np.build_arrays(scorer)

	rows = ms_np.run_rows(score_row, nr_matched, Args.threads)

	total_LDDT = 0.0
	nr_pairs = 0
	for i in range(nr_matched - 1):
		labeli = scorer.labels[scorer.msa_idxs[i]]
		for k, j in enumerate(range(i + 1, nr_matched)):
			score = rows[i][k]
			total_LDDT += score
			nr_pairs += 1
			if Args.pairs:
				labelj = scorer.labels[scorer.msa_idxs[j]]
				print("%s\t%s\t%.4f" % (labeli, labelj, score))

	print("LDDT_rce_mean\t%.4f\t%s" % (total_LDDT/nr_pairs, msa_fn))

if __name__ == "__main__":
	main()
