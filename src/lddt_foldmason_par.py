#!/usr/bin/python3

# Parallel, numpy-vectorized equivalent of lddt_foldmason.py.
#
# Same CLI and same stdout format, plus --threads. The column scoring of
# ms.MSTAScorer.calc_col_score / calc_mean_col_score is transcribed to numpy and
# turned inside out: the serial version loops columns outermost and rescans
# every sequence pair for each one, this loops sequence pairs outermost and
# scatters each pair's per-column contribution into shared accumulators. PDB/MSA
# parsing and sequence-to-structure matching are reused unchanged from ms.py.
#
# set_dist_mxs() is not called. lddt_foldmason.py calls it to fill the
# dist_mxs that calc_col_score reads, which costs one L*L Python-float matrix
# per structure; here each pair works directly on the coordinates of its own
# common columns, so peak memory stays flat in the number of sequences.
#
# Note that this scoring path deliberately reproduces two quirks of
# ms.calc_col_score that ms.lddt_score (the --mu scorer) does not share:
# the thresholds are hardcoded 0.5/1/2/4 and --dists is ignored, and the
# comparison is a strict < rather than <=.

import sys
import argparse
import numpy as np
from ms import *
import ms_np

Usage = \
(
'Calculate LDDT for a sequence alignment by comparison with '
'reference structures, using the Foldmason method (parallel).'
)

AP = argparse.ArgumentParser(description = Usage)
AP.add_argument("--msa", required=True, help="Multiple Sequence Alignment (FASTA format)")
AP.add_argument("--pdbfiles", required=True, help="Text file with one PDB pathname per line")
AP.add_argument("--radius", required=False, type=float, default=15.0, help="LDDT inclusion radius (default 15)")
AP.add_argument("--dists", required=False, default="0.5,1,2,4", help="LDDT distance thresholds, comma-separated (default 0.5,1,2,4)")
AP.add_argument("--symmetry", required=False, choices=[ "first", "both", "either" ], default="first", help="Set R0 according to both / either / first (default first)")
AP.add_argument("--cols", action="store_true", help="Report column scores")
AP.add_argument("--threads", required=False, type=int, default=1, help="Number of worker processes (default 1)")

# ms.calc_col_score hardcodes these and ignores --dists; so does this.
FM_THRESHOLDS = ( 0.5, 1.0, 2.0, 4.0 )

# Set by main() before the pool is forked, so workers inherit them copy-on-write.
COORDS = None      # list of (L, 3) float64 arrays, one per matched sequence
COL2POS = None     # (nr_seqs, nr_cols) int32, -1 where the row has a gap
NR_COLS = None
R0 = None
SYMMETRY = None

def foldmason_pair(p1, p2, C1, C2):
	"""(cols, scores): this pair's contribution to each column it spans.

	scores[k] is the `pair_score` that ms.calc_col_score would add to column
	cols[k]'s running total. A column pair with no included neighbours scores 0
	but still counts as a sequence pair for that column, which is the gap
	accounting the FoldMason method is criticized for.
	"""
	cols, d1, d2 = ms_np.common_dists(p1, p2, C1, C2)
	if cols.size == 0:
		return cols, np.zeros(0, dtype=np.float64)
	sel = ms_np.symmetry_mask(d1, d2, R0, SYMMETRY)
	diff = np.abs(d1 - d2)
	counts = sel.sum(1)
	preserved = np.zeros(cols.size, dtype=np.int64)
	for t in FM_THRESHOLDS:
		preserved += (sel & (diff < t)).sum(1) # strict <, as in ms.calc_col_score
	# the serial version sums indicator/4 per neighbour then divides by the
	# neighbour count; preserved/(4*counts) is the same correctly-rounded value
	den = np.maximum(counts, 1)*len(FM_THRESHOLDS)
	return cols, np.where(counts > 0, preserved/den, 0.0)

def score_row(i):
	"""Column sums and counts for sequence i against every j > i."""
	p1 = COL2POS[i]
	C1 = COORDS[i]
	col_sums = np.zeros(NR_COLS, dtype=np.float64)
	col_counts = np.zeros(NR_COLS, dtype=np.int64)
	for j in range(i + 1, len(COORDS)):
		cols, scores = foldmason_pair(p1, COL2POS[j], C1, COORDS[j])
		col_sums[cols] += scores
		col_counts[cols] += 1
	return i, (col_sums, col_counts)

def main():
	global COORDS, COL2POS, NR_COLS, R0, SYMMETRY

	Args = AP.parse_args()
	msa_fn = Args.msa
	paths_fn = Args.pdbfiles

	scorer = create_scorer(Args)

	scorer.read_pdbs(paths_fn)
	scorer.read_msa(msa_fn)
	scorer.match_seqs_to_pdbs()

	nr_matched = len(scorer.msa_idxs)
	if nr_matched != scorer.nr_seqs:
		sys.stderr.write("\n===ERROR===\n%d / %d MSA sequences matched to structures\n" %
						 (scorer.nr_matched, scorer.nr_seqs))
		sys.stderr.write("first not matched >%s\n" %
						 (scorer.not_matched_labels[0]))
		sys.exit(1)

	NR_COLS = scorer.nr_cols
	R0 = scorer.R0
	SYMMETRY = scorer.symmetry

	COORDS, COL2POS = ms_np.build_arrays(scorer)

	rows = ms_np.run_rows(score_row, nr_matched, Args.threads)

	# adding the rows back in ascending i, having summed j ascending inside each
	# worker, reproduces the serial accumulation order term for term
	col_sums = np.zeros(NR_COLS, dtype=np.float64)
	col_counts = np.zeros(NR_COLS, dtype=np.int64)
	for i in range(nr_matched - 1):
		col_sums += rows[i][0]
		col_counts += rows[i][1]
	col_scores = np.where(col_counts > 0, col_sums/np.maximum(col_counts, 1), 0.0)

	# sequential, like ms.calc_mean_col_score; np.sum would reassociate
	total_col_scores = 0
	for col_score in col_scores:
		total_col_scores += col_score
	LDDT = total_col_scores/NR_COLS

	if Args.cols:
		print("col_nr\tcol_str\tscore")
		for col_nr in range(NR_COLS):
			col_str = scorer.msa_col(col_nr)
			print("%d\t%s\t%.4f" % (col_nr, col_str, col_scores[col_nr]))

	print("LDDT_foldmason\t%.4f\t%s\n" % (LDDT, msa_fn))

if __name__ == "__main__":
	main()
