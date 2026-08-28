#!/usr/bin/python3

# Shared numpy / multiprocessing machinery for the parallel scorers
# (daliz_par.py, lddt_foldmason_par.py, lddt_mu_par.py).
#
# Each of those transcribes one ms.MSTAScorer scoring loop to numpy and spreads
# the sequence-pair loop over a process pool. PDB/MSA parsing and
# sequence-to-structure matching are reused unchanged from ms.py; only the
# O(pairs * cols^2) inner loops live here.
#
# float64 throughout: float32 rounding can flip a distance-difference
# comparison sitting right on a threshold.

import multiprocessing
import numpy as np

def build_arrays(scorer):
	"""(coords, col2pos) for the matched sequences, in matched order.

	coords[i] is an (L, 3) float64 array of CA coordinates, col2pos is an
	(nr_matched, nr_cols) int32 map with -1 where the row has a gap.

	This indexes through msa_idxs/pdb_idxs instead of assuming that every
	sequence matched, so it is also correct for daliz, which tolerates
	unmatched rows. ms.set_col2pos_vec() cannot be used there: it loops over
	nr_seqs but indexes msa_idxs, which is only nr_matched long.

	scorer.fn2data is dropped afterwards to free the per-structure Python float
	lists; len(coords[i]) is the full PDB sequence length if a caller needs it.
	"""
	nr_matched = len(scorer.msa_idxs)
	coords = []
	rows = []
	for i in range(nr_matched):
		pdb_fn = scorer.pdb_fns[scorer.pdb_idxs[i]]
		_, xs, ys, zs = scorer.fn2data[pdb_fn]
		coords.append(np.array([ xs, ys, zs ], dtype=np.float64).T.copy())
		row = scorer.msa[scorer.labels[scorer.msa_idxs[i]]]
		rows.append([ -1 if pos is None else pos for pos in scorer.calc_col2pos(row) ])
	col2pos = np.array(rows, dtype=np.int32)
	scorer.fn2data = None
	return coords, col2pos

def common_dists(p1, p2, C1, C2):
	"""(cols, d1, d2) for the columns where neither row has a gap.

	cols indexes the MSA, d1/d2 are the (n, n) intra-structure CA distance
	matrices restricted to those columns. Working on the pair's own common
	columns rather than on full L1*L1 / L2*L2 matrices is what keeps peak
	memory flat in the number of sequences.
	"""
	cols = np.flatnonzero((p1 >= 0) & (p2 >= 0))
	A = C1[p1[cols]]
	B = C2[p2[cols]]
	d1 = np.sqrt(((A[:, None, :] - A[None, :, :])**2).sum(-1))
	d2 = np.sqrt(((B[:, None, :] - B[None, :, :])**2).sum(-1))
	return cols, d1, d2

def symmetry_mask(d1, d2, R0, symmetry):
	"""Which column pairs are inside the inclusion radius.

	The --symmetry names are inverted relative to intuition; this is the
	negation of the `continue` conditions in ms.py:

	  first   skips d1 > R0                 -> keep d1 <= R0
	  both    skips d1 > R0 and d2 > R0     -> keep d1 <= R0 or  d2 <= R0
	  either  skips d1 > R0 or  d2 > R0     -> keep d1 <= R0 and d2 <= R0

	R0 is None (the daliz default) means no radius filter at all. The diagonal
	is always cleared: that is the coli == colj skip.
	"""
	if R0 is None:
		sel = np.ones(d1.shape, dtype=bool)
	elif symmetry == "first":
		sel = d1 <= R0
	elif symmetry == "both":
		sel = (d1 <= R0) | (d2 <= R0)
	else: # "either"
		sel = (d1 <= R0) & (d2 <= R0)
	np.fill_diagonal(sel, False)
	return sel

def run_rows(score_row, nr_seqs, threads):
	"""Payloads for rows 0 .. nr_seqs-2, in ascending row order.

	score_row(i) must return (i, payload) and cover every j > i in ascending j.
	Keeping that order, and reassembling the rows in ascending i here, makes the
	cross-pair accumulation match the serial scorers term for term.

	The pool is forked so workers inherit the caller's module globals
	copy-on-write instead of pickling the coordinates; those globals are set
	before the pool exists, so spawn would hand the workers None.
	"""
	rows = [ None ]*(nr_seqs - 1)
	if threads <= 1:
		for i in range(nr_seqs - 1):
			rows[i] = score_row(i)[1]
	else:
		ctx = multiprocessing.get_context("fork")
		with ctx.Pool(threads) as pool:
			# row 0 is the longest task, so hand out one row at a time
			for i, payload in pool.imap_unordered(score_row, range(nr_seqs - 1), chunksize=1):
				rows[i] = payload
	return rows
