"""pLDDT-masked LDDT-mu (lddt_mu_par.py --min_plddt, default 70).

Same AB/AC motifs as test_lddt_ref.py: sequence i = AB, j = AC, B and C are not
homologous. Here the structures are written as AlphaFold models, with the
per-residue pLDDT in the B-factor column. Only columns where the residues of
both sequences are confident are scored (also as neighbours), and the
length-normalised score divides by the smaller number of confident residues.
"""

import math

import numpy as np
import pytest

from conftest import DATA, SRC, run_script
from test_lddt_ref import (ONE2THREE, SEQ_A, SEQ_B, SEQ_C, N, H, GAP, CORRECT,
						   WRONG, UNDER, FAR, NEAR, motifs)

import ms
import lddt_mu_par

HIGH = 95.0
LOW = 40.0

AF_TITLE = "TITLE     ALPHAFOLD MONOMER V2.0 PREDICTION FOR TEST PROTEIN\n"


def write_pdb(fn, seq, coords, plddt, alphafold=True):
	with open(fn, "w") as f:
		if alphafold:
			f.write(AF_TITLE)
		for k, (aa, (x, y, z), b) in enumerate(zip(seq, coords, plddt)):
			f.write("ATOM  %5d  CA  %3s A%4d    %8.3f%8.3f%8.3f  1.00%6.2f           C\n"
					% (k + 1, ONE2THREE[aa], k + 1, x, y, z, b))
		f.write("END\n")


def run(tmp_path, rows, plddt_i, plddt_j, b_offset=FAR, alphafold=True, extra=()):
	"""(pair scores, summary lines) for i = AB, j = AC aligned as rows.

	The pair scores are LDDT-mu, LDDT-mu,L, and their confident-only versions.
	"""
	A, B, C = motifs(b_offset)
	write_pdb(tmp_path / "i.pdb", SEQ_A + SEQ_B, np.vstack([ A, B ]), plddt_i, alphafold)
	write_pdb(tmp_path / "j.pdb", SEQ_A + SEQ_C, np.vstack([ A, C ]), plddt_j, alphafold)
	(tmp_path / "pdbs.txt").write_text("%s\n%s\n" % (tmp_path / "i.pdb", tmp_path / "j.pdb"))
	msa = tmp_path / "msa.afa"
	msa.write_text(">seq_i\n%s\n>seq_j\n%s\n" % rows)
	out = run_script("lddt_mu_par.py", [ "--msa", msa, "--pdbfiles", tmp_path / "pdbs.txt",
										 "--pairs", "--lengthnorm" ] + list(extra)).stdout
	lines = [ l.split("\t") for l in out.splitlines() if l ]
	pair = next(l for l in lines if l[0] == "seq_i")
	assert pair[1] == "seq_j"
	summary = { l[0]: float(l[1]) for l in lines if l[0].startswith("LDDT_") }
	return tuple(float(x) for x in pair[2:]), summary


def plddt(a, bc):
	"""pLDDT of AB (or AC): a for the A motif, bc for the B/C motif."""
	return [ a ]*N + [ bc ]*N


@pytest.mark.parametrize("b_offset", [ FAR, NEAR ])
@pytest.mark.parametrize("rows", [ CORRECT, WRONG, UNDER ], ids=[ "correct", "wrong", "under" ])
def test_all_confident_is_unmasked(tmp_path, rows, b_offset):
	(mu, mu_L, mu_c, mu_L_c), _ = run(tmp_path, rows, plddt(HIGH, HIGH), plddt(HIGH, HIGH), b_offset)
	assert (mu_c, mu_L_c) == (mu, mu_L)


def test_bfactor_of_other_structures_is_ignored(tmp_path):
	"""Without an AlphaFold header the B-factor is not a pLDDT, so nothing is
	masked even though every value is below the cutoff."""
	(mu, mu_L, mu_c, mu_L_c), _ = run(tmp_path, WRONG, plddt(0, 0), plddt(0, 0), alphafold=False)
	assert (mu, mu_L) == (0.5, 0.5)
	assert (mu_c, mu_L_c) == (mu, mu_L)


@pytest.mark.parametrize("b_offset", [ FAR, NEAR ])
def test_low_confidence_motif_is_not_scored(tmp_path, b_offset):
	"""B and C below the cutoff: only A is judged, as scored residue and as
	neighbour. Aligning B to C then costs nothing, and the unmasked
	under-alignment penalty is kept, relative to the 10 confident residues."""
	lo = plddt(HIGH, LOW)
	correct, _ = run(tmp_path, CORRECT, lo, lo, b_offset)
	wrong, _ = run(tmp_path, WRONG, lo, lo, b_offset)
	under, _ = run(tmp_path, UNDER, lo, lo, b_offset)
	assert correct[2:] == (1.0, 1.0)
	assert wrong[2:] == (1.0, 1.0)
	assert under[2:] == (1.0, 0.5)
	if b_offset == FAR:
		assert correct[:2] == (1.0, 0.5)
		assert wrong[:2] == (0.5, 0.5)
		assert under[:2] == (1.0, 0.25)
	else:
		# unmasked, the wrongly aligned B/C neighbours also hurt A
		assert wrong[0] < 0.5


def test_low_confidence_partner_costs(tmp_path):
	"""The first half of A is uncertain in j only. Those columns are not
	scored, but the confident residues of i aligned there still count in the
	denominator C_s = min(20, 15) = 15: 5 scored A residues give 5/15."""
	pj = [ LOW ]*H + [ HIGH ]*(N - H) + [ HIGH ]*N
	(mu, mu_L, mu_c, mu_L_c), _ = run(tmp_path, CORRECT, plddt(HIGH, HIGH), pj)
	assert (mu, mu_L) == (1.0, 0.5)
	assert mu_c == 1.0
	assert mu_L_c == pytest.approx(5/15, abs=1e-4)


def test_sequence_without_confident_residues(tmp_path):
	"""Nothing to judge: the pair is left out of the confident means, which
	are -1 when no pair is left."""
	pair, summary = run(tmp_path, CORRECT, plddt(HIGH, HIGH), plddt(LOW, LOW))
	assert pair[:2] == (1.0, 0.5)
	assert math.isnan(pair[2]) and math.isnan(pair[3])
	assert summary["LDDT_rce_conf_mean"] == -1
	assert summary["LDDT_rce_L_conf_mean"] == -1
	assert summary["LDDT_rce_mean"] == 1.0


def test_min_plddt_zero_disables(tmp_path):
	lo = plddt(HIGH, LOW)
	pair, summary = run(tmp_path, WRONG, lo, lo, extra=[ "--min_plddt", "0" ])
	assert pair == (0.5, 0.5)
	assert set(summary) == { "LDDT_rce_mean", "LDDT_rce_L_mean" }


def test_cutoff_is_inclusive(tmp_path):
	"""pLDDT exactly at the cutoff counts as confident."""
	at = plddt(HIGH, 70.0)
	(mu, mu_L, mu_c, mu_L_c), _ = run(tmp_path, WRONG, at, at)
	assert (mu_c, mu_L_c) == (mu, mu_L)


def test_bfactors_line_up_with_read_pdb(tmp_path):
	"""read_bfactors must skip exactly the lines ms.MSTAScorer.read_pdb skips,
	or the mask would be shifted against the coordinates."""
	fn = tmp_path / "x.pdb"
	atom = "ATOM  %5d  CA %s%3s A%4d    %8.3f%8.3f%8.3f  1.00%6.2f           C\n"
	with open(fn, "w") as f:
		f.write(AF_TITLE)
		f.write(atom % (1, " ", "ALA", 1, 0, 0, 0, 91.0))
		f.write(atom % (2, " ", "UNK", 2, 1, 0, 0, 12.0))      # not in three2one
		f.write(atom % (3, "A", "GLY", 3, 2, 0, 0, 55.0))      # altloc A kept
		f.write(atom % (4, "B", "GLY", 3, 2, 0, 0, 13.0))      # altloc B skipped
		f.write("ATOM      5  CB  GLY A   3       2.000   1.000   0.000  1.00 14.00           C\n")
		f.write(atom % (6, " ", "SER", 4, 3, 0, 0, 77.0))
		f.write("TER \n")
		f.write(atom % (7, " ", "LYS", 5, 4, 0, 0, 15.0))      # after TER
	scorer = ms.MSTAScorer()
	seq, xs, _, _ = scorer.read_pdb(str(fn))
	b = lddt_mu_par.read_bfactors(scorer, str(fn))
	assert seq == "AGS"
	assert b.tolist() == [ 91.0, 55.0, 77.0 ]
	assert lddt_mu_par.is_alphafold(str(fn))


AFDB = next(p for p in sorted((DATA / "afdb").iterdir()) if (p / "msa.afa").is_file())


def afdb_args(tmp_path):
	pdbs = tmp_path / "pdbs.txt"
	pdbs.write_text("".join("%s\n" % p for p in sorted((AFDB / "pdb").glob("*.pdb"))))
	return [ "--msa", AFDB / "msa.afa", "--pdbfiles", pdbs, "--symmetry", "both" ]


def summary_of(out):
	return { l.split("\t")[0]: float(l.split("\t")[1])
			 for l in out.splitlines() if l.startswith("LDDT_") }


def test_afdb_mask_is_active_and_keeps_unmasked_scores(tmp_path):
	"""On real AlphaFold models the mask removes something, and the unmasked
	scores are the same as with --min_plddt 0."""
	args = afdb_args(tmp_path) + [ "--lengthnorm" ]
	off = summary_of(run_script("lddt_mu_par.py", args + [ "--min_plddt", "0" ]).stdout)
	on = run_script("lddt_mu_par.py", args)
	s = summary_of(on.stdout)
	assert "AlphaFold models" in on.stderr
	assert s["LDDT_rce_mean"] == off["LDDT_rce_mean"]
	assert s["LDDT_rce_L_mean"] == off["LDDT_rce_L_mean"]
	assert s["LDDT_rce_conf_mean"] != s["LDDT_rce_mean"]
	assert s["LDDT_rce_L_conf_mean"] != s["LDDT_rce_L_mean"]


def test_afdb_threads(tmp_path):
	args = afdb_args(tmp_path) + [ "--lengthnorm", "--pairs" ]
	serial = run_script("lddt_mu_par.py", args + [ "--threads", 1 ]).stdout
	parallel = run_script("lddt_mu_par.py", args + [ "--threads", 4 ]).stdout
	assert serial == parallel
