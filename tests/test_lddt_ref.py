"""LDDT-mu behaves like precision, LDDT-ref like recall.

Three synthetic motifs A, B, C of equal length. Sequence i = AB, sequence
j = AC, and B and C are not homologous: C has B's shape scaled by 2, so every
B/C distance pair differs by more than the largest threshold. The correct
alignment is AB- / A-C, a wrong one is AB / AC.

LDDT-mu (lddt_mu_par.py) only looks at aligned columns: correct = 1, wrong = 0.5.
LDDT-ref (lddt_ref_par.py) normalises by the reference length: correct = 0.5,
wrong = 0.5. It cannot tell a residue that is correctly left unaligned (B) from
a missed one. When B lies within R0 of A it is even worse: B's gaps turn into
failed neighbours of A, and the correct alignment drops below 0.5.
"""

import numpy as np
import pytest

from conftest import DATA, run_script

N = 10  # residues per motif

ONE2THREE = { 'A':'ALA', 'C':'CYS', 'D':'ASP', 'E':'GLU', 'F':'PHE', 'G':'GLY',
			  'H':'HIS', 'I':'ILE', 'K':'LYS', 'L':'LEU', 'M':'MET', 'N':'ASN',
			  'P':'PRO', 'Q':'GLN', 'R':'ARG', 'S':'SER', 'T':'THR', 'V':'VAL',
			  'W':'TRP', 'Y':'TYR' }

SEQ_A = "ACDEFGHIKL"
SEQ_B = "MNPQRSTVWY"
SEQ_C = "YWVTSRQPNM"


def random_points(seed, box, min_dist):
	"""N points in a box x box x box cube, pairwise at least min_dist apart."""
	rng = np.random.default_rng(seed)
	pts = []
	while len(pts) < N:
		p = rng.uniform(0, box, 3)
		if all(np.linalg.norm(p - q) >= min_dist for q in pts):
			pts.append(p)
	return np.array(pts)


def motifs(b_offset):
	"""(A, B, C) coordinates; B and C are shifted by b_offset along x.

	A fits in an 8 A cube, so all A residues are within R0 = 15 of each other
	and every A residue has neighbours. B residues are >= 4.5 A apart, so
	scaling by 2 moves every B/B distance by >= 4.5 A > 4 A.
	"""
	A = random_points(1, 8.0, 3.0)
	B = random_points(2, 10.0, 4.5)
	C = B.mean(0) + 2.0*(B - B.mean(0))
	shift = np.array([ b_offset, 0.0, 0.0 ])
	return A, B + shift, C + shift


def write_pdb(fn, seq, coords):
	with open(fn, "w") as f:
		for k, (aa, (x, y, z)) in enumerate(zip(seq, coords)):
			f.write("ATOM  %5d  CA  %3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00           C\n"
					% (k + 1, ONE2THREE[aa], k + 1, x, y, z))
		f.write("END\n")


def score(script, tmp_path, b_offset, row_i, row_j):
	"""Pair score of seq_i / seq_j aligned as row_i / row_j."""
	A, B, C = motifs(b_offset)
	write_pdb(tmp_path / "i.pdb", SEQ_A + SEQ_B, np.vstack([ A, B ]))
	write_pdb(tmp_path / "j.pdb", SEQ_A + SEQ_C, np.vstack([ A, C ]))
	(tmp_path / "pdbs.txt").write_text("%s\n%s\n" % (tmp_path / "i.pdb", tmp_path / "j.pdb"))
	msa = tmp_path / "msa.afa"
	msa.write_text(">seq_i\n%s\n>seq_j\n%s\n" % (row_i, row_j))
	out = run_script(script, [ "--msa", msa, "--pdbfiles", tmp_path / "pdbs.txt", "--pairs" ]).stdout
	pair = out.splitlines()[0].split("\t")
	assert pair[:2] == [ "seq_i", "seq_j" ]
	return float(pair[2])


GAP = "-"*N
CORRECT = (SEQ_A + SEQ_B + GAP, SEQ_A + GAP + SEQ_C)
WRONG = (SEQ_A + SEQ_B, SEQ_A + SEQ_C)
# only the first half of A aligned, everything else gapped
H = N//2
UNDER = (SEQ_A[:H] + SEQ_A[H:] + "-"*H + SEQ_B + GAP,
		 SEQ_A[:H] + "-"*H + SEQ_A[H:] + GAP + SEQ_C)

FAR = 100.0   # B/C far beyond R0 from A
NEAR = 10.0   # B/C within R0 of A


def test_far_motifs(tmp_path):
	"""The exact numbers of the AB/AC example."""
	mu_correct = score("lddt_mu_par.py", tmp_path, FAR, *CORRECT)
	mu_wrong = score("lddt_mu_par.py", tmp_path, FAR, *WRONG)
	ref_correct = score("lddt_ref_par.py", tmp_path, FAR, *CORRECT)
	ref_wrong = score("lddt_ref_par.py", tmp_path, FAR, *WRONG)
	assert mu_correct == 1.0
	assert mu_wrong == 0.5
	assert ref_correct == 0.5
	assert ref_wrong == 0.5


def test_near_motifs(tmp_path):
	"""B within R0 of A: the gapped B neighbours make the correct alignment fail
	in LDDT-ref, which then prefers the wrong alignment (about 0.31 vs 0.46)."""
	mu_correct = score("lddt_mu_par.py", tmp_path, NEAR, *CORRECT)
	mu_wrong = score("lddt_mu_par.py", tmp_path, NEAR, *WRONG)
	ref_correct = score("lddt_ref_par.py", tmp_path, NEAR, *CORRECT)
	ref_wrong = score("lddt_ref_par.py", tmp_path, NEAR, *WRONG)
	assert mu_correct == 1.0
	assert mu_wrong < mu_correct
	assert ref_correct < 0.5
	assert ref_correct < ref_wrong


def test_under_alignment(tmp_path):
	"""Aligning only half of A is free in LDDT-mu but costs recall in LDDT-ref.

	Each aligned A residue keeps 4 of its 9 A neighbours, so LDDT-ref is
	5 * 4/9 / 20 = 1/9.
	"""
	mu = score("lddt_mu_par.py", tmp_path, FAR, *UNDER)
	ref = score("lddt_ref_par.py", tmp_path, FAR, *UNDER)
	assert mu == 1.0
	assert ref == pytest.approx(1/9, abs=1e-4)


def test_threads(tmp_path):
	family = DATA / "homstrad" / "ChtBD"
	pdbs = tmp_path / "pdbs.txt"
	pdbs.write_text("".join("%s\n" % p for p in sorted((family / "pdb").glob("*.pdb"))))
	args = [ "--msa", family / "msa.afa", "--pdbfiles", pdbs, "--pairs" ]
	serial = run_script("lddt_ref_par.py", args + [ "--threads", 1 ]).stdout
	parallel = run_script("lddt_ref_par.py", args + [ "--threads", 4 ]).stdout
	assert serial == parallel
