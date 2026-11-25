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

LDDT-mu,L (lddt_mu_par.py --lengthnorm) divides the LDDT-mu column sum by the
shorter length instead: correct = 0.5, wrong = 0.5, and aligning only half of A
gives 0.25, so under-alignment is penalised without blaming correctly unaligned
residues for their neighbours.
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


def score(script, tmp_path, b_offset, row_i, row_j, extra=(), field=2):
	"""Pair score of seq_i / seq_j aligned as row_i / row_j.

	field picks the --pairs column; 3 is the length-normalised score of
	lddt_mu_par.py --lengthnorm.
	"""
	A, B, C = motifs(b_offset)
	write_pdb(tmp_path / "i.pdb", SEQ_A + SEQ_B, np.vstack([ A, B ]))
	write_pdb(tmp_path / "j.pdb", SEQ_A + SEQ_C, np.vstack([ A, C ]))
	(tmp_path / "pdbs.txt").write_text("%s\n%s\n" % (tmp_path / "i.pdb", tmp_path / "j.pdb"))
	msa = tmp_path / "msa.afa"
	msa.write_text(">seq_i\n%s\n>seq_j\n%s\n" % (row_i, row_j))
	args = [ "--msa", msa, "--pdbfiles", tmp_path / "pdbs.txt", "--pairs" ] + list(extra)
	out = run_script(script, args).stdout
	pair = next(l for l in out.splitlines() if l.startswith("seq_i\t")).split("\t")
	assert pair[:2] == [ "seq_i", "seq_j" ]
	return float(pair[field])


def score_mu_L(tmp_path, b_offset, row_i, row_j):
	"""(LDDT-mu, LDDT-mu,L) of one pair, from a single --lengthnorm run."""
	extra = [ "--lengthnorm" ]
	return (score("lddt_mu_par.py", tmp_path, b_offset, row_i, row_j, extra, 2),
			score("lddt_mu_par.py", tmp_path, b_offset, row_i, row_j, extra, 3))


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


def test_length_normalised(tmp_path):
	"""The AB/AC table of LDDT-mu,L: correct 0.5, wrong 0.5, half of A 0.25,
	with LDDT-mu itself unchanged by --lengthnorm."""
	assert score_mu_L(tmp_path, FAR, *CORRECT) == (1.0, 0.5)
	assert score_mu_L(tmp_path, FAR, *WRONG) == (0.5, 0.5)
	assert score_mu_L(tmp_path, FAR, *UNDER) == (1.0, 0.25)


def score_mu_L_cov(tmp_path, b_offset, row_i, row_j):
	"""(LDDT-mu, LDDT-mu,L, coverage) of one pair; --min_plddt 0 keeps the
	--pairs columns at mu, mu_L, cov."""
	extra = [ "--lengthnorm", "--coverage", "--min_plddt", "0" ]
	return tuple(score("lddt_mu_par.py", tmp_path, b_offset, row_i, row_j, extra, f)
				 for f in (2, 3, 4))


def test_coverage(tmp_path):
	"""Coverage of the AB/AC table: correct aligns A only (10/20), wrong aligns
	everything, half of A is 5/20. It ignores structure, and LDDT-mu,L is
	coverage times LDDT-mu."""
	for rows, cov in ((CORRECT, 0.5), (WRONG, 1.0), (UNDER, 0.25)):
		mu, mu_L, c = score_mu_L_cov(tmp_path, FAR, *rows)
		assert c == cov
		assert mu_L == pytest.approx(c*mu, abs=1e-4)


def homstrad_args(tmp_path):
	family = DATA / "homstrad" / "ChtBD"
	pdbs = tmp_path / "pdbs.txt"
	pdbs.write_text("".join("%s\n" % p for p in sorted((family / "pdb").glob("*.pdb"))))
	return [ "--msa", family / "msa.afa", "--pdbfiles", pdbs ]


def summary(out, name):
	return next(l for l in out.splitlines() if l.startswith(name + "\t")).split("\t")[1]


def test_lengthnorm_keeps_mu(tmp_path):
	"""--lengthnorm only adds output; LDDT-mu stays the same and LDDT-mu,L <= it."""
	args = homstrad_args(tmp_path) + [ "--symmetry", "both" ]
	plain = run_script("lddt_mu_par.py", args).stdout
	both = run_script("lddt_mu_par.py", args + [ "--lengthnorm" ]).stdout
	assert summary(both, "LDDT_rce_mean") == summary(plain, "LDDT_rce_mean")
	assert "LDDT_rce_L_mean" not in plain
	assert float(summary(both, "LDDT_rce_L_mean")) <= float(summary(both, "LDDT_rce_mean"))


def test_coverage_keeps_output(tmp_path):
	"""--coverage only appends COV_mean; it is in [0, 1] and LDDT-mu,L <=
	coverage."""
	args = homstrad_args(tmp_path) + [ "--symmetry", "both", "--lengthnorm" ]
	plain = run_script("lddt_mu_par.py", args).stdout
	cov = run_script("lddt_mu_par.py", args + [ "--coverage" ]).stdout
	assert cov.splitlines()[:-1] == plain.splitlines()
	assert cov.splitlines()[-1].startswith("COV_mean\t")
	c = float(summary(cov, "COV_mean"))
	assert 0.0 <= c <= 1.0
	assert float(summary(cov, "LDDT_rce_L_mean")) <= c + 1e-4


def test_coverage_threads(tmp_path):
	args = homstrad_args(tmp_path) + [ "--pairs", "--lengthnorm", "--coverage" ]
	serial = run_script("lddt_mu_par.py", args + [ "--threads", 1 ]).stdout
	parallel = run_script("lddt_mu_par.py", args + [ "--threads", 4 ]).stdout
	assert serial == parallel


def test_lengthnorm_threads(tmp_path):
	args = homstrad_args(tmp_path) + [ "--pairs", "--lengthnorm" ]
	serial = run_script("lddt_mu_par.py", args + [ "--threads", 1 ]).stdout
	parallel = run_script("lddt_mu_par.py", args + [ "--threads", 4 ]).stdout
	assert serial == parallel


def test_threads(tmp_path):
	family = DATA / "homstrad" / "ChtBD"
	pdbs = tmp_path / "pdbs.txt"
	pdbs.write_text("".join("%s\n" % p for p in sorted((family / "pdb").glob("*.pdb"))))
	args = [ "--msa", family / "msa.afa", "--pdbfiles", pdbs, "--pairs" ]
	serial = run_script("lddt_ref_par.py", args + [ "--threads", 1 ]).stdout
	parallel = run_script("lddt_ref_par.py", args + [ "--threads", 4 ]).stdout
	assert serial == parallel
