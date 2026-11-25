"""Every parallel scorer must reproduce its serial original exactly.

For each scorer, family and option set, the serial script and the parallel
script are run and their stdout compared verbatim. Exact string equality is the
right assertion here: snakeMSA parses these fields out of stdout, so anything
that survives the printed precision is a real behaviour change. Running the
parallel side at several --threads settings also pins that the output does not
depend on how the work was distributed.
"""

import difflib
import re

import numpy as np
import pytest

from conftest import DATA, run_script

import ms_np


class Family:
	"""One vendored test family: an MSA plus one PDB per sequence."""

	def __init__(self, path):
		self.path = path
		self.collection = path.parent.name
		self.name = path.name
		self.msa = path / "msa.afa"

	def pdbs(self):
		return sorted((self.path / "pdb").glob("*.pdb"))

	def __str__(self):
		return "%s/%s" % (self.collection, self.name)


def _discover():
	fams = [ Family(p) for p in sorted(DATA.glob("*/*")) if (p / "msa.afa").is_file() ]
	assert fams, "no test families found under %s" % DATA
	return fams


FAMILIES = _discover()

# The AFDB family is the deliberately larger case; the serial scorers it is
# checked against are pure-Python O(pairs * cols^2) loops.
SLOW_COLLECTIONS = ( "afdb", )

FAMILY_PARAMS = [
	pytest.param(f, id=str(f),
				 marks=[ pytest.mark.slow ] if f.collection in SLOW_COLLECTIONS else [])
	for f in FAMILIES
]

SMALL = next(f for f in FAMILIES if f.name == "ChtBD")

SCRIPTS = {
	"daliz":          ( "daliz.py",          "daliz_par.py" ),
	"lddt_foldmason": ( "lddt_foldmason.py", "lddt_foldmason_par.py" ),
	"lddt_mu":        ( "lddt_mu.py",        "lddt_mu_par.py" ),
}

# Options the parallel side needs to print exactly what the serial one does.
# lddt_mu_par.py adds pLDDT-masked scores unless --min_plddt is 0.
PARALLEL_ONLY = {
	"lddt_mu": [ "--min_plddt", "0" ],
}

# daliz leaves R0 unset by default, so --symmetry does nothing unless a radius
# is given too; pair them, or the mask would never be exercised.
OPTIONS = {
	"daliz": [
		[],
		[ "--radius", "15" ],
		[ "--radius", "15", "--symmetry", "both" ],
		[ "--radius", "15", "--symmetry", "either" ],
		[ "--horizon", "10", "--diagwt", "0.5" ],
	],
	"lddt_foldmason": [
		[],
		[ "--symmetry", "both" ],
		[ "--symmetry", "either" ],
		[ "--radius", "10" ],
		# calc_col_score hardcodes 0.5/1/2/4 and ignores --dists; the parallel
		# version must ignore it identically
		[ "--dists", "1,3" ],
		[ "--cols" ],
	],
	"lddt_mu": [
		[],
		[ "--symmetry", "both" ],
		[ "--symmetry", "either" ],
		[ "--radius", "10" ],
		[ "--dists", "0.5,2" ],
		[ "--pairs" ],
	],
}

SCORER_OPT_PARAMS = [
	pytest.param(scorer, opts, id="%s[%s]" % (scorer, " ".join(opts) or "default"))
	for scorer in sorted(SCRIPTS)
	for opts in OPTIONS[scorer]
]


def assert_same_output(expected, actual, context):
	"""Exact stdout equality, with a readable message when it fails."""
	if expected == actual:
		return
	numbers = re.compile(r"-?\d+\.\d+|-?\d+")
	e = [ float(x) for x in numbers.findall(expected) ]
	a = [ float(x) for x in numbers.findall(actual) ]
	detail = ""
	if len(e) == len(a) and e:
		deltas = [ (abs(x - y), i, x, y) for i, (x, y) in enumerate(zip(e, a)) ]
		worst = max(deltas)
		detail = ("\nlargest numeric delta %g at field %d (serial %r, parallel %r)"
				  % (worst[0], worst[1], worst[2], worst[3]))
	diff = "\n".join(difflib.unified_diff(
		expected.splitlines(), actual.splitlines(),
		"serial", "parallel", lineterm="", n=1))
	pytest.fail("%s: stdout differs%s\n%s" % (context, detail, diff[:2000]))


# ============================================================
# serial vs parallel
# ============================================================

@pytest.mark.parametrize("threads", [ 1, 4 ])
@pytest.mark.parametrize("family", FAMILY_PARAMS)
@pytest.mark.parametrize("scorer,opts", SCORER_OPT_PARAMS)
def test_parallel_matches_serial(scorer, opts, family, threads, pdbfiles, serial_output):
	serial, parallel = SCRIPTS[scorer]
	files = pdbfiles(family)
	expected = serial_output(serial, family.msa, files, opts)
	args = [ "--msa", family.msa, "--pdbfiles", files, "--threads", threads ] + opts \
		+ PARALLEL_ONLY.get(scorer, [])
	actual = run_script(parallel, args).stdout
	assert_same_output(expected, actual,
					   "%s %s %s --threads %d" % (parallel, family, " ".join(opts), threads))


@pytest.mark.parametrize("threads", [ 2, 3, 5, 8 ])
@pytest.mark.parametrize("scorer", sorted(SCRIPTS))
def test_thread_count_invariance(scorer, threads, pdbfiles, serial_output):
	"""Output must not depend on the pool size, including sizes that do not
	divide the number of rows evenly (results come back via imap_unordered)."""
	serial, parallel = SCRIPTS[scorer]
	files = pdbfiles(SMALL)
	expected = serial_output(serial, SMALL.msa, files, [])
	args = [ "--msa", SMALL.msa, "--pdbfiles", files, "--threads", threads ] \
		+ PARALLEL_ONLY.get(scorer, [])
	actual = run_script(parallel, args).stdout
	assert_same_output(expected, actual, "%s --threads %d" % (parallel, threads))


# ============================================================
# unmatched sequences
# ============================================================

@pytest.fixture(scope="session")
def msa_with_unmatched(tmp_path_factory):
	"""SMALL's MSA plus one row that matches no structure."""
	rows = SMALL.msa.read_text().rstrip("\n").split("\n")
	ncols = len(rows[1])
	out = tmp_path_factory.mktemp("unmatched") / "msa.afa"
	out.write_text("%s\n>no_such_structure\n%s\n" % ("\n".join(rows), "A"*ncols))
	return out


def test_daliz_tolerates_unmatched(msa_with_unmatched, pdbfiles):
	"""daliz scores what it can and reports the rest; daliz_par must agree.

	This is the one path where nr_matched < nr_seqs, so indexing through
	msa_idxs/pdb_idxs rather than assuming msa_idxs[i] == i actually matters.
	"""
	files = pdbfiles(SMALL)
	args = [ "--msa", msa_with_unmatched, "--pdbfiles", files ]
	expected = run_script("daliz.py", args).stdout
	actual = run_script("daliz_par.py", args + [ "--threads", 4 ]).stdout
	assert "MSA_label_not_matched=no_such_structure" in expected
	assert_same_output(expected, actual, "daliz_par.py with an unmatched sequence")


@pytest.mark.parametrize("scorer", [ "lddt_foldmason", "lddt_mu" ])
def test_lddt_scorers_reject_unmatched(scorer, msa_with_unmatched, pdbfiles):
	"""Both LDDT scorers require a full match, and both halves must agree on
	refusing: their index arrays are only valid when every sequence matched."""
	files = pdbfiles(SMALL)
	args = [ "--msa", msa_with_unmatched, "--pdbfiles", files ]
	serial, parallel = SCRIPTS[scorer]
	a = run_script(serial, args, check=False)
	b = run_script(parallel, args + [ "--threads", 4 ], check=False)
	assert a.returncode == 1, a.stderr
	assert b.returncode == 1, b.stderr
	assert "ERROR" in a.stderr and "ERROR" in b.stderr


# ============================================================
# fixtures and helpers
# ============================================================

def test_families_are_complete():
	"""Every sequence has a structure, or the scorers would refuse the family."""
	for family in FAMILIES:
		labels = [ ln[1:].strip() for ln in family.msa.read_text().splitlines()
				   if ln.startswith(">") ]
		assert labels, "%s has no sequences" % family
		assert len(labels) == len(family.pdbs()), \
			"%s: %d sequences but %d PDBs" % (family, len(labels), len(family.pdbs()))


def test_afdb_family_has_all_gap_columns():
	"""The AFDB family is a subsample of a 115-sequence alignment, so it keeps
	all-gap columns. Those hit calc_col_score's nr_seq_pairs == 0 branch, which
	is the FoldMason gap accounting this repo exists to document."""
	family = next(f for f in FAMILIES if f.collection == "afdb")
	rows = [ ln for ln in family.msa.read_text().splitlines() if not ln.startswith(">") ]
	all_gap = sum(1 for col in zip(*rows) if set(col) <= set("-."))
	assert all_gap > 0, "%s has no all-gap columns" % family


@pytest.mark.parametrize("symmetry", [ "first", "both", "either" ])
def test_symmetry_mask_matches_ms_branches(symmetry):
	"""ms_np.symmetry_mask against the `continue` conditions it negates.

	The --symmetry names are inverted relative to intuition, so this is worth
	pinning independently of the end-to-end comparisons.
	"""
	rng = np.random.default_rng(0)
	n = 12
	R0 = 15.0
	d1 = rng.uniform(0, 30, (n, n))
	d2 = rng.uniform(0, 30, (n, n))
	sel = ms_np.symmetry_mask(d1, d2, R0, symmetry)
	for i in range(n):
		for j in range(n):
			if i == j:
				expected = False
			elif symmetry == "first":
				expected = not (d1[i][j] > R0)
			elif symmetry == "both":
				expected = not (d1[i][j] > R0 and d2[i][j] > R0)
			else:
				expected = not (d1[i][j] > R0 or d2[i][j] > R0)
			assert bool(sel[i][j]) == expected, (symmetry, i, j)


def test_symmetry_mask_without_radius():
	"""daliz's default: no inclusion radius, but the diagonal is still skipped."""
	d = np.full((5, 5), 1e9)
	sel = ms_np.symmetry_mask(d, d, None, "first")
	assert sel.sum() == 5*5 - 5
	assert not sel.diagonal().any()
