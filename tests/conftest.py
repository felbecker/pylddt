"""Shared fixtures for the pylddt equivalence tests.

The scorers are compared by running them as subprocesses. That is not just
convenience: the serial scripts execute argparse at import time, so they cannot
be imported and called in-process. Both sides are run with sys.executable so the
serial and parallel numbers come from one interpreter and one numpy.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = Path(__file__).resolve().parent / "data"

# so the unit tests can import the shared vectorized helpers directly
sys.path.insert(0, str(SRC))


def run_script(script, args, check=True):
	"""Run src/<script> and return the CompletedProcess.

	With check, an empty stdout is a failure as well as a non-zero exit: a
	crashed scorer otherwise compares equal to another crashed scorer.
	"""
	cmd = [ sys.executable, str(SRC / script) ] + [ str(a) for a in args ]
	proc = subprocess.run(cmd, capture_output=True, text=True)
	if check:
		assert proc.returncode == 0, \
			"%s exited %d\n%s" % (script, proc.returncode, proc.stderr)
		assert proc.stdout.strip(), \
			"%s produced no stdout\n%s" % (script, proc.stderr)
	return proc


@pytest.fixture(scope="session")
def pdbfiles(tmp_path_factory):
	"""family -> a --pdbfiles list holding absolute paths to its structures."""
	out_dir = tmp_path_factory.mktemp("pdbfiles")
	cache = {}

	def make(family):
		if family.path not in cache:
			fn = out_dir / ("%s_%s.files" % (family.collection, family.name))
			fn.write_text("".join("%s\n" % p for p in family.pdbs()))
			cache[family.path] = fn
		return cache[family.path]

	return make


@pytest.fixture(scope="session")
def serial_output():
	"""Memoized stdout of a serial scorer.

	Each serial run is reused across the --threads settings it is compared
	against; the serial scorers are the expensive half of this suite.
	"""
	cache = {}

	def get(script, msa, pdbfiles_fn, opts):
		key = (script, str(msa), str(pdbfiles_fn), tuple(opts))
		if key not in cache:
			args = [ "--msa", msa, "--pdbfiles", pdbfiles_fn ] + list(opts)
			cache[key] = run_script(script, args).stdout
		return cache[key]

	return get
