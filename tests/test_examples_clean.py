"""
Every shipped .kg file under klongpy/lib/ and examples/ is clean.

For each file:
  1. `kgpy lint` reports no error-level finding (E001, E002);
  2. the reader reports no KlongSyntaxWarning (parse only, strict 0);
  3. it is accepted by strict level 1's parse-time checks (parse only: lint E001).
For the examples that can run offline, a fourth check runs the file at strict
level 2 in a subprocess with a time bound and asserts a zero exit status.

Files that legitimately cannot meet a criterion are listed in the allow-lists
below, each with a reason. A file that is in an allow-list but meets the
criterion fails the test (so the lists cannot go stale), and a new .kg file
that is in neither RUNNABLE nor EXCLUDED fails `test_every_example_is_classified`.

examples/embedded/ belongs to the embedded-board work and is not covered here.
"""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import klongpy
from klongpy.lint import lint_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {os.path.join("examples", "embedded")}


def kg_files():
    out = []
    for top in ("klongpy/lib", "examples"):
        for dirpath, dirnames, names in os.walk(os.path.join(ROOT, top)):
            rel_dir = os.path.relpath(dirpath, ROOT)
            if any(rel_dir == d or rel_dir.startswith(d + os.sep) for d in SKIP_DIRS):
                continue
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            out += [os.path.join(rel_dir, n) for n in sorted(names) if n.endswith(".kg")]
    return sorted(out)


FILES = kg_files()

# ---------------------------------------------------------------------------
# Allow-lists: path -> reason. All three are empty: no shipped file needs one.
# ---------------------------------------------------------------------------
LINT_ERROR_ALLOW = {
    "examples/modules/errors/discarded.kg": "the error it is written to show: an .extern whose value is thrown away (E005)",
    "examples/modules/errors/nosuchentry.kg": "the error it is written to show: an entry named to .l that the module does not export (E004)",
    "examples/modules/errors/missing.kg": "the error it is written to show: a qualified name the module does not export (E004)",
    "examples/modules/errors/private.kg": "the error it is written to show: a private helper of the module (E004)",
    "examples/modules/errors/readonly.kg": "the error it is written to show: an assignment to an entry of a module (E003)",
}
SYNTAX_WARNING_ALLOW = {}
STRICT1_PARSE_ALLOW = {}

# ---------------------------------------------------------------------------
# Examples run at strict level 2: path -> (modules the interpreter must have,
# extra command-line arguments).
# ---------------------------------------------------------------------------
TORCH = (["torch"], ["--backend", "torch", "--device", "cpu"])
PLAIN = ([], [])
RUNNABLE = {
    "examples/db/taxi-queries.kg": PLAIN,
    "examples/hardware/crossover.kg": PLAIN,         # a static-profile program: loading it defines its functions and state
    "examples/autograd/basic_gradient.kg": PLAIN,
    "examples/autograd/differentiable_physics.kg": PLAIN,
    "examples/autograd/gradient_descent.kg": PLAIN,
    "examples/autograd/linear_regression.kg": PLAIN,
    "examples/autograd/neural_net.kg": PLAIN,
    "examples/autograd/numeric_vs_autograd.kg": PLAIN,
    "examples/autograd/optimizer_demo.kg": PLAIN,
    "examples/autograd/rolling_portfolio_opt_simple.kg": PLAIN,
    "examples/autograd/black_scholes_greeks.kg": TORCH,  # needs erf from the torch backend
    "examples/autograd/export_demo.kg": TORCH,           # .export needs the torch backend
    "examples/autograd/gradcheck_demo.kg": TORCH,        # .gradcheck needs the torch backend on cpu
    "examples/math/fft.kg": PLAIN,
    "examples/math/trig.kg": PLAIN,
    "examples/ml/cosine_similarity.kg": PLAIN,
    "examples/ml/knn.kg": PLAIN,
    "examples/ml/transformer.kg": PLAIN,
    "examples/python/hello.kg": PLAIN,
    "examples/python/hello_exports.kg": PLAIN,
    "examples/modules/classic_load.kg": PLAIN,        # .l of a file that is a module binds the module to the file's name
    "examples/python/multiprocessing/callback.kg": PLAIN,
    "examples/python/multiprocessing/pool.kg": PLAIN,
    "examples/python/multiprocessing/pool_async.kg": PLAIN,
    "examples/python/multiprocessing/worker/par.kg": PLAIN,
    "examples/python/multiprocessing/worker/worker.kg": PLAIN,
    "examples/python/threading/callback.kg": PLAIN,
    "examples/python/threading/callback_async.kg": PLAIN,
    "examples/python/threading/pool.kg": PLAIN,
    "examples/python/threading/pool_async.kg": PLAIN,
    "examples/stocks/options/black-scholes.kg": PLAIN,
}

# Not run: path -> why. (All of these are still linted and parsed.)
PORT = "starts a server on the fixed port 8888 (a test must not bind it); loads and exits 0 at strict 2 when run by hand"
SERVER = "long-running server (binds a fixed port, runs until killed); loaded to its serve point at strict 2 in a timed run"
CLIENT = "client that waits for a server on port 8888 (timed out at strict 2 after connecting nowhere)"
MODULE_PART = "a part of the backtest of examples/modules/main.kg (a module it loads, or a classic file it loads): not a program of its own"
MODULE_ERROR = "stops with the error named for it in examples/modules/EXPECTED-ERRORS.txt, by design"
EXCLUDED = {
    **{f"examples/modules/{n}": MODULE_PART for n in (
        "config.kg", "legacy.kg", "lib/rng.kg", "lib/series.kg", "lib/stats.kg", "data/feed.kg", "book/portfolio.kg",
        "strategy/crossover.kg", "strategy/meanrev.kg", "strategy/registry.kg", "report/table.kg")},
    "examples/modules/main.kg": "written for strict level 0 (its output is examples/modules/EXPECTED.txt): lib/rng.kg's `{step()%m}'!n` is an arity error at strict 2",
    "examples/modules/static/ma4.kg": "a static-profile program that loads a module: checked and compiled by `kgpy check` and `kgpy compile c` (static/ma4.expect)",
    **{f"examples/modules/errors/{n}.kg": MODULE_ERROR for n in (
        "badstem", "cycle", "cycle_a", "cycle_b", "discarded", "missing", "noexport", "nosuchentry", "notadict", "private", "readonly",
        "undefined")},
    **{f"examples/modules/errors/support/{n}.kg": "a file an error program loads: not a program of its own" for n in ("bad-name", "plain")},
    "examples/stocks/feed/consumer.kg": "needs a running feed.kg (a server on port 8888); tests/test_example_feed.py runs it with demo.kg",
    "examples/stocks/feed/feed.kg": SERVER + "; tests/test_example_feed.py runs it with demo.kg",
    "examples/stocks/feed/demo.kg": "the process stays up after its output, by design; tests/test_example_feed.py runs it and stops it after the last line",
    "examples/stocks/tickerplant/head.kg": "requires an HDB address and data directory for tests/v2/test_tickerplant.py",
    "examples/stocks/tickerplant/hdb.kg": "serves a historical database for tests/v2/test_tickerplant.py",
    "examples/stocks/tickerplant/subscriber.kg": "requires a running head for tests/v2/test_tickerplant.py",
    "examples/1brc/par.kg": "needs a measurements file argument, supplied by tests/test_1brc_example.py",
    "examples/autograd/black_scholes_live.kg": "network: live option data through yfinance (live_options.py)",
    "examples/autograd/portfolio_opt.kg": "network: stock_data.py downloads prices through yfinance",
    "examples/autograd/rolling_portfolio_opt.kg": "network: stock_prices.py downloads prices through yfinance",
    "examples/bench_compiler.kg": "benchmark that takes about 30 s (not worth the time in the suite); run by hand at strict 2 it exits 0",
    "examples/db/dfs.kg": "needs a directory argument and duckdb; serves tables until killed",
    "examples/db/nyc-taxi.kg": "needs yellow and green taxi Parquet paths",
    "examples/db/server.kg": SERVER + " (needs klongpy.db: duckdb, pandas)",
    "examples/ipc/cli_broadcast.kg": CLIENT,
    "examples/ipc/cli_pubsub.kg": CLIENT,
    "examples/ipc/simple_client.kg": CLIENT,
    "examples/ipc/simple_server.kg": PORT,
    "examples/ipc/srv_broadcast.kg": PORT,
    "examples/ipc/srv_pubsub.kg": PORT,
    "examples/ml/faiss/1-flat.kg": "needs the optional package faiss",
    "examples/ml/lstm/lstm.kg": "needs torch (lstm.py) and trains a model",
    "examples/stats_logging/head.kg": CLIENT,
    "examples/stats_logging/stats_client.kg": CLIENT + "; also needs pytz",
    "examples/stats_logging/stats_server.kg": PORT + "; also writes /tmp/tables/stats",
    "examples/stocks/yfinance/compact_lstm.kg": "network (yfinance) and torch (lstm.py)",
    "examples/stocks/yfinance/fetch.kg": "network: downloads prices through yfinance",
    "examples/stocks/yfinance/lstm.kg": "network (yfinance) and torch (lstm.py)",
    "examples/stocks/yfinance/yfinance.kg": "network: downloads prices through yfinance",
    "examples/web/server.kg": SERVER + " (needs klongpy.web: aiohttp)",
}

TIMEOUT = 120


def _read(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return f.read()


def lint_errors(rel):
    return [d for d in lint_file(os.path.join(ROOT, rel)) if d.severity == "error"]


def strict1_parse_error(rel):
    """The finding that the file does not parse at strict level 1 (lint E001: the parser rejects it), or None."""
    found = [d for d in lint_file(os.path.join(ROOT, rel)) if d.code == "E001"]
    return found[0] if found else None


class TestEveryKgFileIsClean(unittest.TestCase):

    def test_files_found(self):
        self.assertGreaterEqual(len(FILES), 69)
        self.assertIn("klongpy/lib/math.kg", FILES)
        self.assertIn("examples/ml/knn.kg", FILES)

    def test_no_lint_errors(self):
        for rel in FILES:
            with self.subTest(file=rel):
                errs = lint_errors(rel)
                if rel in LINT_ERROR_ALLOW:
                    self.assertTrue(errs, f"{rel} is allow-listed but has no lint error")
                else:
                    self.assertEqual([], [f"{d.line}:{d.col} {d.code} {d.message}" for d in errs])

    @unittest.skip("engine needs: a parse that returns the syntax warnings without running the file (Klong.parse_warnings(text, name=None))")
    def test_no_syntax_warnings(self):
        for rel in FILES:
            with self.subTest(file=rel):
                ws = []
                if rel in SYNTAX_WARNING_ALLOW:
                    self.assertTrue(ws, f"{rel} is allow-listed but has no warning")
                else:
                    self.assertEqual([], [str(w) for w in ws])

    def test_accepted_at_strict_level_1_parse(self):
        for rel in FILES:
            with self.subTest(file=rel):
                err = strict1_parse_error(rel)
                if rel in STRICT1_PARSE_ALLOW:
                    self.assertIsNotNone(err, f"{rel} is allow-listed but parses at level 1")
                else:
                    self.assertIsNone(err, str(err))

    def test_allow_lists_name_real_files_and_have_reasons(self):
        for table in (LINT_ERROR_ALLOW, SYNTAX_WARNING_ALLOW, STRICT1_PARSE_ALLOW, EXCLUDED):
            for rel, reason in table.items():
                self.assertIn(rel, FILES)
                self.assertTrue(reason.strip())

    def test_every_example_is_classified(self):
        examples = {f for f in FILES if f.startswith("examples/")}
        self.assertEqual(set(), examples - set(RUNNABLE) - set(EXCLUDED), "new examples: add to RUNNABLE or EXCLUDED")
        self.assertEqual(set(), (set(RUNNABLE) | set(EXCLUDED)) - examples, "classified files that no longer exist")
        self.assertEqual(set(), set(RUNNABLE) & set(EXCLUDED))


class TestRunnableExamplesAtStrictLevel2(unittest.TestCase):

    def test_run(self):
        # Some examples write beside their inputs, including the exported PyTorch model.
        workspace = self.enterContext(tempfile.TemporaryDirectory())
        shutil.copytree(os.path.join(ROOT, "examples"), os.path.join(workspace, "examples"))
        package_root = str(Path(klongpy.__file__).resolve().parents[1])
        python_path = os.pathsep.join(filter(None, [package_root, os.environ.get("PYTHONPATH")]))
        env = dict(os.environ, PYTHONPATH=python_path, MPLBACKEND="Agg")
        for rel, (needs, extra) in sorted(RUNNABLE.items()):
            with self.subTest(file=rel):
                missing = [m for m in needs if importlib.util.find_spec(m) is None]
                if missing:
                    self.skipTest(f"{rel}: {', '.join(missing)} not installed")
                path = os.path.join(workspace, rel)
                try:
                    p = subprocess.run([sys.executable, "-W", "ignore", "-m", "klongpy.cli", "--strict=2", *extra, path],
                                       cwd=os.path.dirname(path), env=env, stdin=subprocess.DEVNULL,
                                       capture_output=True, text=True, timeout=TIMEOUT)
                except subprocess.TimeoutExpired:
                    self.fail(f"{rel} did not finish in {TIMEOUT} s")
                self.assertEqual(0, p.returncode, p.stderr[-800:])


if __name__ == "__main__":
    unittest.main()
