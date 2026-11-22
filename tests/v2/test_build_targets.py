"""Build and run specific products. Enable on a source-build worker with KLONG_V2_BUILD=1."""
import json
import os
import shutil
import subprocess
import sys

import pytest

from klongpy.static.native import emit_hls_program
from .runtime import PROGRAMS, ROOT, case_names, evaluate, load, required

pytestmark = pytest.mark.v2_build
NATIVE = ROOT / "native"
PICO = "thumbv6m-none-eabi"
RISCV = "riscv32imac-unknown-none-elf"


def run(args, *, cwd=NATIVE, timeout=600):
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    assert p.returncode == 0, f"{args}\n{p.stdout}\n{p.stderr}"
    return p.stdout


@pytest.fixture(scope="module")
def cargo():
    if os.environ.get("KLONG_V2_BUILD") != "1":
        required("Set KLONG_V2_BUILD=1 on the source-build worker")
    if not shutil.which("cargo"):
        required("The profile builds need Cargo")
    toolchain = os.environ.get("KLONG_V2_CARGO_TOOLCHAIN")
    return ["cargo", f"+{toolchain}"] if toolchain else ["cargo"]


@pytest.fixture(scope="module")
def target_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("v2-target")


def build(cargo, target_dir, package, features, target=None):
    args = [*cargo, "build", "--locked", "--release", "-p", package, "--target-dir", str(target_dir)]
    if features is not None:
        args.append("--no-default-features")
        if features:
            args.extend(["--features", features])
    if target:
        args.extend(["--target", target])
    # The first desktop build can compile DuckDB from source.
    run(args, timeout=1800)


def version(command, directory):
    output = evaluate(command, ".p(.json(.ver()));1", directory)
    return json.loads(output.splitlines()[0])


@pytest.fixture(scope="module")
def installed_kg(cargo, target_dir, tmp_path_factory):
    prefix = tmp_path_factory.mktemp("v2-install")
    run([*cargo, "install", "--locked", "--path", "native/klong-cli",
         "--root", str(prefix), "--target-dir", str(target_dir)], cwd=ROOT, timeout=1800)
    return [str(prefix / "bin" / ("kg.exe" if os.name == "nt" else "kg"))]


def test_default_install_is_a_daily_driver(installed_kg, tmp_path):
    command = installed_kg
    # A freshly linked executable can undergo a platform signature check on first launch.
    # Startup latency is measured separately by the performance suite.
    assert evaluate(command, "+/!100", tmp_path, timeout=60) == "4950"
    script = tmp_path / "daily.kg"
    script.write_text('.p(+/[2 3 5])\n')
    assert run([*command, str(script)], cwd=tmp_path).strip() == "10"
    repl = subprocess.run(command, cwd=tmp_path, input="1+2\n", text=True, capture_output=True, timeout=5)
    assert repl.returncode == 0 and "3" in repl.stdout.split(), repl.stdout


def test_default_desktop_includes_database(installed_kg, tmp_path):
    command = installed_kg
    assert {"tables", "duckdb"} <= set(version(command, tmp_path)["features"])
    assert evaluate(command, 'd::.db(:{});d("SELECT 42")', tmp_path) == "42"


def test_core_only_cli_has_no_table_or_sql(cargo, target_dir, tmp_path):
    build(cargo, target_dir, "klong-cli", "")
    command = [str(target_dir / "release" / ("kg.exe" if os.name == "nt" else "kg"))]
    assert evaluate(command, "+/[2 3 5]", tmp_path) == "10"
    assert {"tables", "duckdb"}.isdisjoint(version(command, tmp_path)["features"])
    for call in ('.table(:{;"q";[1 2]})', '.db(:{})'):
        assert evaluate(command, f'(.try({{{call}}};[])@1)@"kind"', tmp_path) == '"name"'


def test_selecting_duckdb_also_provides_tables(cargo, target_dir, tmp_path):
    build(cargo, target_dir, "klong-cli", "duckdb")
    command = [str(target_dir / "release" / ("kg.exe" if os.name == "nt" else "kg"))]
    assert {"tables", "duckdb"} <= set(version(command, tmp_path)["features"])
    expr = 't::.table(:{;"qty";[2 3]});d::.db(:{;"trades";t});d("SELECT sum(qty) FROM trades")'
    assert evaluate(command, expr, tmp_path) == "5"


def test_table_only_profile(cargo, target_dir, tmp_path):
    build(cargo, target_dir, "klong-cli", "tables")
    table_only = [str(target_dir / "release" / ("kg.exe" if os.name == "nt" else "kg"))]
    features = version(table_only, tmp_path)["features"]
    assert "tables" in features and "duckdb" not in features
    assert evaluate(table_only, '(.try({.db(":memory:")};[])@1)@"kind"', tmp_path) == '"name"'
    deps = run([*cargo, "tree", "--locked", "-p", "klong-cli", "--no-default-features",
                "--features", "tables", "-e", "normal,build", "--prefix", "none"]).lower()
    assert "duckdb" not in deps
    # Replay the complete standalone table contract on an actual tables-only build.
    for name in case_names("table_cases.kg"):
        assert evaluate(table_only, f'{load("table_cases.kg")};{name}()', tmp_path) == "1", name


@pytest.mark.parametrize("target,features", [
    (PICO, "pico"), (PICO, "pico,cfun"), (PICO, "pico,tables"), (PICO, "pico,tables,cfun"),
    (RISCV, "tables"),
], ids=["pico-minimal", "pico-device-driver", "pico-tables", "pico-tables-and-device-driver", "bare-riscv-tables"])
def test_board_archive_excludes_desktop_engines(cargo, target_dir, target, features):
    build(cargo, target_dir, "klong-ffi", features, target)
    archive = target_dir / target / "release" / "libklong_ffi.a"
    assert archive.stat().st_size > 0
    deps = run([*cargo, "tree", "--locked", "-p", "klong-ffi", "--target", target,
                "--no-default-features", "--features", features, "-e", "normal,build",
                "--prefix", "none"]).lower()
    assert all(dep not in deps for dep in ("duckdb", "pyo3", "klong-py", "klong-host"))


def build_attached_board(cargo, target_dir, directory, features):
    if sys.platform not in {"linux", "darwin"} or not shutil.which("cc") or not shutil.which("c++"):
        required("Use a POSIX source-build worker with C and C++ compilers for the board ABI test")
    build(cargo, target_dir, "klong-ffi", features)
    source = PROGRAMS / "attached"
    generated = emit_hls_program(str(source / "energy.kg"), str(directory), int_bits=64,
                                 prefix="probe", wrapped=False)
    kernel = generated.entries[0]
    assert (kernel["in_bytes"], kernel["out_bytes"], kernel["top"]) == (32, 8, "probe_energy_kernel")
    flags = ["-O1", "-DKG_TRAP_FLAG=kg_hls_trap", "-DKG_INT64", "-I", str(directory)]
    run(["cc", "-std=c11", "-I", str(NATIVE / "klong-ffi" / "include"), "-c",
         str(source / "driver.c"), "-o", "driver.o"], cwd=directory)
    run(["cc", *flags, "-c", "probe.c", "-o", "cpu.o"], cwd=directory)
    run(["c++", "-std=c++14", *flags, "-c", kernel["files"]["top"], "-o", "top.o"], cwd=directory)
    libs = ["-ldl", "-lpthread", "-lm"] if sys.platform == "linux" else []
    run(["c++", "driver.o", "top.o", "cpu.o", str(target_dir / "release" / "libklong_ffi.a"),
         *libs, "-o", "board"], cwd=directory)
    return str(directory / "board")


@pytest.fixture(scope="module")
def attached_board(cargo, target_dir, tmp_path_factory):
    return build_attached_board(cargo, target_dir, tmp_path_factory.mktemp("v2-attached"), "cfun")


@pytest.mark.parametrize("mode,body,expected,submissions", [
    ("absent", "runblock([-3 4 -5 6])", 86, 0),
    ("wrong-image", "runblock([-3 4 -5 6])", 86, 0),
    ("absent", "(.try({accelerate([1 2 3 4])};[])@0)", 0, 0),
    ("attached", "runblock([-3 4 -5 6])", 86, 1),
    ("attached", "accelerate([1 2 3 4])+accelerate([5 6 7 8])", 204, 2),
    ("attached", "(.try({accelerate([1 2 3])};[])@0)", 0, 0),
    ("fault", "(.try({runblock([1 2 3 4])};[])@0)", 0, 1),
], ids=["cpu-alone", "incompatible-bitstream", "missing-driver", "device-result", "two-submissions",
        "reject-shape-before-io", "completion-error-without-replay"])
def test_attached_kernel_delivery(attached_board, mode, body, expected, submissions):
    check_delivery(attached_board, mode, body, expected, submissions)


def check_delivery(board, mode, body, expected, submissions, setup=""):
    script = (PROGRAMS / "attached" / "energy.kg").read_text()
    script += (PROGRAMS / "attached" / "route.kg").read_text()
    script += setup
    script += f'.p(.fmt("answer={{}}";{body}))\n'
    p = subprocess.run([board, mode, script], text=True, capture_output=True, timeout=10)
    assert p.returncode == 0, p.stderr
    assert any(line.removeprefix("klong> ").strip() == f"answer={expected}"
               for line in p.stdout.splitlines()), p.stdout
    assert p.stderr.strip() == f"submissions={submissions}"


def test_board_table_column_reaches_kernel(cargo, target_dir, tmp_path):
    board = build_attached_board(cargo, target_dir, tmp_path, "cfun,tables")
    for mode, submissions in [("absent", 0), ("attached", 1)]:
        check_delivery(board, mode, 'runblock(t@"reading")', 86, submissions,
                       setup='t::.table(:{;"reading";[-3 4 -5 6]})\n')
