from collections import defaultdict
from decimal import Decimal
from fractions import Fraction
from math import floor
from pathlib import Path
import os
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "1brc"


def reference(rows):
    stations = defaultdict(list)
    for name, temperature in rows:
        stations[name].append(int(Decimal(temperature) * 10))

    def stats(tenths):
        mean = Fraction(sum(tenths), len(tenths))
        rounded = floor(mean + Fraction(1, 2))
        return "/".join(
            f"{Decimal(value).scaleb(-1):.1f}" for value in (min(tenths), rounded, max(tenths))
        )

    names = sorted(stations, key=lambda name: name.encode("utf-16-be"))
    return "{" + ", ".join(f"{name}={stats(stations[name])}" for name in names) + "}\n"


@pytest.fixture(params=["kgpy", "kg"])
def run_example(request):
    if request.param == "kg":
        kg = Path(os.environ.get("KLONG_NATIVE", ROOT / "native/target/release/kg"))
        if not kg.is_file():
            pytest.skip("build the standalone kg CLI to run this case")
        command = [str(kg.resolve()), str(EXAMPLE / "par.kg")]
        cwd = ROOT
    else:
        kgpy = os.environ.get("KLONG_V2_KGPY")
        command = [kgpy] if kgpy else [sys.executable, "-m", "klongpy.cli"]
        command += [str(EXAMPLE / "par.kg"), "--"]
        cwd = ROOT

    def run(path):
        result = subprocess.run([*command, str(path)], cwd=cwd, capture_output=True, timeout=20)
        assert result.returncode == 0, result.stderr.decode("utf-8")
        return result.stdout.decode("utf-8")

    return run


def test_weather_statistics_across_file(run_example, tmp_path):
    rows = [
        ("São Paulo", "99.9"), ('"Quoted"', "-8.1"), ("東京", "25.1"),
        ("São Paulo", "-99.9"), ("東京", "26.2"), ("São Paulo", "-8.1"),
        ("São Paulo", "1.2"), ("\ue000", "4.2"), ("\U00010000", "4.3"),
    ] * 512
    rows.extend([
        ("Positive tie", "0.0"), ("Positive tie", "0.1"),
        ("Zero tie", "0.0"), ("Zero tie", "-0.1"),
        ("Negative tie", "-0.1"), ("Negative tie", "-0.2"),
        ("Tail only", "7.0"),
    ])
    path = tmp_path / "station's weather.txt"
    path.write_text("\n".join(f"{name};{temperature}" for name, temperature in rows), encoding="utf-8")
    assert run_example(path) == reference(rows)


def test_utf8_crosses_scan_buffer(run_example, tmp_path):
    path = tmp_path / "measurements.txt"
    first = "東京;1.0\n".encode("utf-8")
    padding = b"Padding;0.0\n"
    # DuckDB's default 32,000,000-byte CSV buffer ends inside the first character of a station.
    offset = 32_000_000 - 1
    copies, extra = divmod(offset - len(first) - len(padding), len(padding))
    long_name = "Padding" + "x" * extra
    with path.open("wb") as output:
        output.write(first)
        output.write(padding * copies)
        output.write(f"{long_name};0.0\n".encode("ascii"))
        assert output.tell() == offset
        output.write("東京;2.1\nTail;7.0".encode("utf-8"))
    assert run_example(path) == reference([
        ("東京", "1.0"), ("東京", "2.1"), ("Padding", "0.0"), (long_name, "0.0"), ("Tail", "7.0"),
    ])


@pytest.mark.parametrize("rows", [
    [("\ufeffName", "1.0"), ("Name", "2.0"), ("N\0ame", "3.0"), (" ", "4.0")],
    [("\rName", "1.0"), ("Name", "2.0")],
    [("Name", "1.0"), ("Na\rme", "-2.0"), ("Name\r", "3.0")],
    [],
])
def test_station_text(run_example, tmp_path, rows):
    path = tmp_path / "measurements.txt"
    path.write_bytes("".join(f"{name};{temperature}\n" for name, temperature in rows).encode("utf-8"))
    assert run_example(path) == reference(rows)
