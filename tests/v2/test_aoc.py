"""Run the puzzle interfaces on both desktops and the generated benchmark workloads on kg."""
import json
import re
import subprocess

import numpy as np
import pytest

from examples.aoc.benchmark import ROOT, Case, measure, sample_cases, scale_cases
from klongpy import KlongInterpreter
from .runtime import PROGRAMS, environment


@pytest.mark.parametrize("program,expected", [
    ("aoc_bitxor.kg", "PASS bitxor"),
    ("aoc_collections.kg", "PASS aoc collections"),
    ("aoc_exact.kg", "PASS exact integers"),
])
def test_aoc_primitives(host, program, expected):
    proc = subprocess.run([*host, "--strict", "2", PROGRAMS / program],
                          env=environment(), capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == expected


def test_bitxor_python_array():
    kg = KlongInterpreter()
    values = np.arange(-256, 256, dtype=np.int64).reshape(32, 16)
    kg["values"] = values
    np.testing.assert_array_equal(kg(".bitxor(values;3)"), np.bitwise_xor(values, 3))


@pytest.mark.parametrize("case", list(sample_cases()), ids=lambda case: case.key)
def test_aoc_sample(host, kgpy, case):
    # Both hosts accept file arguments. The hosted CLI uses -- before them.
    args = case.command(host[0], 0)[1:]
    if host == kgpy:
        args.insert(3, "--")
    proc = subprocess.run([*host, *args], env=environment(), capture_output=True,
                          text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["answer"] == case.answer


def test_aoc_scale(kg, tmp_path):
    for case in scale_cases(kg[0], tmp_path):
        measure(case, kg[0], 0)


@pytest.mark.parametrize("year,day,part,lines,answer", [
    (2022, 5, 1, ["[A]", "[B]", " 1", "", "move 2 from 1 to 1"], "A"),
    (2022, 7, 1, ["$ cd /", "$ ls", "100000 f", "$ ls", "100000 f"], 100000),
    (2022, 8, 1, ["123", "456"], 6),
    (2022, 13, 1, ["[]", "0", "", "[1,2]", "[1,3]"], 3),
    (2023, 1, 2, ["oneight", "twone"], 39),
    (2023, 3, 2, ["1.2", ".*.", ".3."], 0),
    (2023, 4, 2, ["Card 1: 1 2 | 1 2", "Card 2: 3 | 3"], 3),
    (2023, 5, 2, ["seeds: 10 8", "", "seed-to-location map:", "40 15 2", "50 10 3"], 13),
    (2024, 8, 2, ["A.A.."], 5),
    (2024, 17, 1, ["Register A: 0", "Register B: 5", "Register C: 9", "",
                   "Program: 1,7,4,0,5,5"], "3"),
    (2024, 9, 2, ["12023"], 12),
    (2024, 13, 1, ["Button A: X+0, Y+36", "Button B: X+0, Y+44", "Prize: X=0, Y=60"], 0),
    (2024, 13, 1, ["Button A: X+2, Y+2", "Button B: X+1, Y+1", "Prize: X=4, Y=4"], 4),
])
def test_aoc_boundaries(kg, tmp_path, year, day, part, lines, answer):
    path = tmp_path / "input.txt"
    path.write_text("\n".join(lines))
    measure(Case(year, day, part, path, answer, "edge"), kg[0], 0)


@pytest.mark.parametrize("part,answer", [(1, 44276447), (2, 45)])
def test_market_prunes_to_24_bits(kg, tmp_path, part, answer):
    # Checked independently with integer XOR and a 0xffffff mask at each mixing step.
    path = tmp_path / "seeds.txt"
    seeds = [0, 1, 123, 2**24 - 1, 2**24, 2**32 + 123, 2**63 - 1]
    path.write_text("\n".join(map(str, seeds)))
    measure(Case(2024, 22, part, path, answer, "edge"), kg[0], 0)


def test_all_parts_have_a_sample_and_source_record():
    expected = {f"{year}/{day:02}/{part}"
                for year, last_day in [(2022, 25), (2023, 25), (2024, 25), (2025, 12)]
                for day in range(1, last_day + 1)
                for part in range(1, 2 if day == last_day else 3)}
    cases = list(sample_cases())
    assert len(cases) == len(expected)
    assert {case.key.removeprefix("sample/") for case in cases} == expected
    sources = (ROOT / "SOURCES.md").read_text()
    rows = re.findall(r"^\| \[(\d{4}/\d{2}/[12])\]\(([^)]+)\) \| (.+)$", sources, re.M)
    assert len(rows) == len(expected)
    assert {key for key, _, _ in rows} == expected
    for _, module, columns in rows:
        assert (ROOT / module).is_file()
        assert len(columns.rstrip(" |").split(" | ")) == 4
    for target in re.findall(r"\]\(([^)]+)\)", sources):
        if not target.startswith("https://"):
            assert (ROOT / target.split("#", 1)[0]).is_file(), target
    definitions = dict(re.findall(r"^\[([^\]]+)\]: (\S+)$", sources, re.M))
    for ref in re.findall(r"\]\[([^\]]+)\]", sources):
        assert ref in definitions
        if definitions[ref].startswith("https://github.com/"):
            assert re.search(r"/blob/[0-9a-f]{40}/", definitions[ref])


@pytest.mark.parametrize("offset", [0, 10**16, 2**62],
                         ids=["puzzle-scale", "past-float-precision", "sum-exceeds-signed-word"])
def test_hail_intersections_at_puzzle_scale(kg, tmp_path, offset):
    position = tuple(p + offset for p in (300000000000002, 200000000000005, 400000000000007))
    velocity = (32, -71, 15)
    encounters = [(11, (8, 5, -21)), (27, (-13, 32, 44)), (123, (33, 21, 14)),
                  (125, (19, -74, 8)), (178, (27, 15, 7)), (381, (0, -43, 0))]
    lines = []
    for t, hail_velocity in encounters:
        origin = [p + t * (v - h) for p, v, h in zip(position, velocity, hail_velocity)]
        lines.append(", ".join(map(str, origin)) + " @ " + ", ".join(map(str, hail_velocity)))
    path = tmp_path / "hail.txt"
    path.write_text("\n".join(lines))
    case = Case(2023, 24, 2, path, sum(position), "edge")
    measure(case, kg[0], 0)
    path.write_text("\n".join(lines + ["0, 0, 0 @ 0, 0, 0"]))
    result = subprocess.run(case.command(kg[0], 0), capture_output=True, text=True, timeout=5)
    assert result.returncode and "trajectory misses a hailstone" in result.stderr


def test_unreachable_blizzard_basin(kg, tmp_path):
    path = tmp_path / "valley.txt"
    path.write_text("#.#\n#^#\n#.#\n")
    case = Case(2022, 24, 1, path, None, "edge")
    result = subprocess.run(case.command(kg[0], 0), capture_output=True, text=True, timeout=5)
    assert result.returncode and "no route through valley" in result.stderr
