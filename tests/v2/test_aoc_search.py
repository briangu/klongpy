"""Regression cases for the native AoC search and graph solvers."""
import pytest

from examples.aoc.benchmark import Case, measure


def check(kg, directory, year, day, part, lines, answer):
    path = directory / "input.txt"
    path.write_text("\n".join(lines) + "\n")
    measure(Case(year, day, part, path, answer, "edge"), kg[0], 0)


@pytest.mark.parametrize("shapes,region,answer", [
    # Oppositely handed tetrominoes need a reflection to tile the rectangle.
    (["###/#../...", "###/..#/..."], "2x4: 1 1", 1),
    # Equal occupied and available areas still leave incompatible corners.
    (["##./.##/..."], "2x4: 2", 0),
    # Two L pieces can interlock inside a single 3x3 bounding box.
    (["###/#../..."], "3x3: 2", 1),
    # Long strips exercise the search without relying on interpreter recursion.
    (["###/.../..."], "1x600: 200", 1),
])
def test_packing(kg, tmp_path, shapes, region, answer):
    lines = []
    for index, shape in enumerate(shapes):
        lines.extend([f"{index}:", *shape.split("/"), ""])
    check(kg, tmp_path, 2025, 12, 1, [*lines, region], answer)


def test_concave_polygon(kg, tmp_path):
    # The inward notch rules out the bounding rectangle. Large coordinates
    # also make allocating a grid up to the largest coordinate impractical.
    corners = [(0, 0), (8, 0), (8, 7), (5, 7), (5, 2), (3, 2), (3, 7), (0, 7)]
    lines = [f"{10**9 + x},{10**9 + y}" for x, y in corners]
    check(kg, tmp_path, 2025, 9, 2, lines, 32)


def test_joltage_minimum(kg, tmp_path):
    # Use the shared button three times and the first counter's button twice.
    check(kg, tmp_path, 2025, 10, 2, ["[##] (0,1) (0) (1) {5,3}"], 5)


def test_reversed_device_order_and_dead_end(kg, tmp_path):
    lines = ["svr: dac", "dac: a0", *[f"a{i}: a{i+1}" for i in range(300)],
             "a300: fft", "fft: out other", "other: dead"]
    check(kg, tmp_path, 2025, 11, 2, lines, 1)


def adder(width, exchanges):
    # All x bits are set and only y00 is set. Addition propagates the carry
    # from the least significant bit through every stage.
    inputs = [f"{prefix}{i:02}: {int(prefix == 'x' or i == 0)}"
              for prefix in "xy" for i in range(width)]
    gates = [("x00 XOR y00", "z00"), ("x00 AND y00", "c01")]
    for i in range(1, width):
        x, y, p, g, c, t, z = (f"{name}{i:02}" for name in "xypgctz")
        carry = f"c{i+1:02}" if i + 1 < width else f"z{width:02}"
        gates.extend([(f"{y} XOR {x}", p), (f"{x} AND {y}", g),
                      (f"{c} XOR {p}", z), (f"{p} AND {c}", t),
                      (f"{t} OR {g}", carry)])
    rename = {wire: other for a, b in exchanges for wire, other in ((a, b), (b, a))}
    return [*inputs, "", *[f"{expr} -> {rename.get(out, out)}" for expr, out in reversed(gates)]]


def test_long_carry_and_four_distant_swaps(kg, tmp_path):
    width = 45
    check(kg, tmp_path, 2024, 24, 1, adder(width, []), 1 << width)
    swaps = [("z00", "g04"), ("p31", "z12"), ("z45", "c17"), ("t32", "z44")]
    check(kg, tmp_path, 2024, 24, 2, adder(width, swaps),
          ",".join(sorted(wire for pair in swaps for wire in pair)))
