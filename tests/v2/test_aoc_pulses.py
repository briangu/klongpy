"""Pulse ordering, persistent conjunction memory and unreachable receivers."""
import subprocess

import pytest

from examples.aoc.benchmark import Case, measure


@pytest.mark.parametrize("lines,part,expected", [
    # Second public example from Advent of Code 2023, day 20.
    (["broadcaster -> a", "%a -> inv, con", "&inv -> b", "%b -> con",
      "&con -> output"], 1, 11687500),
    # The gate remembers c's high from press 2 when a sends high on press 3.
    (["broadcaster -> a, b", "%a -> gate", "%b -> c", "%c -> gate",
      "&gate -> rx"], 2, 3),
    # Both branches share a flip-flop, so their states must be traced together.
    (["broadcaster -> a", "%a -> b, gate", "%b -> gate", "&gate -> rx"], 2, 3),
    # The conjunction emits twice in one press. Its first pulse opens the gate.
    (["broadcaster -> a, b", "%a -> c", "%b -> c", "&c -> gate",
      "&gate -> rx"], 2, 1),
    (["broadcaster -> a", "%a -> rx"], 2, 2),
])
def test_fifo_and_memory(kg, tmp_path, lines, part, expected):
    path = tmp_path / "network.txt"
    path.write_text("\n".join(lines) + "\n")
    measure(Case(2023, 20, part, path, expected, "pulse"), kg[0], 0)


def test_unreachable_receiver(kg, tmp_path):
    # A direct low from the broadcaster leaves this gate permanently closed.
    path = tmp_path / "network.txt"
    path.write_text("broadcaster -> a, gate\n%a -> gate\n&gate -> rx\n")
    case = Case(2023, 20, 2, path, None, "pulse")
    result = subprocess.run(case.command(kg[0], 0), capture_output=True,
                            text=True, timeout=10)
    assert result.returncode
    assert "rx never receives low" in result.stderr
