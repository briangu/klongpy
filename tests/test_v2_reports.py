"""Regression checks for combining the release suite's distributed JUnit reports."""
from xml.etree import ElementTree as ET

import pytest

from tests.v2.verify_reports import check_reports


def report(path, cases):
    suite = ET.Element("testsuite")
    for name, outcome in cases:
        case = ET.SubElement(suite, "testcase", classname="tests.v2.test_example", name=name)
        if outcome:
            ET.SubElement(case, outcome)
    ET.ElementTree(suite).write(path)
    return path


def test_identifies_missing_case_when_report_count_is_unchanged(tmp_path):
    desktop = report(tmp_path / "desktop.xml", [("a", None), ("a", None)])
    expected = {("tests.v2.test_example", name) for name in ("a", "b")}
    assert "missing: tests.v2.test_example::b" in check_reports(expected, [desktop])


@pytest.mark.parametrize("outcome", ["skipped", "error", "failure"])
def test_repeated_success_preserves_unsuccessful_result(tmp_path, outcome):
    first = report(tmp_path / "cpu.xml", [("a", None)])
    second = report(tmp_path / "gpu.xml", [("a", outcome)])
    problems = check_reports({("tests.v2.test_example", "a")}, [first, second])
    assert any(outcome in problem and "gpu.xml" in problem for problem in problems)


def test_split_reports_and_successful_repeats(tmp_path):
    first = report(tmp_path / "first.xml", [("a[1.5]", None)])
    second = report(tmp_path / "second.xml", [("b", None), ("a[1.5]", None)])
    expected = {("tests.v2.test_example", name) for name in ("a[1.5]", "b")}
    assert check_reports(expected, [first, second]) == []


def test_suite_level_error_is_preserved(tmp_path):
    path = tmp_path / "interrupt.xml"
    suite = ET.Element("testsuite", errors="1")
    ET.SubElement(suite, "testcase", classname="tests.v2.test_example", name="a")
    ET.ElementTree(suite).write(path)
    problems = check_reports({("tests.v2.test_example", "a")}, [path])
    assert any("errors=1" in problem for problem in problems)


def test_source_option_detects_a_different_or_unidentified_candidate(tmp_path):
    path = report(tmp_path / "worker.xml", [("a", None)])
    expected = {("tests.v2.test_example", "a")}
    digest = "a" * 64
    assert check_reports(expected, [path], digest)
    tree = ET.parse(path)
    prop = ET.SubElement(ET.SubElement(tree.getroot(), "properties"), "property",
                         name="source_archive_sha256", value="b" * 64)
    tree.write(path)
    assert check_reports(expected, [path], digest)
    prop.set("value", digest)
    tree.write(path)
    assert check_reports(expected, [path], digest) == []
