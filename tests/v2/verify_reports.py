"""Check that JUnit reports cover every test collected for the v2 acceptance suite."""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import pytest


def check_reports(expected, reports, source_sha256=None):
    observed = set()
    problems = []
    for path in reports:
        root = ET.parse(path).getroot()
        for suite in root.iter("testsuite"):
            if source_sha256 is not None:
                prop = suite.find("./properties/property[@name='source_archive_sha256']")
                if prop is None or prop.get("value") != source_sha256:
                    problems.append(f"{path}: missing or different source archive digest")
            for name in ("errors", "failures", "skipped"):
                count = suite.get(name, "0")
                if count != "0":
                    problems.append(f"{path}: {name}={count}")
        cases = list(root.iter("testcase"))
        if not cases:
            problems.append(f"{path}: no test cases")
        for case in cases:
            key = case.get("classname", ""), case.get("name", "")
            observed.add(key)
            for child in case:
                if child.tag in {"error", "failure", "skipped"}:
                    problems.append(f"{path}: {child.tag}: {'::'.join(key)}")
    for key in sorted(expected - observed):
        problems.append(f"missing: {'::'.join(key)}")
    for key in sorted(observed - expected):
        problems.append(f"unexpected: {'::'.join(key)}")
    return problems


class CollectedCases:
    def __init__(self):
        self.keys = set()

    def pytest_collection_finish(self, session):
        for item in session.items:
            owner = item.module.__name__
            if item.cls is not None:
                owner += "." + item.cls.__qualname__
            self.keys.add((owner, item.name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="require the SHA-256 of this source archive in every suite")
    parser.add_argument("reports", type=Path, nargs="+", help="JUnit XML from the v2 selections")
    args = parser.parse_args()
    cases = CollectedCases()
    with redirect_stdout(io.StringIO()) as output:
        status = pytest.main(["--collect-only", "-q", str(Path(__file__).resolve().parent)], plugins=[cases])
    if status or not cases.keys:
        sys.stderr.write(output.getvalue())
        return 1
    try:
        digest = None
        if args.source:
            with args.source.open("rb") as archive:
                digest = hashlib.file_digest(archive, "sha256").hexdigest()
        problems = check_reports(cases.keys, args.reports, digest)
    except (OSError, ET.ParseError) as error:
        parser.exit(1, f"Cannot read test report: {error}\n")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"All {len(cases.keys)} collected v2 acceptance cases passed across {len(args.reports)} reports.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
