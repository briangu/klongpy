"""Verify each original wheel's full package suite and its two completed feed workloads."""
import hashlib
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def wheel_reports(directory, *, accept_external_feed):
    package = directory / "package-tests.xml"
    feed = directory / "feed-latency.xml"
    suites = list(ET.parse(package).iter("testsuite"))
    if not suites or any(int(s.get("tests", 0)) == 0 or s.get("failures") != "0"
                         or s.get("errors") != "0" for s in suites):
        raise ValueError(f"Package tests failed or were absent in {package}")
    tested = sum(int(s.get("tests")) for s in suites)
    suite, = ET.parse(feed).iter("testsuite")
    if (suite.get("tests"), suite.get("skipped"), suite.get("errors")) != ("2", "0", "0"):
        raise ValueError(f"Both feed workloads must complete in {feed}")
    reports = {}
    for case in suite.findall("testcase"):
        for host in ("kg", "kgpy"):
            if case.get("name") == f"test_tick_service_under_load[{host}]":
                break
        else:
            raise ValueError(f"Unexpected feed case: {case.get('name')}")
        prop = case.find(f"./properties/property[@name='{host}']")
        if prop is None or host in reports:
            raise ValueError(f"Missing or duplicate correctness-checked {host} result")
        raw = json.loads(prop.attrib["value"])
        times = {"max_ingest_lag_ms": raw["max_ingest_lag_ms"], "latest_p99_ms": raw["p99_ms"]["latest"],
                 "asof_p99_ms": raw["p99_ms"]["asofbid"]}
        if not all(math.isfinite(ms) and ms >= 0 for ms in times.values()):
            raise ValueError(f"Invalid measured {host} timing")
        failure = case.find("failure")
        if max(times.values()) >= 100:
            if not accept_external_feed or failure is None:
                raise ValueError(f"Runner exceeded feed bounds for {host}: {times}")
            assertions = ("assert actual.max_ingest_lag_ms < 100", "assert max(actual.p99_ms.values()) < 100")
            if not any(assertion in (failure.text or "") for assertion in assertions):
                raise ValueError(f"Unexpected {host} feed failure")
        elif failure is not None:
            raise ValueError(f"Unexpected {host} feed failure below bounds")
        reports[host] = times
    if set(reports) != {"kg", "kgpy"}:
        raise ValueError("Incomplete feed report")
    return {"package_junit_sha256": file_sha(package), "feed_junit_sha256": file_sha(feed),
            "package_tests_including_subtests": tested, "github_feed_ms": reports}
