"""Check pinned distribution files and the measured Apple Silicon timing gate."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile

from check_reports import wheel_reports


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    receipt_path, artifact_root, dist = map(Path, sys.argv[1:])
    receipt = json.loads(receipt_path.read_text())
    if dist.exists():
        raise ValueError(f"Destination already exists: {dist}")
    if receipt["version"] != "2.0.0rc1":
        raise ValueError("This receipt belongs to another version")
    files = {}
    for entry in receipt["distributions"]:
        if entry["artifact"] in files:
            raise ValueError("Duplicate distribution in receipt")
        file, = (artifact_root / entry["artifact"]).iterdir()
        if file.name != entry["filename"] or digest(file) != entry["sha256"]:
            raise ValueError(f"File differs from the qualified build: {file.name}")
        artifact = entry["artifact"]
        if artifact.startswith("wheel-"):
            report = artifact_root / artifact.replace("wheel-", "tests-", 1)
            evidence = wheel_reports(report, accept_external_feed=artifact == "wheel-macos-arm64")
            if evidence != entry["original_test_reports"]:
                raise ValueError(f"Original wheel test results differ for {artifact}")
        files[artifact] = file

    evidence = receipt["apple_silicon"]
    if evidence["source_archive_sha256"] != digest(files["sdist"]):
        raise ValueError("Reference host source does not match the upload")
    required_members = {"klongpy/_native.abi3.so", "klongpy-2.0.0rc1.data/scripts/kg"}
    if set(evidence["wheel_members"]) != required_members:
        raise ValueError("Reference host must identify the interpreter and the Python extension")
    with ZipFile(files["wheel-macos-arm64"]) as wheel:
        for filename, expected in evidence["wheel_members"].items():
            if hashlib.sha256(wheel.read(filename)).hexdigest() != expected:
                raise ValueError(f"Reference host binary does not match the wheel: {filename}")
    if set(evidence["feed_ms"]) != {"kg", "kgpy"}:
        raise ValueError("Both runtime measurements are required")
    for host, times in evidence["feed_ms"].items():
        if set(times) != {"max_ingest_lag_ms", "latest_p99_ms", "asof_p99_ms"} or not all(0 <= t < 100 for t in times.values()):
            raise ValueError(f"Reference host missed its 100 ms bounds: {host}, {times}")
    dist.mkdir()
    for file in files.values():
        shutil.copy2(file, dist / file.name)
        print(digest(file), file.name)


if __name__ == "__main__":
    main()
