import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import pytest
from klongpy.native_ext import native
from .runtime import required
from .peer import connect, server


ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@pytest.fixture(scope="session")
def kg(record_testsuite_property):
    candidate = ROOT / "native" / "target" / "release" / ("kg.exe" if os.name == "nt" else "kg")
    path = os.environ.get("KLONG_NATIVE") or (str(candidate) if candidate.is_file() else shutil.which("kg"))
    if not path or not Path(path).is_file():
        required("Build desktop kg or set KLONG_NATIVE to the candidate binary")
    record_testsuite_property("kg_command_sha256", file_sha256(path))
    return [str(Path(path).resolve())]


@pytest.fixture(scope="session")
def kgpy(record_testsuite_property):
    if path := os.environ.get("KLONG_V2_KGPY"):
        assert Path(path).is_file(), f"Missing kgpy command: {path}"
        record_testsuite_property("kgpy_launcher_sha256", file_sha256(path))
        return [str(Path(path).resolve())]
    return [sys.executable, "-m", "klongpy.cli"]


@pytest.fixture(params=["kg", "kgpy"])
def host(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def q(tmp_path_factory, record_testsuite_property):
    path = os.environ.get("KLONG_V2_KDB") or os.environ.get("KLONG_TEST_PEACHQ")
    if not path:
        required("Set KLONG_V2_KDB to KX q or KLONG_TEST_PEACHQ to PeachQ for the reference tests")
    assert Path(path).is_file(), f"Missing reference interpreter: {path}"
    command = [str(Path(path).resolve())]
    record_testsuite_property("q_command_sha256", file_sha256(path))
    with server(command, "q", tmp_path_factory.mktemp("reference-version")) as (port, _):
        with connect(port) as client:
            peach = client.request("count key `.pq") > 0
            engine = "PeachQ" if peach else "KX q"
            version = client.request(".pq.version" if peach else ".z.K")
            record_testsuite_property("q_reference", engine)
            record_testsuite_property("q_version", str(version))
            if not peach:
                record_testsuite_property("q_build_date_kdb_epoch", str(client.request(".z.k")))
    return command


@pytest.fixture(scope="session", autouse=True)
def identify_candidate(record_testsuite_property):
    record_testsuite_property("pytest_native_extension_sha256", file_sha256(native().__file__))
    if source := os.environ.get("KLONG_V2_SOURCE_ARCHIVE"):
        record_testsuite_property("source_archive_sha256", file_sha256(source))
    if shutil.which("git") and (ROOT / ".git").exists():
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True)
        record_testsuite_property("source_revision", revision.strip())
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        record_testsuite_property("working_tree_dirty", str(bool(status.strip())))
    record_testsuite_property("python_version", sys.version.split()[0])
    record_testsuite_property("platform", platform.platform())
    record_testsuite_property("machine", platform.node())


@pytest.fixture(scope="session")
def gpu_device(record_testsuite_property):
    device = os.environ.get("KLONG_V2_GPU_DEVICE")
    if not device:
        required("Set KLONG_V2_GPU_DEVICE to the hardware device on this runner")
    assert device.split(":")[0].lower() not in {"cpu", "auto"}, "Select a physical GPU"
    record_testsuite_property("gpu_device", device)
    return device
