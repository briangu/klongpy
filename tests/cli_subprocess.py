"""Launch the real CLI with the test provider registered when requested."""
import runpy
import sys

from klongpy.backends.registry import register_backend
from tests.custom_backend import TestBackendProvider


def cli_command(backend="numpy", device=None):
    module = "tests.cli_subprocess" if backend == "test_backend" else "klongpy.cli"
    command = [sys.executable, "-W", "ignore", "-m", module]
    if backend != "numpy":
        command += ["--backend", backend]
        if device:
            command += ["--device", device]
    return command


if __name__ == "__main__":
    register_backend("test_backend", TestBackendProvider)
    runpy.run_module("klongpy.cli", run_name="__main__", alter_sys=True)
