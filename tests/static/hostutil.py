"""Helpers for static-profile tests that run a program on the host under whatever backend the session selected."""
import os


def session_backend():
    return os.environ.get("_KLONGPY_TEST_BACKEND", "numpy")


def host_rtol():
    """Relative tolerance for numbers computed by a program on the host: the torch backend computes in float32."""
    return 0.0 if session_backend() == "numpy" else 2e-6


def to_np(klong, v):
    """A value of the interpreter as a numpy value (what an entry or extern boundary hands the host)."""
    return klong._backend.to_numpy(v)
