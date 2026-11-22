import os
from pathlib import Path
import re
import subprocess

import klongpy
import pytest


PROGRAMS = Path(__file__).with_name("programs")
ROOT = PROGRAMS.parents[2]


def environment(**extra):
    # Match the parent installation, including when pytest is exercising a wheel.
    env = os.environ.copy()
    if Path(klongpy.__file__).resolve().parent.parent == ROOT:
        paths = [str(ROOT), env.get("PYTHONPATH", "")]
        env["PYTHONPATH"] = os.pathsep.join(filter(None, paths))
    return {**env, **extra}


def required(message):
    if os.environ.get("KLONG_V2_RELEASE") == "1":
        pytest.fail(message, pytrace=False)
    pytest.skip(message)


def load(program):
    path = (PROGRAMS / program).as_posix()
    return '.l("' + path.replace('"', '""') + '")'


def evaluate(command, expression, cwd, *, options=(), extra_env=None, timeout=15):
    proc = subprocess.run([*command, *options, "-e", expression], cwd=cwd,
                          env=environment(**(extra_env or {})), text=True,
                          capture_output=True, timeout=timeout)
    assert proc.returncode == 0, f"{expression}\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    return proc.stdout.strip()


def case_names(program):
    """Discover named, zero-argument .kg acceptance cases, each independently callable."""
    names = re.findall(r"^(case[A-Z][A-Za-z0-9]*)::\{", (PROGRAMS / program).read_text(), re.M)
    if not names or len(names) != len(set(names)):
        raise ValueError(f"{program} needs unique case functions")
    return names
