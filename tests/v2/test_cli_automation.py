"""Command-line contracts for build tools and coding agents running without a terminal."""
import json
import subprocess

import pytest

from .runtime import environment


def invoke(command, args, cwd):
    return subprocess.run([*command, *args], cwd=cwd, stdin=subprocess.DEVNULL,
                          env=environment(), capture_output=True, text=True, timeout=10)


def test_discover_capabilities_without_a_repl(kg, tmp_path):
    help_text = invoke(kg, ["--help"], tmp_path)
    assert help_text.returncode == 0 and help_text.stderr == ""
    for command in ("--capabilities", "lint", "ast", "check"):
        assert command in help_text.stdout
    result = invoke(kg, ["--capabilities"], tmp_path)
    assert result.returncode == 0 and result.stderr == ""
    info = json.loads(result.stdout)
    assert info["platform"] == "host"
    assert info["int"] in (32, 64) and info["float"] in (32, 64)
    assert isinstance(info["features"], list)
    assert invoke(kg, ["--version"], tmp_path).stdout.strip() == "kg " + info["version"]
    script = tmp_path / "inspect.kg"
    script.write_text(".p(.json(.ver()))\n")
    assert json.loads(invoke(kg, [str(script)], tmp_path).stdout) == info


def test_file_arguments_arrive_verbatim(kg, tmp_path):
    script = tmp_path / "arguments with spaces.kg"
    script.write_text(".p(.json(.a))\n")
    args = ["two words", "--stats", "-e", "literal", "", "--capabilities"]
    result = invoke(kg, [str(script), *args], tmp_path)
    assert result.returncode == 0 and result.stderr == ""
    assert json.loads(result.stdout) == args


@pytest.mark.parametrize("tool,output", [
    ("lint", "0 errors, 0 warnings in 1 file"),
    ("ast", '(call (sym .l) (str "./support.kg"))'),
], ids=["lint", "ast"])
def test_inspection_leaves_imported_file_unevaluated(host, tool, output, tmp_path):
    project = tmp_path / "project files"
    project.mkdir()
    (project / "support.kg").write_text('.tc(.oc("ran.marker"));.p("loaded");.cc(.tc(0))\n')
    script = project / "generated.kg"
    script.write_text('.l("./support.kg")\n.p("done")\n')
    marker = tmp_path / "ran.marker"

    inspect = invoke(host, [tool, str(script)], tmp_path)
    assert inspect.returncode == 0 and inspect.stderr == ""
    assert output in inspect.stdout.splitlines()
    assert not marker.exists()

    run = invoke(host, [str(script)], tmp_path)
    assert run.returncode == 0 and run.stderr == ""
    assert run.stdout.splitlines() == ["done"]
    assert marker.read_text().splitlines() == ["loaded"]


def test_error_is_located_and_failed_command_has_no_result(kg, tmp_path):
    script = tmp_path / "broken.kg"
    script.write_text("n::1\nnoSuchFunction(n)\n")
    result = invoke(kg, [str(script)], tmp_path)
    assert result.returncode == 1 and result.stdout == ""
    assert "broken.kg:2:" in result.stderr and "name error" in result.stderr
    bad_option = invoke(kg, ["--unknown-option"], tmp_path)
    assert bad_option.returncode == 2 and bad_option.stdout == ""
    assert "usage: kg" in bad_option.stderr
