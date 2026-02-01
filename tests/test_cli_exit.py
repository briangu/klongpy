import os
import subprocess
import sys
import tempfile
import unittest


class TestCliExit(unittest.TestCase):
    def test_exit_from_file(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        with tempfile.TemporaryDirectory() as tmp:
            exit_path = os.path.join(tmp, "exit.kg")
            with open(exit_path, "w", encoding="utf-8") as f:
                f.write(".x(0)\n")

            env = os.environ.copy()
            stub_dir = None
            try:
                import colorama  # noqa: F401
            except ModuleNotFoundError:
                stub_dir = os.path.join(tmp, "colorama_stub")
                os.makedirs(stub_dir, exist_ok=True)
                with open(os.path.join(stub_dir, "colorama.py"), "w", encoding="utf-8") as stub:
                    stub.write(
                        "class _Fore:\n"
                        "    BLACK=RED=GREEN=YELLOW=BLUE=MAGENTA=CYAN=WHITE=RESET=''\n"
                        "Fore=_Fore()\n"
                        "def init(*args, **kwargs):\n"
                        "    return None\n"
                    )

            pythonpath = [p for p in [stub_dir, repo_root, env.get("PYTHONPATH")] if p]
            env["PYTHONPATH"] = os.pathsep.join(pythonpath)
            env["PYTHONPYCACHEPREFIX"] = os.path.join(tmp, "pycache")

            result = subprocess.run(
                [sys.executable, "-m", "klongpy.cli", "-d", exit_path],
                cwd=repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )

        self.assertEqual(result.returncode, 0, msg=f"stderr:\n{result.stderr}")
        self.assertNotIn("Traceback", result.stderr)
