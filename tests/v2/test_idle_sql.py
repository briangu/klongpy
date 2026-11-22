"""Standalone idle SQL publication and its foreground fallback."""
import re
import subprocess
import time

import numpy as np
import pytest

from .peer import connect, server
from .runtime import environment, load


ALIASES = [f"alias{i}" for i in range(48)]
IDLE_PAGE = re.compile(r"sql-idle rows (\d+)\.\.(\d+)(?: charged (\d+))?")
IDLE_WINDOW = "idle-window-complete"


def wait_for_log(path, process, ready):
    deadline = time.monotonic() + 5
    while True:
        output = path.read_text(encoding="utf-8", errors="replace")
        assert process.poll() is None, output
        if ready(output):
            return output
        assert time.monotonic() < deadline, output
        time.sleep(0.01)


def test_idle_sql_publishes_one_alias_and_foreground_reads_all(kg, tmp_path, monkeypatch):
    monkeypatch.setenv("KLONG_SQL_IDLE", "1")
    monkeypatch.setenv("KLONG_SQL_IDLE_TRACE", "1")
    rows = 32768
    log = tmp_path / "server.log"
    with server(kg, "kg", tmp_path) as (port, process), connect(port) as writer, connect(port) as reader:
        assert writer.request(load("idle_sql.kg") + ";1") == 1
        assert writer.call("makeViews", ALIASES) == 1
        assert writer.call("fill", rows) == 1
        # Let a full prefix become visible to the idle publisher. Then wait
        # through a timer window without running any foreground SQL.
        wait_for_log(log, process, lambda output: any(int(end) == rows for _, end, _ in IDLE_PAGE.findall(output)))
        assert writer.request("observeIdle()") == 1
        output = wait_for_log(log, process, lambda output: IDLE_WINDOW in output)
        pages = IDLE_PAGE.findall(output.split(IDLE_WINDOW, 1)[0])
        cursor = 16
        charge = 0
        for start, end, charged in pages:
            assert int(start) == cursor, pages
            assert int(end) > cursor, pages
            assert charged and charge < int(charged) <= 8 * 1024 * 1024, pages
            cursor, charge = int(end), int(charged)
        assert cursor == rows

        # Independent arithmetic oracle. The unpublished alias must see the
        # same full current table, including rows beyond the last idle page.
        count = rows + 16
        expected = [count, count * (count - 1) // 2]
        np.testing.assert_array_equal(reader.request("readLast()"), expected)
        np.testing.assert_array_equal(reader.request("readFirst()"), expected)


@pytest.mark.parametrize("options", [("--heap-limit", "2M"), ("--fail-every", "10000000")])
def test_bounded_cli_sessions_skip_idle_sql(kg, tmp_path, options):
    # Use the REPL because the file runner resets allocator settings before
    # entering its server loop. Do not queue EOF or exit before the timer.
    log = tmp_path / "repl.log"
    names = " ".join(f'"{name}"' for name in ALIASES)
    with log.open("wb") as terminal:
        process = subprocess.Popen([*kg, *options], cwd=tmp_path, stdin=subprocess.PIPE,
                                   stdout=terminal, stderr=subprocess.STDOUT,
                                   env=environment(KLONG_SQL_IDLE="1", KLONG_SQL_IDLE_TRACE="1"))
        try:
            source = f'{load("idle_sql.kg")};1\nmakeViews([{names}]);fill(4096);observeIdle()\n'
            process.stdin.write(source.encode())
            process.stdin.flush()
            output = wait_for_log(log, process, lambda output: IDLE_WINDOW in output)
            assert not IDLE_PAGE.search(output), output
            process.communicate(b'.p("bounded-repl-alive");1\n]q\n', timeout=5)
            output = log.read_text(encoding="utf-8", errors="replace")
            assert process.returncode == 0 and "bounded-repl-alive" in output, output
            assert not IDLE_PAGE.search(output), output
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
