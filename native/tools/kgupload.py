#!/usr/bin/env python3
"""
Talk to the console of a Klong board over a serial port: upload a file, run a
program, ask for the version. Python standard library only.

    kgupload.py PORT ver
    kgupload.py PORT upload FILE [REMOTE-PATH] [--hex | --raw]
    kgupload.py PORT run FILE            (or: run -e 'TEXT')
    kgupload.py PORT reset

PORT is a serial device (/dev/ttyUSB0, /dev/cu.usbserial-110) or the pseudo-terminal a
simulated board prints (`native/target/release/klong-sim --pty`). `upload` sends text with
`put` and anything that is not valid UTF-8 as Intel HEX (`--hex` and `--raw` force one or the
other); the remote path defaults to the file's name. `run` prints what the program writes and
then its value; a Klong error is `error: <kind>: <message>` on standard error and exit
status 1. A board that does not answer, or answers nonsense, is exit status 3.

Options: --baud N (a real serial port; a pseudo-terminal has no speed), --timeout SECONDS
(for one answer, default 10; for the result of `run`, --run-timeout, default 60), --retries N
(default 3), --chunk BYTES and --chunk-delay MS (pace raw data for a board whose UART has no
flow control), --record BYTES (data bytes per HEX record, default 16), --window N (HEX records
sent ahead of their acknowledgements, default 32), --board-timeout SECONDS (how long the board
waits before it abandons a transfer, default 5: the tool keeps quiet that long to get a stuck
board back), -v (show the conversation on standard error).

The protocol (docs/native/index.md, as built in native/klong-console): the byte 0x01 at
the start of an empty input line starts a command; every reply is one line that starts with
0x01.

    ^A ver                              ^A ok ver=... int=... float=... features=... heap_free=...
    ^A put <path> <length> <crc32-hex>  ^A go, raw bytes, ^A ok <length> <crc32> [warn ...]
    ^A run <length> <crc32-hex>         ^A go, raw bytes, output, the value, ^A ok value <n> | ^A ok | ^A err <kind>: <message>
    ^A hex <path>                       ^A go, records, ^A ack <n> | ^A nak <n> <reason> each, ^A ok <length> <crc32>

As a library: `Board(Port.open(path))`, then `.ver()`, `.put(data, remote)`, `.hex(data,
remote)`, `.run(text)`, `.reset()`; see native/scripts/console_conformance.py.
"""
import argparse
import os
import select
import sys
import time
import zlib

try:
    import termios
except ImportError:  # not a POSIX system
    termios = None

BOARD_TIMEOUT = 5.0  # seconds without input after which a board abandons a transfer (ConsoleConfig::timeout_us)


class BoardError(Exception):
    """The board answered `err <kind>: <message>`."""

    def __init__(self, kind, message, output=""):
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.message = message
        self.output = output


class ProtocolError(Exception):
    """No answer in time, or an answer that makes no sense."""


class Port:
    """A byte stream to a board: a serial device, a pseudo-terminal, or two file descriptors."""

    def __init__(self, rfd, wfd, close=True):
        self.rfd = rfd
        self.wfd = wfd
        self._close = close
        self.buf = bytearray()
        self._cr = False

    @classmethod
    def open(cls, path, baud=None):
        try:
            fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        except OSError as e:
            raise ProtocolError(f"cannot open {path}: {e.strerror}") from None
        if termios is not None and os.isatty(fd):
            a = termios.tcgetattr(fd)
            # raw: no echo, no line editing, no translation, 8 bits, no flow control
            a[0] = 0
            a[1] = 0
            a[2] = (a[2] & ~(termios.CSIZE | termios.PARENB | termios.CSTOPB)) | termios.CS8 | termios.CREAD | termios.CLOCAL
            a[3] = 0
            a[6][termios.VMIN] = 0
            a[6][termios.VTIME] = 0
            if baud:
                speed = getattr(termios, f"B{baud}", None)
                if speed is None:
                    os.close(fd)
                    raise ProtocolError(f"{baud} baud is not a speed this system knows")
                a[4] = a[5] = speed
            termios.tcsetattr(fd, termios.TCSANOW, a)
        return cls(fd, fd)

    def close(self):
        if self._close:
            for fd in {self.rfd, self.wfd}:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def _fill(self, timeout):
        """Read what is there (waiting at most `timeout` for the first byte) into the buffer. CR LF becomes LF."""
        r, _, _ = select.select([self.rfd], [], [], max(0.0, timeout))
        if not r:
            return False
        try:
            data = os.read(self.rfd, 65536)
        except BlockingIOError:
            return False
        except OSError as e:
            raise ProtocolError(f"the port failed: {e.strerror}") from None
        if not data:
            # a pipe that ended, or a pseudo-terminal whose other side is gone
            time.sleep(min(0.01, max(0.0, timeout)))
            return False
        if self._cr:
            data = b"\r" + data
            self._cr = False
        if data.endswith(b"\r"):
            data = data[:-1]
            self._cr = True
        self.buf += data.replace(b"\r\n", b"\n")
        return True

    def write(self, data):
        """Write everything, reading meanwhile so that neither side waits on a full buffer."""
        view = memoryview(bytes(data))
        deadline = time.monotonic() + 30.0
        while view:
            r, w, _ = select.select([self.rfd], [self.wfd], [], 1.0)
            if r:
                self._fill(0)
            if w:
                try:
                    n = os.write(self.wfd, view[:4096])
                except BlockingIOError:
                    n = 0
                except OSError as e:
                    raise ProtocolError(f"the port failed: {e.strerror}") from None
                view = view[n:]
                if n:
                    deadline = time.monotonic() + 30.0
            if time.monotonic() > deadline:
                raise ProtocolError("the port takes no more data")

    def drain(self, quiet=0.05):
        """Read until nothing has come for `quiet` seconds; return and forget everything read."""
        while self._fill(quiet):
            pass
        data = bytes(self.buf)
        self.buf.clear()
        return data


class Board:
    def __init__(self, port, timeout=10.0, run_timeout=60.0, retries=3, chunk=1024, chunk_delay=0.0, record=16, window=32, log=None,
                 board_timeout=BOARD_TIMEOUT):
        self.port = port
        self.timeout = timeout
        self.run_timeout = run_timeout
        self.retries = retries
        self.chunk = max(1, chunk)
        self.chunk_delay = chunk_delay
        self.record = max(1, min(255, record))
        self.window = max(1, window)
        self.log = log
        self.board_timeout = board_timeout

    def _note(self, text):
        if self.log:
            self.log.write(text + "\n")

    # ----- replies ----------------------------------------------------------------------------

    def _reply(self, timeout):
        """
        The next reply line: (bytes before it, word, rest of the line). The bytes before it are
        whatever the board printed that is not a reply: echo, a prompt, the output of a program.
        """
        deadline = time.monotonic() + timeout
        buf = self.port.buf
        while True:
            at = buf.find(b"\x01")
            if at >= 0:
                end = buf.find(b"\n", at)
                if end >= 0:
                    before = bytes(buf[:at])
                    line = bytes(buf[at + 1:end]).decode("utf-8", "replace").strip()
                    del buf[:end + 1]
                    word, _, rest = line.partition(" ")
                    if word != "ack":
                        self._note(f"<- {word} {rest}".rstrip())
                    return before, word, rest
            left = deadline - time.monotonic()
            if left <= 0:
                raise ProtocolError(f"no answer from the board within {timeout:g} s")
            self.port._fill(min(left, 0.25))

    def _expect(self, words, timeout):
        """A reply whose word is one of `words`; `err` raises BoardError; anything else is skipped as stale."""
        skipped = 0
        while True:
            before, word, rest = self._reply(timeout)
            if word in words:
                return before, word, rest
            if word == "err":
                kind, _, message = rest.partition(": ")
                raise BoardError(kind, message, before.decode("utf-8", "replace"))
            skipped += 1
            if skipped > 10000:
                raise ProtocolError(f"the board keeps answering {word!r}")

    def _command(self, line):
        self._note(f"-> {line}")
        self.port.write(b"\x01" + line.encode("utf-8") + b"\n")

    def _send(self, data):
        for i in range(0, len(data), self.chunk):
            self.port.write(data[i:i + self.chunk])
            if self.chunk_delay:
                time.sleep(self.chunk_delay)

    # ----- commands ---------------------------------------------------------------------------

    def sync(self):
        """
        Get the board to its prompt, whatever it was doing: Ctrl-C abandons a line, an
        evaluation or a HEX transfer; a raw transfer ends when the board has seen no byte for
        its timeout. Returns the answer of `ver`.
        """
        last = None
        for attempt in range(3):
            self.port.drain()
            self.port.write(b"\x03")
            self.port.drain(0.05)
            try:
                return self._ver(min(self.timeout, 2.0))
            except (ProtocolError, BoardError) as e:
                last = e
                self._note(f"no answer to ver ({e}); waiting for the board to give up what it is doing")
                # say nothing for longer than the board's timeout
                end = time.monotonic() + self.board_timeout * 1.2 + 0.1
                while time.monotonic() < end:
                    self.port._fill(0.25)
        raise ProtocolError(f"the board does not answer: {last}")

    def _ver(self, timeout):
        self._command("ver")
        while True:
            _, _, rest = self._expect(("ok",), timeout)
            if rest.startswith("ver="):
                return dict(item.partition("=")[::2] for item in rest.split())

    def ver(self):
        """The version line as a dictionary: ver, int, float, features, platform, fs, heap_free, ..."""
        return self._retrying(lambda: self._ver(self.timeout))

    def _retrying(self, action):
        problem = None
        for attempt in range(self.retries + 1):
            if attempt:
                self._note(f"retry {attempt} of {self.retries}: {problem}")
                self.sync()
            try:
                return action()
            except ProtocolError as e:
                problem = e
            except BoardError as e:
                # A damaged transfer is worth another try; a refusal is not. (This tool sends
                # no malformed command, so a command the board cannot read was damaged too.)
                damaged = e.kind == "system" and (e.message.startswith("check sum") or e.message == "transfer timed out")
                garbled = e.kind == "syntax" and e.message.startswith(("unknown command", "usage:", "empty command", "command longer", "the command is not"))
                if not (damaged or garbled):
                    raise
                problem = e
        raise ProtocolError(f"gave up after {self.retries + 1} attempts: {problem}")

    @staticmethod
    def _stored(rest, data):
        """Check the `ok <length> <crc32> [...]` of a stored file against what was sent; the rest of the line."""
        words = rest.split(" ", 2)
        try:
            length, crc = int(words[0]), int(words[1], 16)
        except (ValueError, IndexError):
            raise ProtocolError(f"unexpected answer: ok {rest}") from None
        if length != len(data) or crc != zlib.crc32(data):
            raise ProtocolError(f"the board stored {length} bytes with check sum {crc:08x}, not {len(data)} bytes with {zlib.crc32(data):08x}")
        return words[2] if len(words) > 2 else ""

    def put(self, data, remote):
        """Store bytes as a file with `put`. Returns the board's remark ("" or "warn syntax: ...")."""
        def once():
            self._command(f"put {remote} {len(data)} {zlib.crc32(data):08x}")
            self._expect(("go",), self.timeout)
            self._send(data)
            _, _, rest = self._expect(("ok",), self.timeout)
            return self._stored(rest, data)
        _check_path(remote)
        return self._retrying(once)

    def run(self, text):
        """
        Evaluate a program. Returns (value, output): the printed value of the last statement
        or None, and what the program wrote. A Klong error raises BoardError (with .output).
        """
        data = text.encode("utf-8") if isinstance(text, str) else bytes(text)

        def start():
            self._command(f"run {len(data)} {zlib.crc32(data):08x}")
            self._expect(("go",), self.timeout)
            self._send(data)
            # (a check sum error comes back at once and is retried; once the program runs,
            # nothing is sent twice)
            return self._expect(("ok",), self.run_timeout)
        before, _, rest = self._retrying(start)
        out = before.decode("utf-8", "replace")
        if not rest:
            return None, out
        words = rest.split()
        if len(words) != 2 or words[0] != "value" or not words[1].isdigit():
            raise ProtocolError(f"unexpected answer: ok {rest}")
        n = int(words[1])
        if len(before) < n + 1 or before[-1:] != b"\n":
            raise ProtocolError(f"the value of {n} bytes is not before its answer")
        value = before[len(before) - n - 1:-1].decode("utf-8", "replace")
        return value, before[:len(before) - n - 1].decode("utf-8", "replace")

    def hex(self, data, remote):
        """Store bytes as a file with Intel HEX records. Returns the board's remark."""
        _check_path(remote)
        records = [(at, data[at:at + self.record]) for at in range(0, len(data), self.record)]
        # a record must not cross a 64 KiB boundary
        split = []
        for at, chunk in records:
            room = 0x10000 - (at & 0xFFFF)
            if len(chunk) > room:
                split.append((at, chunk[:room]))
                split.append((at + room, chunk[room:]))
            else:
                split.append((at, chunk))
        records = split

        def once():
            self._command(f"hex {remote}")
            self._expect(("go",), self.timeout)
            todo = list(range(len(records)))
            rounds = 0
            while todo:
                rounds += 1
                if rounds > self.retries + 1:
                    self.port.write(b"\x03")
                    raise ProtocolError(f"{len(todo)} records are still refused after {rounds - 1} rounds")
                todo = self._hex_round(records, todo)
                if todo:
                    self._note(f"sending {len(todo)} records again")
            self.port.write(hex_record(1, 0, b"") + b"\n")
            _, _, rest = self._expect(("ok",), self.timeout)
            return self._stored(rest, data)
        return self._retrying(once)

    def _hex_round(self, records, todo):
        """
        Send the records `todo` (indices) without waiting for each answer, at most `window`
        ahead. Returns the indices to send again: those the board refused, or all of them when
        the answers cannot be matched to the lines (a line was lost) or an address record failed.
        """
        lines = []      # (record index or None for an address record, line)
        upper = None
        for i in todo:
            at, chunk = records[i]
            if at >> 16 != upper:
                upper = at >> 16
                lines.append((None, hex_record(4, 0, upper.to_bytes(2, "big"))))
            lines.append((i, hex_record(0, at & 0xFFFF, chunk)))
        answers = []    # (word, n, reason) in the order they came
        sent = 0
        first = None

        def collect(timeout):
            nonlocal first
            try:
                _, word, rest = self._expect(("ack", "nak"), timeout)
            except ProtocolError:
                return False
            n, _, reason = rest.partition(" ")
            if not n.isdigit():
                raise ProtocolError(f"unexpected answer: {word} {rest}")
            if first is None:
                first = int(n)
            answers.append((word, int(n), reason))
            return True

        while sent < len(lines):
            while sent - len(answers) >= self.window:
                if not collect(self.timeout):
                    break
            else:
                batch = lines[sent:sent + max(1, self.window - (sent - len(answers)))]
                self.port.write(b"".join(line + b"\n" for _, line in batch))
                sent += len(batch)
                if self.chunk_delay:
                    time.sleep(self.chunk_delay)
                continue
            break   # the board stopped answering: decide below
        while len(answers) < sent and collect(min(self.timeout, 2.0)):
            pass
        numbers = [n for _, n, _ in answers]
        if len(answers) != len(lines) or numbers != list(range(first or 0, (first or 0) + len(lines))):
            self._note(f"{len(lines)} lines sent, {len(answers)} answered: sending all records again")
            return list(range(len(records)))
        again = []
        for (index, _), (word, n, reason) in zip(lines, answers):
            if word == "nak":
                self._note(f"record {n} refused: {reason}")
                if index is None:
                    return list(range(len(records)))
                again.append(index)
        return again

    def reset(self):
        """`.x(0)`: a board resets and prints its banner again. Returns the answer of `ver` afterwards."""
        data = b".x(0)"
        self._command(f"run {len(data)} {zlib.crc32(data):08x}")
        self._expect(("go",), self.timeout)
        self._send(data)
        # the banner of the new life; what came before it (the answer of the old one, if it
        # got as far as answering) is not wanted
        deadline = time.monotonic() + self.timeout
        buf = self.port.buf
        while True:
            at = buf.find(b"Klong ")
            end = buf.find(b"\n", at) if at >= 0 else -1
            if end >= 0:
                del buf[:end + 1]
                break
            if time.monotonic() > deadline:
                raise ProtocolError("the board did not come back after .x(0)")
            self.port._fill(0.1)
        try:
            return self._ver(min(self.timeout, 2.0))
        except (ProtocolError, BoardError):
            return self.sync()


def _check_path(remote):
    if not remote or any(c.isspace() or ord(c) < 0x20 for c in remote) or len(remote.encode("utf-8")) > 200:
        raise BoardError("syntax", f"a remote path must be 1 to 200 bytes without blanks: {remote!r}")


def hex_record(kind, addr, data):
    """One Intel HEX record as bytes, without a line end."""
    body = bytes([len(data), (addr >> 8) & 0xFF, addr & 0xFF, kind]) + bytes(data)
    return b":" + (body + bytes([(-sum(body)) & 0xFF])).hex().upper().encode("ascii")


def hex_lines(data, record=16):
    """A whole Intel HEX file for `data` at address 0, as a list of lines (bytes)."""
    out = []
    upper = 0
    for at in range(0, len(data), record):
        chunk = data[at:at + record]
        room = 0x10000 - (at & 0xFFFF)
        for start, part in ((at, chunk[:room]), (at + room, chunk[room:])):
            if part:
                if start >> 16 != upper:
                    upper = start >> 16
                    out.append(hex_record(4, 0, upper.to_bytes(2, "big")))
                out.append(hex_record(0, start & 0xFFFF, part))
    out.append(hex_record(1, 0, b""))
    return out


def is_text(data):
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return b"\x00" not in data


def _common_options():
    """The options every command takes. A fresh parser at each call: the parent's actions are
    shared with the parsers that use it, and `set_defaults` changes the shared actions."""
    common = argparse.ArgumentParser(add_help=False)
    S = argparse.SUPPRESS
    common.add_argument("--baud", type=int, default=S)
    common.add_argument("--timeout", type=float, default=S)
    common.add_argument("--run-timeout", type=float, default=S)
    common.add_argument("--retries", type=int, default=S)
    common.add_argument("--chunk", type=int, default=S)
    common.add_argument("--chunk-delay", type=float, default=S, metavar="MS")
    common.add_argument("--record", type=int, default=S)
    common.add_argument("--window", type=int, default=S)
    common.add_argument("--board-timeout", type=float, default=S)
    common.add_argument("-v", "--verbose", action="store_true", default=S)
    return common


def build_parser():
    # The options are accepted before or after the subcommand: the same options are added to the main
    # parser and to every subcommand. They have no default of their own there (a
    # default in a subparser would overwrite a value given before the subcommand); the
    # defaults are set once, on the main parser.
    common = _common_options()
    ap = argparse.ArgumentParser(prog="kgupload.py", description="Upload to and run on a Klong board over its serial console.", parents=[common])
    ap.add_argument("port", help="serial device or pseudo-terminal")
    ap.set_defaults(baud=None, timeout=10.0, run_timeout=60.0, retries=3, chunk=1024, chunk_delay=0.0, record=16, window=32,
                    board_timeout=BOARD_TIMEOUT, verbose=False)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ver", help="version, widths, features, free heap", parents=[_common_options()])
    up = sub.add_parser("upload", help="store a file on the board", parents=[_common_options()])
    up.add_argument("file")
    up.add_argument("remote", nargs="?")
    how = up.add_mutually_exclusive_group()
    how.add_argument("--hex", action="store_true", help="send as Intel HEX (the default for files that are not UTF-8 text)")
    how.add_argument("--raw", action="store_true", help="send with put, whatever the file holds")
    run = sub.add_parser("run", help="evaluate a program on the board without storing it", parents=[_common_options()])
    run.add_argument("file", nargs="?")
    run.add_argument("-e", dest="expr", metavar="TEXT")
    sub.add_parser("reset", help="restart the board with .x(0)", parents=[_common_options()])
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.cmd == "run" and (args.file is None) == (args.expr is None):
        ap.error("run takes a FILE or -e TEXT")
    data = None
    if args.cmd in ("upload", "run") and getattr(args, "file", None):
        try:
            with open(args.file, "rb") as f:
                data = f.read()
        except OSError as e:
            print(f"kgupload: {args.file}: {e.strerror}", file=sys.stderr)
            return 2
    try:
        port = Port.open(args.port, args.baud)
    except ProtocolError as e:
        print(f"kgupload: {e}", file=sys.stderr)
        return 3
    board = Board(port, timeout=args.timeout, run_timeout=args.run_timeout, retries=args.retries, chunk=args.chunk,
                  chunk_delay=args.chunk_delay / 1000.0, record=args.record, window=args.window,
                  log=sys.stderr if args.verbose else None, board_timeout=args.board_timeout)
    try:
        info = board.sync()
        if args.cmd == "ver":
            print(" ".join(f"{k}={v}" for k, v in info.items()))
        elif args.cmd == "reset":
            info = board.reset()
            print(" ".join(f"{k}={v}" for k, v in info.items()))
        elif args.cmd == "upload":
            remote = args.remote or os.path.basename(args.file)
            as_hex = args.hex or (not args.raw and not is_text(data))
            start = time.monotonic()
            remark = board.hex(data, remote) if as_hex else board.put(data, remote)
            took = time.monotonic() - start
            print(f"{remote}: {len(data)} bytes, crc32 {zlib.crc32(data):08x}, {'hex' if as_hex else 'put'}, {took:.2f} s")
            if remark:
                print(f"kgupload: {remote}: {remark}", file=sys.stderr)
        elif args.cmd == "run":
            text = data if data is not None else args.expr.encode("utf-8")
            value, out = board.run(text)
            sys.stdout.write(out)
            if value is not None:
                sys.stdout.write(value + "\n")
        return 0
    except BoardError as e:
        if e.output:
            sys.stdout.write(e.output)
            sys.stdout.flush()
        print(f"error: {e.kind}: {e.message}", file=sys.stderr)
        return 1
    except ProtocolError as e:
        print(f"kgupload: {e}", file=sys.stderr)
        return 3
    finally:
        port.close()


if __name__ == "__main__":
    sys.exit(main())
