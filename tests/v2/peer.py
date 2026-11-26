"""Run candidate processes in private directories and exercise their public IPC endpoints."""
from contextlib import closing, contextmanager
from pathlib import Path
import socket
import subprocess
import time

from klongpy.core import KGSym
from klongpy.ipc import HandshakeAccepted, KdbError, Message, Session
from tests.ipc_helpers import free_port
from .runtime import environment


class Client:
    def __init__(self, sock, timeout=10, *, user="", password=""):
        self.sock = sock
        self.timeout = timeout
        self.session = Session.client(user, password)
        sock.sendall(self.session.hello())
        event = self.receive()
        assert isinstance(event, HandshakeAccepted), f"IPC handshake failed: {event}"

    def receive(self):
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            self.sock.settimeout(max(0.01, deadline - time.monotonic()))
            packet = self.sock.recv(65536)
            assert packet, "peer closed the connection before responding"
            events = self.session.feed(packet)
            if events:
                assert len(events) == 1, events
                return events[0]
        raise TimeoutError("IPC response deadline expired")

    def request(self, value, *, call=False):
        packet, pending = self.session.send_sync(value, call=call)
        self.sock.sendall(packet)
        event = self.receive()
        assert isinstance(event, Message) and event.pending is pending, event
        assert not event.error, event.error
        if isinstance(event.value, KdbError):
            operation = value[0] if call else value
            raise AssertionError(f"{operation!r}: {event.value.text}")
        return event.value

    def call(self, name, *args):
        return self.request([KGSym(name), *args], call=True)


@contextmanager
def connect(port, host="127.0.0.1", timeout=10, *, user="", password=""):
    with closing(socket.create_connection((host, port), timeout=5)) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        yield Client(sock, timeout, user=user, password=password)


@contextmanager
def server(command, kind, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    port = free_port()
    addr = f"127.0.0.1:{port}"
    if kind == "q":
        args = [*command, "-q", "-s", "0", "-p", addr]
    else:
        boot = directory / "server.kg"
        boot.write_text(f'.srv("{addr}")\n', encoding="utf-8")
        args = [*command, *(["-l"] if kind == "kgpy" else []), str(boot)]
    log_path = directory / "server.log"
    with log_path.open("wb") as log:
        proc = subprocess.Popen(args, cwd=directory, env=environment(), stdin=subprocess.PIPE,
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 8
            while proc.poll() is None and time.monotonic() < deadline:
                try:
                    with connect(port) as client:
                        assert client.request("1+1") == 2
                    break
                except ConnectionRefusedError:
                    time.sleep(0.025)
            else:
                raise AssertionError(f"{kind} failed to listen: {log_path.read_text(errors='replace')}")
            yield port, proc
        finally:
            proc.stdin.close()
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
