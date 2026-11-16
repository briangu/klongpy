"""Ticket 040/105: IPC failures are KlongError classes.

A remote error arrives as the class of its wire name (so a range error on the
peer is a ``KlongRangeError`` here) that is also a ``KlongRemoteException``;
transport failures are ``KlongSystemError``; ``wire_error_text`` names an error
by its ``kind``.
"""
import unittest

from ipc_helpers import FakeKdbServer

from klongpy.errors import (KlongArityError, KlongError, KlongInternalError, KlongInterrupt,
                            KlongMemoryError, KlongNameError, KlongPythonError, KlongRangeError,
                            KlongRemoteError, KlongShapeError, KlongSyntaxError, KlongSystemError,
                            KlongTypeError)
from klongpy.ipc.endpoint import KlongRemoteException, wire_error_text
from klongpy.sys_fn_ipc import (KlongIPCConnectionClosedException,
                                KlongIPCConnectionFailureException,
                                KlongIPCCreateConnectionException, KlongIPCException,
                                KlongIPCHandshakeException, parse_address)

from test_ipc_endpoint import IPCCase


class TestRemoteErrors(IPCCase):
    def test_range_error_arrives_as_a_range_error(self):
        server, client = self.pair()
        self.kg(server, "get::{x@10}")
        with self.assertRaises(KlongRangeError) as cm:
            self.kg(client, "c(:get,,[1 2 3])")
        e = cm.exception
        self.assertEqual(e.kind, "range")
        self.assertIsInstance(e, KlongRemoteException)
        self.assertTrue(e.remote)
        self.assertEqual(e.wire_name, "index")
        self.assertIn(str(self.port), e.hint)                      # the peer is in the hint
        self.assertTrue(e.text.startswith("index: "))
        self.assertEqual(self.kg(client, 'c("1+1")'), 2)

    def test_each_wire_name_has_its_class(self):
        server, client = self.pair()
        self.kg(server, "boom::{x+\"a\"}")
        with self.assertRaises(KlongNameError) as cm:
            self.kg(client, "c(:nosuchfn,,1)")
        self.assertEqual(cm.exception.kind, "name")
        with self.assertRaises(KlongTypeError) as cm:
            self.kg(client, "c(:boom,,1)")
        self.assertEqual(cm.exception.kind, "type")
        self.assertNotIn("type:", str(cm.exception))                # the message, not the wire text
        with self.assertRaises(KlongSyntaxError) as cm:
            self.kg(client, 'c("1+")')
        self.assertEqual(cm.exception.wire_name, "parse")
        for e in (cm.exception,):
            self.assertIsInstance(e, KlongRemoteException)

    def test_classes_for_every_wire_name(self):
        for wire, cls in (("parse", KlongSyntaxError), ("name", KlongNameError), ("rank", KlongArityError),
                          ("type", KlongTypeError), ("length", KlongShapeError), ("index", KlongRangeError),
                          ("os", KlongSystemError), ("python", KlongPythonError),
                          ("internal", KlongInternalError), ("wsfull", KlongMemoryError),
                          ("stop", KlongInterrupt)):
            e = KlongRemoteException(f"{wire}: oops", ("h", 1))
            self.assertIsInstance(e, cls, wire)
            self.assertIsInstance(e, KlongRemoteException, wire)
            self.assertEqual(e.kind, cls.kind, wire)
            self.assertEqual(e.message, "oops")
            self.assertEqual(e.wire_name, wire)
            self.assertEqual(e.text, f"{wire}: oops")

    def test_other_text_is_a_remote_error(self):
        for text in ("'length", "free text", "bogus: not a wire name", ""):
            e = KlongRemoteException(text, "p")
            self.assertIsInstance(e, KlongRemoteError)
            self.assertIsInstance(e, KlongRemoteException)
            self.assertEqual(e.kind, "remote")
            self.assertIsNone(e.wire_name)
            self.assertEqual(str(e), text)

    def test_kdb_error_text_from_another_peer(self):
        def handler(value):
            raise RuntimeError("'length")
        fake = FakeKdbServer(handler)
        self.addCleanup(fake.stop)
        client = self.repl()
        self.kg(client, f"c::.cli({fake.port})")
        with self.assertRaises(KlongRemoteError) as cm:
            self.kg(client, 'c("1+1")')
        self.assertEqual(cm.exception.kind, "remote")

    def test_remote_errors_pickle(self):
        import pickle
        e = pickle.loads(pickle.dumps(KlongRemoteException("index: out", "p")))
        self.assertIsInstance(e, KlongRangeError)
        self.assertEqual(e.wire_name, "index")


class TestTransportErrors(IPCCase):
    def test_hierarchy(self):
        for cls in (KlongIPCException, KlongIPCConnectionClosedException,
                    KlongIPCConnectionFailureException, KlongIPCCreateConnectionException,
                    KlongIPCHandshakeException):
            self.assertTrue(issubclass(cls, KlongSystemError), cls)
            self.assertTrue(issubclass(cls, KlongIPCException), cls)
            self.assertEqual(cls("x").kind, "system")
        self.assertTrue(issubclass(KlongIPCHandshakeException, KlongIPCCreateConnectionException))

    def test_call_timeout_is_a_system_error(self):
        fake = FakeKdbServer(None)                                  # never answers
        self.addCleanup(fake.stop)
        client = self.repl()
        self.kg(client, f"c::.cli({fake.port})")
        self.kg(client, ".clit(c;0.2)")
        with self.assertRaises(KlongSystemError) as cm:
            self.kg(client, 'c("1+1")')
        self.assertIn("timed out", str(cm.exception))
        self.assertIsInstance(cm.exception, KlongIPCException)
        self.assertIsInstance(cm.exception, TimeoutError)

    def test_refused_handshake_is_a_system_error(self):
        fake = FakeKdbServer(None, reject=True)
        self.addCleanup(fake.stop)
        client = self.repl()
        with self.assertRaises(KlongSystemError):
            self.kg(client, f".cli({fake.port})")

    def test_connection_lost_is_a_system_error(self):
        server, client = self.pair()
        self.kg(client, ".clic(c)")
        with self.assertRaises(KlongSystemError):
            self.kg(client, 'c("1+1")')


class TestArgumentErrors(IPCCase):
    def test_bad_address(self):
        with self.assertRaises(KlongTypeError):
            parse_address("host:notaport")
        client = self.repl()
        with self.assertRaises(KlongTypeError):
            self.kg(client, '.srv("a:b")')

    def test_async_checks_both_arguments(self):
        client = self.repl()
        self.kg(client, "f::{x+1}")
        with self.assertRaises(KlongTypeError):
            self.kg(client, ".async(1;{x})")
        with self.assertRaises(KlongTypeError):
            self.kg(client, ".async(f;1)")                          # y must be a function too

    def test_clit_argument(self):
        server, client = self.pair()
        with self.assertRaises(KlongTypeError):
            self.kg(client, '.clit(c;"a")')
        with self.assertRaises(KlongRangeError):
            self.kg(client, ".clit(c;-1)")

    def test_cannot_listen(self):
        server, client = self.pair()
        with self.assertRaises(KlongSystemError) as cm:
            self.kg(client, f".srv({self.port})")
        self.assertIn("cannot listen", str(cm.exception))

    def test_ipc_mode_value(self):
        import os
        from unittest.mock import patch
        from klongpy.ipc.legacy import ipc_mode
        with patch.dict(os.environ, {"KLONGPY_IPC": "bogus"}):
            with self.assertRaises(KlongSystemError):
                ipc_mode("kdb")


class TestWireNames(unittest.TestCase):
    def test_name_comes_from_the_kind(self):
        for exc, name in ((KlongRangeError("r"), "index"), (KlongShapeError("s"), "length"),
                          (KlongSystemError("o"), "os"), (KlongMemoryError("m"), "wsfull"),
                          (KlongPythonError("p"), "python"), (KlongArityError("a"), "rank"),
                          (KlongInterrupt("i"), "stop"), (KlongInternalError("x"), "internal"),
                          (KlongSyntaxError("q"), "parse"), (KlongNameError("n"), "name"),
                          (KlongTypeError("t"), "type")):
            self.assertEqual(wire_error_text(exc), f"{name}: {exc.message}", exc.kind)

    def test_plain_klong_error_is_a_type_error(self):
        self.assertEqual(wire_error_text(KlongError("x must be a list")), "type: x must be a list")

    def test_a_remote_error_is_passed_on_as_received(self):
        self.assertEqual(wire_error_text(KlongRemoteException("index: far", "p")), "index: far")
        self.assertEqual(wire_error_text(KlongRemoteException("'length", "p")), "'length")

    def test_library_exceptions_are_classified(self):
        self.assertTrue(wire_error_text(IndexError("bad")).startswith("index: "))
        self.assertTrue(wire_error_text(ZeroDivisionError("z")).startswith("index: "))
        self.assertTrue(wire_error_text(MemoryError()).startswith("wsfull: "))
        self.assertTrue(wire_error_text(OSError("o")).startswith("os: "))
        self.assertTrue(wire_error_text(ValueError("shape mismatch")).startswith("length: "))
        self.assertTrue(wire_error_text(TypeError("t")).startswith("type: "))
        self.assertTrue(wire_error_text(RuntimeError("odd")).startswith("internal: "))


if __name__ == "__main__":
    unittest.main()
