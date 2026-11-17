"""Tests of the kdb+ IPC session state machine.

The session does no I/O, so every test drives two ``Session`` objects (or one
against hand-built bytes) by passing the returned bytes to the other side's
``feed``: a scripted fake peer.
"""
import struct
import unittest

import numpy as np

from klongpy.types import KGSym
from klongpy.ipc import (
    Session, Handshake, HandshakeAccepted, HandshakeRejected, Message, Failure, Closed,
    NestingLimit, KdbError, ProtocolError, IPCStateError, IPCRangeError,
    encode, decode,
)
from klongpy.ipc.session import ASYNC, SYNC, RESPONSE

from tests.test_ipc_codec import (
    load_vectors, expected_value, kg_equal, widths_of, header_consistent,
)


# ----------------------------------------------------------------------------
# scripted peers
# ----------------------------------------------------------------------------

def connect(**kw):
    """A client and a server session, handshake done, no pending bytes."""
    cli = Session.client("u", "p", **kw)
    srv = Session.server(**kw)
    (ev,) = srv.feed(cli.hello())
    assert isinstance(ev, Handshake)
    reply = srv.accept()
    (ev,) = cli.feed(reply)
    assert isinstance(ev, HandshakeAccepted)
    return cli, srv


def only(events, cls=Message):
    assert len(events) == 1, events
    assert isinstance(events[0], cls), events
    return events[0]


def ready_server(**kw):
    """A server session ready to receive hand-built messages."""
    srv = Session.server(**kw)
    srv.feed(b"u:p\x03\x00")
    srv.accept()
    return srv


def ready_client(**kw):
    cli = Session.client(**kw)
    cli.hello()
    cli.feed(b"\x03")
    return cli


def msg(body, typ=0, little=True):
    return bytes([1 if little else 0, typ, 0, 0]) + struct.pack("<I" if little else ">I", 8 + len(body)) + body


class TestHandshake(unittest.TestCase):
    def test_client_bytes(self):
        self.assertEqual(Session.client("u", "p").hello(), bytes.fromhex("75 3a 70 03 00".replace(" ", "")))
        self.assertEqual(Session.client().hello(), bytes.fromhex("3a0300"))
        self.assertEqual(Session.client("u", "p", capability=5).hello(), b"u:p\x05\x00")

    def test_accept(self):
        cli = Session.client("alice", "secret")
        srv = Session.server()
        self.assertEqual(srv.state, "handshake")
        ev = only(srv.feed(cli.hello()), Handshake)
        self.assertEqual((ev.user, ev.password, ev.capability), ("alice", "secret", 3))
        self.assertEqual(srv.state, "auth")
        reply = srv.accept()
        self.assertEqual(reply, b"\x03")
        self.assertTrue(srv.ready)
        ev = only(cli.feed(reply), HandshakeAccepted)
        self.assertEqual(ev.capability, 3)
        self.assertEqual((cli.capability, srv.capability), (3, 3))
        self.assertTrue(cli.ready)

    def test_reject(self):
        cli = Session.client("alice", "bad")
        srv = Session.server()
        only(srv.feed(cli.hello()), Handshake)
        srv.reject()                           # kdb+ just closes: nothing is sent back
        self.assertTrue(srv.closed)
        only(cli.feed_eof(), HandshakeRejected)
        self.assertTrue(cli.closed)
        with self.assertRaises(IPCStateError):
            srv.send_async(1)
        self.assertEqual(srv.feed(b"anything"), [])

    def test_capability_negotiation(self):
        for peer, expected in [(1, 1), (2, 2), (3, 3), (4, 3), (5, 3), (6, 3)]:
            with self.subTest(peer=peer):
                srv = Session.server()
                ev = only(srv.feed(b"u:p" + bytes((peer, 0))), Handshake)
                self.assertEqual(ev.capability, peer)
                self.assertEqual(srv.accept(), bytes((expected,)))
                self.assertEqual(srv.capability, expected)

    def test_capability_of_a_server_below_ours(self):
        cli = Session.client("u", "p")
        cli.hello()
        ev = only(cli.feed(b"\x01"), HandshakeAccepted)
        self.assertEqual(ev.capability, 1)
        cli = Session.client("u", "p", capability=2)
        cli.hello()
        self.assertEqual(only(cli.feed(b"\x06"), HandshakeAccepted).capability, 2)

    def test_capability_zero_and_garbage_are_refused(self):
        for greeting in (b"u:p\x00\x00",      # capability byte 0 followed by the terminator
                         b"u:p\x00",          # qPython's capability-0 form: reads 'p' as the capability
                         b"\x00",
                         b"u:p\x07\x00", b"u:p\xff\x00"):
            with self.subTest(greeting=greeting):
                srv = Session.server()
                ev = only(srv.feed(greeting), Failure)
                self.assertIsInstance(ev.error, ProtocolError)
                self.assertTrue(srv.closed)

    def test_greeting_parsing(self):
        srv = Session.server()
        ev = only(srv.feed(b"u:pa:ss\x03\x00"), Handshake)
        self.assertEqual((ev.user, ev.password), ("u", "pa:ss"))
        srv = Session.server()
        ev = only(srv.feed(b":\x03\x00"), Handshake)
        self.assertEqual((ev.user, ev.password), ("", ""))
        srv = Session.server()
        ev = only(srv.feed(b"justuser\x03\x00"), Handshake)
        self.assertEqual((ev.user, ev.password), ("justuser", ""))
        only(Session.server().feed("u:pé".encode() + b"\x03\x00"), Failure)

    def test_greeting_in_pieces(self):
        srv = Session.server()
        events = []
        for b in b"u:p\x03\x00":
            events += srv.feed(bytes((b,)))
        only(events, Handshake)

    def test_greeting_limit(self):
        srv = Session.server()
        self.assertEqual(srv.feed(b"a" * 255), [])        # still looking for the NUL
        ev = only(srv.feed(b"a"), Failure)                # 256 bytes without one
        self.assertIn("256", str(ev.error))
        srv = Session.server()
        only(srv.feed(b"a" * 300), Failure)
        # a NUL right at the limit is found
        srv = Session.server()
        only(srv.feed(b"u:" + b"x" * 252 + b"\x03\x00"), Handshake)       # 255 bytes + NUL

    def test_bytes_behind_the_greeting_wait_for_the_decision(self):
        srv = Session.server()
        body = encode("1+1")
        events = srv.feed(b"u:p\x03\x00" + msg(body, SYNC))
        only(events, Handshake)                            # nothing else until accept()
        self.assertEqual(srv.feed(b"more"), [])
        # (the stray bytes would corrupt the stream; use a fresh one for the real check)
        srv = Session.server()
        only(srv.feed(b"u:p\x03\x00" + msg(body, SYNC)), Handshake)
        srv.accept()
        m = only(srv.feed(b""))
        self.assertEqual((m.kind, m.value), ("sync", "1+1"))

    def test_client_reply_checks(self):
        only(Session.client("u", "p").feed(b"\x03"), Failure)      # a reply before the greeting was sent
        cli = Session.client("u", "p")
        cli.hello()
        only(cli.feed(b"\x09"), Failure)
        cli = Session.client("u", "p")
        cli.hello()
        self.assertEqual(only(cli.feed(b"\x00"), HandshakeAccepted).capability, 0)   # an old server

    def test_misuse(self):
        cli = Session.client()
        cli.hello()
        with self.assertRaises(IPCStateError):
            cli.hello()
        with self.assertRaises(IPCStateError):
            cli.send_async(1)                       # not ready yet
        with self.assertRaises(IPCStateError):
            Session.server().accept()
        with self.assertRaises(IPCStateError):
            Session.server().hello()
        with self.assertRaises(ValueError):
            Session.client("ué", "p").hello()
        with self.assertRaises(ValueError):
            Session("middle")
        with self.assertRaises(ValueError):
            Session.client(capability=0)


class TestFraming(unittest.TestCase):
    def test_reference_messages(self):
        cli, srv = connect()
        data, p = cli.send_sync("1+1")
        self.assertEqual(data, bytes.fromhex("0101000011000000" "0a0003000000312b31"))
        m = only(srv.feed(data))
        self.assertEqual((m.kind, m.value), ("sync", "1+1"))
        svc = srv.begin_service(m)
        reply = srv.send_response(svc, 2)
        self.assertEqual(reply, bytes.fromhex("0102000011000000f90200000000000000"))
        srv.end_service(svc)
        m = only(cli.feed(reply))
        self.assertEqual((m.kind, m.value, m.pending), ("response", 2, p))
        self.assertTrue(p.done)

    def test_async_and_error_response(self):
        cli, srv = connect()
        data = cli.send_async(KGSym("abc"))
        self.assertEqual(data, bytes.fromhex("010000000d000000f561626300"))
        m = only(srv.feed(data))
        self.assertEqual((m.kind, m.value), ("async", KGSym("abc")))
        data, p = cli.send_sync("boom")
        m = only(srv.feed(data))
        svc = srv.begin_service(m)
        reply = srv.send_error(svc, "type")
        self.assertEqual(reply, bytes.fromhex("010200000e000000807479706500"))
        srv.end_service(svc)
        m = only(cli.feed(reply))
        self.assertEqual(m.value, KdbError("type"))
        self.assertIs(m.pending, p)

    def test_call_envelope(self):
        cli, srv = connect()
        data, _ = cli.send_sync([KGSym("f"), KGSym("x")], call=True)
        self.assertEqual(data[8], 0)                        # general list, not a symbol vector
        data2, _ = cli.send_sync([KGSym("f"), KGSym("x")])
        self.assertEqual(data2[8], 11)

    def test_two_messages_and_pieces(self):
        cli, srv = connect()
        a = cli.send_async(1)
        b = cli.send_async("x")
        events = srv.feed(a + b)
        self.assertEqual([e.value for e in events], [1, "x"])
        events = []
        for i in range(len(a + b)):
            events += srv.feed((a + b)[i:i + 1])
        self.assertEqual([e.value for e in events], [1, "x"])
        self.assertEqual(srv.feed(a[:5]), [])
        self.assertEqual([e.value for e in srv.feed(a[5:])], [1])

    def test_oversize_header_rejected_before_buffering(self):
        srv = ready_server(max_message=1000)
        ev = only(srv.feed(bytes.fromhex("0100000001040000")), Failure)      # 1025
        self.assertIn("exceeds", str(ev.error))
        self.assertTrue(srv.closed)
        srv = ready_server(max_message=1000)
        self.assertEqual(srv.feed(bytes.fromhex("01000000e8030000")), [])    # exactly 1000: accepted, waiting
        srv = ready_server()
        only(srv.feed(bytes.fromhex("01000000ffffff7f")), Failure)           # default limit 256 MiB

    def test_oversize_never_allocates(self):
        srv = ready_server()
        srv.feed(bytes.fromhex("0100000000000070"))          # 1.8 GB declared
        self.assertTrue(srv.closed)
        self.assertEqual(len(srv._buf), 0)

    def test_bad_headers(self):
        cases = ["0200000008000000", "0103000008000000", "0100020008000000", "0100000108000000",
                 "0100000007000000", "01000000ffffffff", "0000000000000007"]
        for h in cases:
            with self.subTest(h=h):
                srv = ready_server()
                only(srv.feed(bytes.fromhex(h)), Failure)

    def test_bad_object_is_fatal(self):
        srv = ready_server()
        ev = only(srv.feed(msg(b"\x14\x00\x00\x00\x00\x00")), Failure)
        self.assertIsInstance(ev.error, ProtocolError)
        self.assertEqual(srv.feed(msg(encode(1))), [])             # ignored after failure

    def test_length_mismatch_is_fatal(self):
        srv = ready_server()
        only(srv.feed(msg(encode(1) + b"\x00")), Failure)          # byte after the object

    def test_big_endian_message(self):
        srv = ready_server()
        body = b"\x07\x00" + struct.pack(">i", 2) + struct.pack(">qq", 1, 2)
        m = only(srv.feed(msg(body, 1, little=False)))
        self.assertTrue(kg_equal(m.value, np.array([1, 2])))

    def test_compression(self):
        cli, srv = connect()
        big = np.arange(4000) % 5
        plain, _ = cli.send_sync(big)
        packed, p = cli.send_sync(big, compress_body=True)
        self.assertEqual(packed[2], 1)
        self.assertLess(len(packed), len(plain) // 2)
        m = only(srv.feed(packed))
        self.assertTrue(kg_equal(m.value, big))
        # small messages are not compressed even if asked
        small = cli.send_async(1, compress_body=True)
        self.assertEqual(small[2], 0)

    def test_compression_needs_peer_capability(self):
        cli = Session.client("u", "p")
        cli.hello()
        cli.feed(b"\x00")                                          # a capability-0 server
        packed = cli.send_async(np.zeros(5000, dtype=np.int64), compress_body=True)
        self.assertEqual(packed[2], 0)

    def test_compressed_reference_message(self):
        srv = ready_server()
        ex = bytes.fromhex("0100010025000000be040000000a00b004000061625503ff6203ff6203ff6203ff620103a4")
        m = only(srv.feed(ex))
        self.assertEqual(m.value, "ab" * 600)

    def test_compressed_message_limits(self):
        bomb = bytes.fromhex("01000100" "0f000000" "00000010" "004141")
        only(ready_server().feed(bomb), Failure)
        ex = bytes.fromhex("0100010025000000be040000000a00b004000061625503ff6203ff6203ff6203ff620103a4")
        srv = ready_server(max_message=1000)                       # declared 1214 > 1000
        only(srv.feed(ex), Failure)

    def test_range_error_is_per_message(self):
        cli, srv = connect(widths=32)
        data, p = cli.send_sync("x")
        big = msg(b"\xf9" + struct.pack("<q", 2 ** 40), RESPONSE)
        m = only(cli.feed(big))
        self.assertEqual(m.kind, "response")
        self.assertIsInstance(m.error, IPCRangeError)
        self.assertIsNone(m.value)
        self.assertIs(m.pending, p)
        self.assertTrue(cli.ready)                                  # the connection survives
        m = only(cli.feed(msg(encode(5, 32), ASYNC)))
        self.assertEqual(m.value, 5)

    def test_frame_size_limit(self):
        cli, _ = connect()
        with self.assertRaises(IPCRangeError):
            # a fake body whose length alone is too big, without allocating it
            class Big(bytes):
                def __len__(self):
                    return 0x80000000
            cli.frame(ASYNC, Big())

    def test_send_encodes_with_session_widths(self):
        cli, _ = connect(widths=32)
        self.assertEqual(cli.send_async(1)[8], 0xFA)                 # int atom
        cli, _ = connect()
        self.assertEqual(cli.send_async(1)[8], 0xF9)                 # long atom


class TestVectorsThroughSession(unittest.TestCase):
    """Every entry of vectors.json, the whole message, through a session."""

    @classmethod
    def setUpClass(cls):
        cls.vectors = load_vectors()

    def _receiver(self, entry, w):
        if entry.get("type", 0) == 2 or (len(entry["hex"]) >= 4 and entry["hex"][2:4] == "02"):
            cli = ready_client(widths=w)
            return cli, cli
        return ready_server(widths=w), None

    def test_decode_and_roundtrip_entries(self):
        for v in self.vectors:
            if v["mode"] == "reject":
                continue
            raw = bytes.fromhex(v["hex"])
            for w in widths_of(v):
                with self.subTest(name=v["name"], widths=w):
                    if raw[1] == RESPONSE:
                        sess = ready_client(widths=w)
                        sess.send_sync("pending")
                    else:
                        sess = ready_server(widths=w)
                    m = only(sess.feed(raw))
                    self.assertEqual(m.kind, {0: "async", 1: "sync", 2: "response"}[raw[1]])
                    self.assertTrue(kg_equal(m.value, expected_value(v)), (m.value, v["name"]))
                    if v["mode"] == "roundtrip":
                        self.assertEqual(sess.frame(raw[1], raw[8:]), raw)

    def test_decode_entries_in_one_byte_chunks(self):
        for v in self.vectors:
            if v["mode"] == "reject":
                continue
            raw = bytes.fromhex(v["hex"])
            if raw[1] == RESPONSE:
                continue
            with self.subTest(name=v["name"]):
                sess = ready_server(widths=widths_of(v)[0])
                events = []
                for i in range(len(raw)):
                    events += sess.feed(raw[i:i + 1])
                m = only(events)
                self.assertTrue(kg_equal(m.value, expected_value(v)))

    def test_reject_entries(self):
        n = 0
        for v in self.vectors:
            if v["mode"] != "reject":
                continue
            n += 1
            raw = bytes.fromhex(v["hex"])
            for w in widths_of(v):
                with self.subTest(name=v["name"], widths=w):
                    sess = ready_server(widths=w)
                    events = sess.feed(raw)
                    # a header that promises more than arrives yields nothing yet
                    # (only legal for the truncated-compressed entries, whose
                    # length field is consistent, so they fail at once)
                    ev = only(events, Failure)
                    self.assertIsInstance(ev.error, ProtocolError)
                    self.assertTrue(sess.closed)
        self.assertGreater(n, 30)

    def test_entries_are_consistent(self):
        for v in self.vectors:
            raw = bytes.fromhex(v["hex"])
            if v["mode"] != "reject":
                self.assertTrue(header_consistent(raw), v["name"])


class TestMatching(unittest.TestCase):
    def test_fifo_at_level_zero(self):
        cli, srv = connect()
        ps = []
        frames = []
        for i in range(3):
            d, p = cli.send_sync("q%d" % i, tag=i)
            ps.append(p)
            frames.append(d)
        got = []
        for d in frames:
            m = only(srv.feed(d))
            svc = srv.begin_service(m)
            reply = srv.send_response(svc, m.value.upper())
            srv.end_service(svc)
            got.append(only(cli.feed(reply)))
        self.assertEqual([m.pending.tag for m in got], [0, 1, 2])
        self.assertEqual([m.value for m in got], ["Q0", "Q1", "Q2"])
        self.assertEqual(cli.pending(), [])

    def test_interleaved_directions(self):
        cli, srv = connect()
        a1 = cli.send_async("a1")
        s1, p1 = cli.send_sync("s1", tag="s1")
        a2 = cli.send_async("a2")
        s2, p2 = cli.send_sync("s2", tag="s2")
        # the server pushes an async message to the client in the middle of it all
        push = srv.send_async(KGSym("tick"))
        events = srv.feed(a1 + s1 + a2 + s2)
        self.assertEqual([(e.kind, e.value) for e in events],
                         [("async", "a1"), ("sync", "s1"), ("async", "a2"), ("sync", "s2")])
        m = only(cli.feed(push))
        self.assertEqual((m.kind, m.value), ("async", KGSym("tick")))
        self.assertEqual(len(cli.pending()), 2)                    # the push did not touch matching
        replies = []
        for e in events:
            if e.kind == "sync":
                svc = srv.begin_service(e)
                replies.append(srv.send_response(svc, e.value + "!"))
                srv.end_service(svc)
            else:
                svc = srv.begin_service(e)
                srv.end_service(svc)
        out = cli.feed(b"".join(replies))
        self.assertEqual([(m.value, m.pending.tag) for m in out], [("s1!", "s1"), ("s2!", "s2")])
        self.assertTrue(p1.done and p2.done)

    def test_two_level_nested_example(self):
        """A calls B (r1); B, serving r1, calls back c1; A, serving c1,
        calls B again (r2).  A sees the r2 response first, then answers c1, then B answers r1."""
        A, B = connect()
        # A -> B: r1 at A's level 0
        d, r1 = A.send_sync("r1")
        mr1 = only(B.feed(d))
        S1 = B.begin_service(mr1)                         # B has nothing outstanding: no new level
        self.assertEqual(B.nesting, 0)
        # B, serving r1, calls back c1 (level 0 of B)
        d, c1 = B.send_sync("c1")
        self.assertEqual(c1.level, 0)
        mc1 = only(A.feed(d))
        S_c1 = A.begin_service(mc1)                       # A is blocked on r1: opens level 1
        self.assertEqual(A.nesting, 1)
        # A, serving c1, calls B again: r2 goes to level 1
        d, r2 = A.send_sync("r2")
        self.assertEqual(r2.level, 1)
        mr2 = only(B.feed(d))
        S2 = B.begin_service(mr2)                         # B is blocked on c1: opens level 1
        self.assertEqual(B.nesting, 1)
        reply_r2 = B.send_response(S2, "r2-result")
        B.end_service(S2)
        self.assertEqual(B.nesting, 0)
        # A: the next response completes r2 (deepest level), not r1
        m = only(A.feed(reply_r2))
        self.assertIs(m.pending, r2)
        self.assertEqual(m.value, "r2-result")
        self.assertFalse(r1.done)
        # A answers c1, closes its level
        reply_c1 = A.send_response(S_c1, "c1-result")
        A.end_service(S_c1)
        self.assertEqual(A.nesting, 0)
        # B: that response completes c1, then B answers r1
        m = only(B.feed(reply_c1))
        self.assertIs(m.pending, c1)
        reply_r1 = B.send_response(S1, "r1-result")
        B.end_service(S1)
        m = only(A.feed(reply_r1))
        self.assertIs(m.pending, r1)
        self.assertEqual(m.value, "r1-result")
        self.assertEqual((A.pending(), B.pending()), ([], []))

    def test_deeper_level_is_served_first_when_both_have_requests(self):
        cli = ready_client()
        _, r1 = cli.send_sync("r1")                        # level 0
        srv_req = Message("sync", "c", None)
        svc = cli.begin_service(srv_req)                   # opens level 1
        _, r2 = cli.send_sync("r2")
        _, r3 = cli.send_sync("r3")
        m = only(cli.feed(msg(encode(1), RESPONSE)))
        self.assertIs(m.pending, r2)                       # oldest of the deepest level
        m = only(cli.feed(msg(encode(2), RESPONSE)))
        self.assertIs(m.pending, r3)
        cli.send_response(svc, 0)
        cli.end_service(svc)
        m = only(cli.feed(msg(encode(3), RESPONSE)))
        self.assertIs(m.pending, r1)

    def test_requests_from_serving_without_a_blocking_call_stay_at_level_zero(self):
        cli, srv = connect()
        m = only(srv.feed(cli.send_async("go")))
        svc = srv.begin_service(m)
        self.assertEqual(srv.nesting, 0)
        _, p = srv.send_sync("callback")
        self.assertEqual(p.level, 0)
        srv.end_service(svc)                                # async service: no response owed
        self.assertEqual(len(srv.pending()), 1)

    def test_non_blocking_requests_do_not_open_levels(self):
        cli = ready_client()
        _, p = cli.send_sync("a", blocking=False)           # like .async(f; cb) on a remote function
        svc = cli.begin_service(Message("async", 1))
        self.assertEqual(cli.nesting, 0)
        cli.end_service(svc)
        m = only(cli.feed(msg(encode(1), RESPONSE)))
        self.assertIs(m.pending, p)

    def test_nesting_limit_answered_with_range_error(self):
        cli = ready_client(max_nesting=8)
        cli.send_sync("outer")
        services = []
        for depth in range(8):
            m = Message("sync", "req%d" % depth)
            services.append(cli.begin_service(m))
            cli.send_sync("inner%d" % depth)                # keeps a blocking call outstanding per level
        self.assertEqual(cli.nesting, 8)
        over = Message("sync", "too deep")
        with self.assertRaises(NestingLimit):
            cli.begin_service(over)
        reply = cli.refuse(over)
        self.assertEqual(reply[1], RESPONSE)
        err = decode(reply[8:])
        self.assertIsInstance(err, KdbError)
        self.assertTrue(err.text.startswith("index: "))
        self.assertEqual(cli.refuse(Message("async", 1)), b"")
        self.assertEqual(cli.nesting, 8)                    # unchanged by the refusal
        # unwind: answer the inner requests so each level closes empty
        for svc in reversed(services):
            cli.feed(msg(encode(0), RESPONSE))
            cli.send_response(svc, 0)
            cli.end_service(svc)
        self.assertEqual(cli.nesting, 0)

    def test_default_nesting_limit_is_eight(self):
        self.assertEqual(Session.client().max_nesting, 8)

    def test_level_closing_with_unanswered_requests_is_fatal(self):
        cli = ready_client()
        cli.send_sync("outer")
        svc = cli.begin_service(Message("sync", "c"))
        _, inner = cli.send_sync("inner")                    # never answered
        cli.send_response(svc, 0)
        with self.assertRaises(ProtocolError) as cm:
            cli.end_service(svc)
        self.assertTrue(cli.closed)
        self.assertIn(inner, cm.exception.pending)
        self.assertTrue(inner.failed)

    def test_unsolicited_response_is_fatal(self):
        cli = ready_client()
        only(cli.feed(msg(encode(1), RESPONSE)), Failure)
        self.assertTrue(cli.closed)
        cli = ready_client()
        _, p = cli.send_sync("x")
        events = cli.feed(msg(encode(1), RESPONSE) + msg(encode(2), RESPONSE))
        self.assertIsInstance(events[0], Message)
        self.assertIsInstance(events[1], Failure)            # the second has nothing to match

    def test_service_state_errors(self):
        cli, srv = connect()
        d, _ = cli.send_sync("x")
        m = only(srv.feed(d))
        svc = srv.begin_service(m)
        with self.assertRaises(IPCStateError):
            srv.end_service(svc)                              # sync request left unanswered
        srv.send_response(svc, 1)
        with self.assertRaises(IPCStateError):
            srv.send_response(svc, 2)                         # a second response
        srv.end_service(svc)
        with self.assertRaises(IPCStateError):
            srv.end_service(svc)
        with self.assertRaises(IPCStateError):
            srv.begin_service(Message("response", 1))
        # services end in reverse order
        s1 = srv.begin_service(Message("async", 1))
        s2 = srv.begin_service(Message("async", 2))
        with self.assertRaises(IPCStateError):
            srv.end_service(s1)
        srv.end_service(s2)
        srv.end_service(s1)
        # a response must be to the innermost sync service
        d, _ = cli.send_sync("y")
        outer = srv.begin_service(only(srv.feed(d)))
        d, _ = cli.send_sync("z")
        srv.send_sync("block")
        inner = srv.begin_service(only(srv.feed(d)))
        with self.assertRaises(IPCStateError):
            srv.send_response(outer, 1)


class TestClosingAndDeadlines(unittest.TestCase):
    def test_close_with_pending_requests(self):
        cli, srv = connect()
        _, p1 = cli.send_sync("a")
        _, p2 = cli.send_sync("b")
        out = cli.close()
        self.assertEqual(out, [p1, p2])
        self.assertTrue(p1.failed and p2.failed and not p1.done)
        self.assertTrue(cli.closed)
        self.assertEqual(cli.close(), [])                    # idempotent
        self.assertEqual(cli.feed(b"junk"), [])
        with self.assertRaises(IPCStateError):
            cli.send_async(1)

    def test_eof_with_pending_requests(self):
        cli, _ = connect()
        _, p = cli.send_sync("a")
        ev = only(cli.feed_eof(), Closed)
        self.assertEqual((ev.reason, ev.pending), ("eof", [p]))
        self.assertTrue(p.failed)
        self.assertEqual(cli.feed_eof(), [])

    def test_eof_in_the_middle_of_a_message(self):
        cli, srv = connect()
        cli_msg = msg(encode("hello"), RESPONSE)
        _, p = cli.send_sync("a")
        self.assertEqual(cli.feed(cli_msg[:10]), [])
        ev = only(cli.feed_eof(), Closed)
        self.assertEqual(ev.reason, "truncated")
        self.assertEqual(ev.pending, [p])

    def test_failure_fails_all_pending(self):
        cli, _ = connect()
        _, p1 = cli.send_sync("a")
        _, p2 = cli.send_sync("b")
        ev = only(cli.feed(bytes.fromhex("0200000008000000")), Failure)
        self.assertEqual(ev.pending, [p1, p2])
        self.assertTrue(p1.failed and p2.failed)

    def test_deadline_closes_the_connection(self):
        cli, _ = connect()
        _, p1 = cli.send_sync("slow", deadline=10.0)
        _, p2 = cli.send_sync("other", deadline=20.0)
        _, p3 = cli.send_sync("no deadline")
        self.assertEqual(cli.next_deadline(), 10.0)
        self.assertIsNone(cli.check_deadlines(9.9))
        self.assertTrue(cli.ready)
        ev = cli.check_deadlines(10.0)
        self.assertIsInstance(ev, Closed)
        self.assertEqual(ev.reason, "timeout")
        self.assertEqual(sorted(map(id, ev.pending)), sorted(map(id, [p1, p2, p3])))   # all fail: no ids to skip one
        self.assertTrue(cli.closed)
        self.assertIsNone(cli.check_deadlines(99.0))

    def test_no_deadline_means_none(self):
        cli, _ = connect()
        self.assertIsNone(cli.next_deadline())
        self.assertIsNone(cli.check_deadlines(1e12))


if __name__ == "__main__":
    unittest.main()


# These tests check the codec's own table form; klongpy.db, when another test
# has loaded it, makes decoded tables arrive as its Table instead.
_saved_table_factory = None


def setUpModule():
    global _saved_table_factory
    from klongpy.ipc import codec as _codec
    _saved_table_factory = _codec._table_factory
    _codec.set_table_factory(None)


def tearDownModule():
    from klongpy.ipc import codec as _codec
    _codec.set_table_factory(_saved_table_factory)
