"""The IPC endpoint over the kdb+ wire protocol.

Two interpreters in one process talk over localhost (ports are chosen
dynamically). Every potentially blocking step runs under a bounded wait, so a
regression fails instead of hanging the suite. Run the whole IPC suite with
``KLONGPY_IPC=kdb`` to check that the variable changes nothing.
"""
import io
import json
import os
import pickle
import signal
import socket
import struct
import tempfile
import threading
import time
import unittest
import uuid
from unittest.mock import patch

import numpy as np

from ipc_helpers import FakeKdbServer, bounded, free_port, wait_until

from klongpy import KlongInterpreter
from klongpy.core import KGSym, KlongException
from klongpy.ipc import KdbLambda, Session, decode, encode
from klongpy.ipc.endpoint import (KlongRemoteException, call_items, display_arity,
                                  lambda_value, wire_error_text)
from klongpy.repl import cleanup_repl, create_repl
from klongpy.sys_fn_ipc import (KGRemoteFnCall, KGRemoteFnProxy, KlongIPCException, NetworkClient,
                                detect_handshake, parse_address)

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples", "ipc")
VECTORS = os.path.join(os.path.dirname(__file__), "ipc_vectors", "vectors.json")


def sym(name):
    return KGSym(name)


class IPCCase(unittest.TestCase):
    """Helpers: interpreters that are shut down cleanly, bounded evaluation."""

    def repl(self):
        klong, loops = create_repl()

        def shutdown():
            bounded(klong['.system']['closeEvent'].trigger, 20)
            bounded(lambda: cleanup_repl(loops), 20)
        self.addCleanup(shutdown)
        return klong

    def kg(self, klong, text, timeout=10):
        return bounded(lambda: klong(text), timeout)

    def pair(self, **system):
        """A serving interpreter and a client interpreter connected with ``c``."""
        server = self.repl()
        client = self.repl()
        for k, v in system.items():
            server['.system'][k] = v
        self.port = free_port()
        self.assertEqual(self.kg(server, f".srv({self.port})"), 1)
        self.kg(client, f"c::.cli({self.port})")
        return server, client


class TestRequests(IPCCase):
    def test_eval(self):
        server, client = self.pair()
        self.assertEqual(self.kg(client, 'c("1+1")'), 2)
        self.kg(client, 'c("avg::{(+/x)%#x}")')
        self.assertEqual(self.kg(client, 'c("avg(!100)")'), 49.5)
        self.assertEqual(self.kg(server, "avg(!10)"), 4.5)          # it ran on the server
        self.assertEqual(self.kg(client, 'c("""hello""")'), "hello")

    def test_call_by_symbol_with_values(self):
        server, client = self.pair()
        self.kg(server, "avg::{(+/x)%#x}")
        self.assertEqual(self.kg(client, "c(:avg,,!100)"), 49.5)
        self.kg(client, "b::!100")
        self.assertEqual(self.kg(client, "c(:avg,,b)"), 49.5)

    def test_symbols_arrive_as_symbols(self):
        """Arguments are applied as they are: a symbol is never looked up or evaluated."""
        server, client = self.pair()
        self.kg(server, "foo::5;update::99;echo::{x};pair::{x,y};count::{#x}")
        r = self.kg(client, "c(:echo,,:foo)")
        self.assertIsInstance(r, KGSym)
        self.assertEqual(r, sym("foo"))
        r = self.kg(client, "c(:pair,:update,:foo)")           # (:pair; :update; :foo)
        self.assertEqual([type(x) for x in r], [KGSym, KGSym])
        self.assertEqual(list(r), [sym("update"), sym("foo")])
        r = self.kg(client, "c(:echo,,[:foo 1])")
        self.assertEqual(list(r), [sym("foo"), 1])
        self.assertEqual(type(r[0]), KGSym)
        # a string is data too, never run as a program
        self.assertEqual(self.kg(client, 'c(:echo,,"1+1")'), "1+1")

    def test_niladic_call(self):
        """f() on the server: a call with no arguments is sent as (`f; ::) and run with none."""
        server, client = self.pair()
        self.kg(server, "tick::0;bump::{tick::tick+1}")
        self.assertEqual(self.kg(client, "c(,:bump)"), 1)
        self.assertEqual(self.kg(client, "c(,:bump)"), 2)
        self.assertEqual(self.kg(server, "tick"), 2)
        q = self.kg(client, "q::c(:bump)")
        self.assertEqual(len(q.args), 0)                             # arity 0 came across
        self.assertEqual(self.kg(client, "q()"), 3)

    def test_symbol_lookup_returns_the_value(self):
        server, client = self.pair()
        self.kg(server, "foo::5;name::\"abc\";arr::1+!3")
        self.assertEqual(self.kg(client, "c(:foo)"), 5)
        self.assertEqual(self.kg(client, "c(:name)"), "abc")
        self.assertEqual(list(self.kg(client, "c(:arr)")), [1, 2, 3])

    def test_values_survive_the_wire(self):
        server, client = self.pair()
        self.kg(server, "echo::{x}")
        for src, check in [("c(:echo,,1.5)", lambda r: self.assertEqual(r, 1.5)),
                           ("c(:echo,,!5)", lambda r: self.assertEqual(list(r), [0, 1, 2, 3, 4])),
                           ('c(:echo,,"abc")', lambda r: self.assertEqual(r, "abc")),
                           ("c(:echo,,[[1 2][3 4]])", lambda r: self.assertEqual(r.tolist(), [[1, 2], [3, 4]])),
                           ("c(:echo,,[1.5 2.5])", lambda r: self.assertEqual(list(r), [1.5, 2.5]))]:
            check(self.kg(client, src))

    def test_dict_get_and_set(self):
        server, client = self.pair()
        self.kg(client, "d::.clid(c)")
        self.kg(client, "d,[:bar 42]")
        self.assertEqual(self.kg(server, "bar"), 42)
        self.assertEqual(self.kg(client, "d?:bar"), 42)
        self.kg(client, 'd,[:name "hello"]')
        self.assertEqual(self.kg(server, "name"), "hello")
        self.kg(client, "d,:vec,,[1 2 3]")
        self.assertEqual(list(self.kg(server, "vec")), [1, 2, 3])
        self.kg(server, "later::7")
        self.assertEqual(self.kg(client, "d?:later"), 7)
        with self.assertRaises(KlongException) as cm:
            self.kg(client, "d?:nosuch")
        self.assertEqual(cm.exception.kind, "name")
        self.assertIn("symbol not found", str(cm.exception))

    def test_dict_set_symbol_value_stays_a_symbol(self):
        server, client = self.pair()
        self.kg(server, "foo::5")
        self.kg(client, "d::.clid(c)")
        self.kg(client, "d,:s,:foo")
        r = self.kg(server, "s")
        self.assertIsInstance(r, KGSym)
        self.assertEqual(r, sym("foo"))

    def test_remote_set_defers_to_a_program_defined_set(self):
        """The text ``set`` means assignment only when the program has not defined its own ``set``."""
        server, client = self.pair()
        self.kg(server, "set::{x+y}")
        # ("set"; 3; 4) is a call of the program's own set, not an assignment
        nc = self.kg(client, "c")
        self.assertEqual(bounded(lambda: nc.call(KGRemoteFnCall("set", [3, 4]))), 7)

    def test_remote_function_proxy(self):
        server, client = self.pair()
        self.kg(server, "sq::{x*x};add::{x+y}")
        proxy = self.kg(client, "q::c(:sq)")
        self.assertIsInstance(proxy, KGRemoteFnProxy)
        self.assertEqual(self.kg(client, "q(7)"), 49)
        self.assertEqual(self.kg(client, "q(1.5)"), 2.25)
        add = self.kg(client, "p::c(:add)")
        self.assertEqual(len(add.args), 2)                        # arity came across
        self.assertEqual(self.kg(client, "p(3;4)"), 7)
        # proxies enumerate like any function
        self.assertEqual(list(self.kg(client, "q'1+!4")), [1, 4, 9, 16])
        # the proxy is bound to the connection and the name: redefinition on the server shows through
        self.kg(server, "sq::{x*x*x}")
        self.assertEqual(self.kg(client, "q(2)"), 8)

    def test_function_value_sent_without_source_text_is_opaque(self):
        """A function without source text (here a projection) goes out as its display form:
        the peer gets an opaque value of that arity, not a callable. Nothing crashes."""
        server, client = self.pair()
        self.kg(client, "d::.clid(c)")
        self.kg(client, "g::{x+y};p::g(1;)")
        self.kg(client, "d,:fn,p")
        self.assertEqual(str(self.kg(server, "fn")), ":monad")      # a projection reports the arguments it still takes
        q = self.kg(client, "q::d?:fn")                             # the opaque value goes back out as a lambda
        self.assertIsInstance(q, KGRemoteFnProxy)
        with self.assertRaises(KlongRemoteException) as cm:
            self.kg(client, "q(1;2)")
        self.assertIn("not callable", str(cm.exception))

    def test_remote_function_proxy_via_dict(self):
        server, client = self.pair()
        self.kg(server, "sq::{x*x}")
        self.kg(client, "d::.clid(c)")
        self.kg(client, "q::d?:sq")
        self.assertEqual(self.kg(client, "q(5)"), 25)

    def test_function_result_prints_as_its_arity(self):
        server, client = self.pair()
        r = self.kg(client, 'c("avg::{(+/x)%#x}")')
        self.assertEqual(str(r), ":monad")

    def test_errors_propagate_with_the_wire_text(self):
        server, client = self.pair()
        self.kg(server, "boom::{x+\"a\"}")
        with self.assertRaises(KlongRemoteException) as cm:
            self.kg(client, "c(:nosuchfn,,1)")
        self.assertEqual(cm.exception.wire_name, "name")
        self.assertTrue(cm.exception.text.startswith("name: "))     # the text as received
        with self.assertRaises(KlongRemoteException) as cm:
            self.kg(client, "c(:boom,,1)")
        self.assertIsNotNone(cm.exception.wire_name)               # "<wire name>: <message>"
        self.assertNotIn("Traceback", str(cm.exception))
        with self.assertRaises(KlongRemoteException):
            self.kg(client, 'c("1+")')                             # a parse error on the server
        # the connection survives errors
        self.assertEqual(self.kg(client, 'c("1+1")'), 2)

    def test_value_with_no_kdb_form_is_a_type_error_not_a_pickle(self):
        server, client = self.pair()
        self.kg(server, "echo::{x}")
        client["obj"] = server["obj"] = object()                    # a Python object has no kdb+ form
        server["bad"] = lambda x: object()
        # on the way out: refused here, nothing is sent, the connection is fine
        with self.assertRaises(KlongException) as cm:
            self.kg(client, "c(:echo,,obj)")
        self.assertEqual(cm.exception.kind, "type")
        self.assertTrue(str(cm.exception).startswith("cannot send"), str(cm.exception))
        self.assertNotIsInstance(cm.exception, KlongRemoteException)
        # on the way back: the server answers with a type error
        with self.assertRaises(KlongRemoteException) as cm:
            self.kg(client, "c(:bad,,1)")
        self.assertEqual(cm.exception.wire_name, "type")
        self.assertEqual(self.kg(client, 'c("1+1")'), 2)

    def test_concurrent_callers_get_their_own_answers(self):
        """Responses carry no ids: several threads sharing one handle are matched by the order of sending."""
        server, client = self.pair()
        self.kg(server, "inc::{x+1}")
        nc = client["c"]
        errors = []

        def worker(base):
            try:
                for i in range(30):
                    r = nc.call(KGRemoteFnCall(sym("inc"), [base + i]))
                    if r != base + i + 1:
                        errors.append((base + i, r))
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(1000 * n,), daemon=True) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(15)
            self.assertFalse(t.is_alive())
        self.assertEqual(errors, [])

    def test_values_that_are_not_requests_are_echoed(self):
        server, client = self.pair()
        self.assertEqual(self.kg(client, "c(5)"), 5)
        self.assertEqual(list(self.kg(client, "c([1 2 3])")), [1, 2, 3])


class TestCallbacks(IPCCase):
    def test_server_calls_back_during_a_client_call(self):
        server, client = self.pair()
        self.kg(server, 'f::{.cli.h("g()")}')
        self.kg(client, "g::{7}")
        self.assertEqual(self.kg(client, 'c("f()")'), 7)

    def test_two_levels_of_callbacks(self):
        """A calls B (r1); B calls back (c1); A, serving c1, calls B again (r2); B serves it at once."""
        server, client = self.pair()
        self.kg(server, 'h::{40+x};f::{.cli.h("g()")}')
        self.kg(client, "g::{c(:h,,2)}")
        self.assertEqual(self.kg(client, 'c("f()")'), 42)

    def test_three_levels_and_state(self):
        server, client = self.pair()
        # f -> g (client) -> k (server) -> m (client) -> n (server)
        self.kg(server, 'n::{x+1};k::{.cli.h("m()")};f::{.cli.h("g()")}')
        self.kg(client, "m::{c(:n,,10)};g::{c(:k,,0)}")
        self.assertEqual(self.kg(client, 'c("f()")'), 11)

    def test_callback_arguments_are_symbols_and_values(self):
        server, client = self.pair()
        self.kg(server, "f::{.cli.h(:upd,,[:ibm 2.5])}")
        self.kg(client, "upd::{x}")
        r = self.kg(client, 'c("f()")')
        self.assertEqual(list(r), [sym("ibm"), 2.5])

    def test_nesting_limit_answers_with_an_error(self):
        """Calls that bounce between the peers deeper than the nesting limit are refused, not hung."""
        server, client = self.pair()
        # (not the same name on both sides: a defined name quoted as :f is the function, not the symbol)
        self.kg(server, 'sf::{.cli.h(:cf,,x+1)}')
        self.kg(client, "cf::{c(:sf,,x+1)}")
        with self.assertRaises(KlongException) as cm:
            self.kg(client, "c(:sf,,0)", 20)
        self.assertIn("nested calls deeper than", str(cm.exception))

    def test_push_to_an_idle_client_is_served_on_the_klong_loop(self):
        """The pub/sub shape: the server calls a client that is not waiting for anything."""
        server, client = self.pair()
        self.kg(server, "hh::0;grab::{hh::.cli.h;1}")
        self.kg(client, "got::0;rec::{got::x;x*2}")
        self.kg(client, 'c("grab()")')
        r = self.kg(server, 'hh(:rec,,5)')
        self.assertEqual(r, 10)
        self.assertEqual(self.kg(client, "got"), 5)

    def test_many_pushes_arrive_in_order(self):
        server, client = self.pair()
        self.kg(server, "hh::0;grab::{hh::.cli.h;1}")
        self.kg(client, "log::[];rec::{log::log,x;1}")
        self.kg(client, 'c("grab()")')
        for i in range(20):
            self.kg(server, f"hh(:rec,,{i})")
        self.assertEqual(list(self.kg(client, "log")), list(range(20)))

    def test_hooks_see_the_connection(self):
        server = self.repl()
        client = self.repl()
        port = free_port()
        self.kg(server, "opened::0;closed::0;last::0")
        self.kg(server, ".srv.o::{last::x;opened::opened+1}")
        self.kg(server, ".srv.c::{closed::closed+1;x}")
        self.kg(server, f".srv({port})")
        self.kg(client, f"c::.cli({port})")
        self.assertTrue(wait_until(lambda: self.kg(server, "opened") == 1))
        self.assertIsInstance(self.kg(server, "last"), NetworkClient)        # the hook gets the connection handle
        self.assertEqual(self.kg(client, ".clic(c)"), 1)
        self.assertTrue(wait_until(lambda: self.kg(server, "closed") == 1))

    def test_hooks_run_on_the_klong_loop_and_may_call_the_client(self):
        """.srv.o may call the connecting client (it would deadlock on the io loop)."""
        server = self.repl()
        client = self.repl()
        port = free_port()
        self.kg(client, "welcomed::0;hello::{welcomed::x;1}")
        self.kg(server, '.srv.o::{x(:hello,,7)}')
        self.kg(server, f".srv({port})")
        self.kg(client, f"c::.cli({port})")
        self.assertTrue(wait_until(lambda: self.kg(client, "welcomed") == 7, 5))

    def test_rejected_connection_runs_no_hooks(self):
        server = self.repl()
        port = free_port()
        self.kg(server, "opened::0;closed::0")
        self.kg(server, ".srv.o::{opened::opened+1;x}")
        self.kg(server, ".srv.c::{closed::closed+1;x}")
        self.kg(server, '.srv.auth::{0}')
        self.kg(server, f".srv({port})")
        client = self.repl()
        with self.assertRaises(KlongException):
            self.kg(client, f".cli({port})")
        time.sleep(0.3)
        self.assertEqual(self.kg(server, "opened"), 0)
        self.assertEqual(self.kg(server, "closed"), 0)

    def test_error_hook_gets_errors_of_async_requests(self):
        server, client = self.pair()
        self.kg(server, "nerr::0;err::0;.srv.e::{x;err::y;nerr::nerr+1}")
        nc = self.kg(client, "c")
        self.kg(client, "c(1)")                                      # make sure the connection is up

        def send_async():                                            # an async message (type 0) that fails
            with nc._lock:
                nc._send(nc.session.send_async("nosuchname(1)"))
        bounded(send_async)
        self.assertTrue(wait_until(lambda: self.kg(server, "nerr") >= 1, 5))
        self.assertTrue(str(self.kg(server, "err")).startswith(("type: ", "name: ")))
        self.assertEqual(self.kg(client, 'c("1+1")'), 2)             # an async error does not hurt the connection


class TestAsync(IPCCase):
    def test_async_on_a_handle_does_not_block(self):
        server, client = self.pair()
        server["slow"] = lambda x: (time.sleep(0.6), x + 1)[1]
        self.kg(client, "res::0;cb::{res::x};af::.async(c;cb)")
        t0 = time.monotonic()
        self.assertEqual(self.kg(client, "af(:slow,,5)"), 1)
        self.assertLess(time.monotonic() - t0, 0.4)                  # returned at once
        self.assertTrue(wait_until(lambda: self.kg(client, "res") == 6, 5))

    def test_async_on_a_remote_function_proxy(self):
        server, client = self.pair()
        self.kg(server, "sq::{x*x}")
        self.kg(client, "res::0;cb::{res::x};q::c(:sq);aq::.async(q;cb)")
        self.assertEqual(self.kg(client, "aq(9)"), 1)
        self.assertTrue(wait_until(lambda: self.kg(client, "res") == 81, 5))

    def test_blocking_call_waits_while_async_is_outstanding(self):
        server, client = self.pair()
        server["slow"] = lambda x: (time.sleep(0.3), x)[1]
        self.kg(client, "res::0;cb::{res::x};af::.async(c;cb)")
        self.kg(client, "af(:slow,,3)")
        self.assertEqual(self.kg(client, 'c("1+1")'), 2)             # answered after the slow one, in order
        self.assertTrue(wait_until(lambda: self.kg(client, "res") == 3, 5))

    def test_async_on_a_local_function_runs_from_the_main_thread(self):
        """The wrapped call may be made from any thread; the callback runs soon after, with nothing else waking the loop."""
        client = self.repl()
        self.kg(client, "res::0;sq::{x*x};cb::{res::x};aq::.async(sq;cb)")
        self.assertEqual(self.kg(client, "aq(7)"), 1)
        self.assertTrue(wait_until(lambda: self.kg(client, "res") == 49, 3))
        self.assertEqual(self.kg(client, "aq(8)"), 1)
        self.assertTrue(wait_until(lambda: self.kg(client, "res") == 64, 3))

    def test_async_error_is_logged_and_callback_not_called(self):
        server, client = self.pair()
        self.kg(client, "res::0;cb::{res::x};af::.async(c;cb)")
        self.kg(client, "af(:nosuchfn,,3)")
        time.sleep(0.3)
        self.assertEqual(self.kg(client, "res"), 0)
        self.assertEqual(self.kg(client, 'c("1+1")'), 2)             # the connection is fine


class TestTimeouts(IPCCase):
    def test_timeout_closes_the_connection_and_fails_pending_requests(self):
        server, client = self.pair()
        server["slow"] = lambda x: (time.sleep(2.0), x)[1]
        self.assertEqual(self.kg(client, ".clit(c;0.5)"), 1)
        nc = client["c"]
        self.assertEqual(nc.call_timeout, 0.5)
        errors = []

        def call(i):
            try:
                nc.call(KGRemoteFnCall(sym("slow"), [i]))
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        t0 = time.monotonic()
        threads = [threading.Thread(target=call, args=(i,), daemon=True) for i in range(2)]
        for t in threads:
            t.start()
            time.sleep(0.05)
        for t in threads:
            t.join(8)
            self.assertFalse(t.is_alive())
        self.assertLess(time.monotonic() - t0, 5)
        self.assertEqual(len(errors), 2)
        self.assertTrue(all("timed out" in str(e) for e in errors), errors)
        self.assertTrue(all(isinstance(e, KlongIPCException) for e in errors))
        self.assertIs(errors[0], errors[1])                          # both fail with the same error
        self.assertTrue(wait_until(lambda: not nc.is_open(), 5))
        with self.assertRaises(KlongException):
            self.kg(client, 'c("1+1")')                             # the connection is gone
        self.assertEqual(self.kg(client, ".clic(c)"), 0)

    @unittest.skipUnless(threading.current_thread() is threading.main_thread()
                         and signal.getsignal(signal.SIGINT) is signal.default_int_handler,
                         "needs Python's default SIGINT handler on the main thread "
                         "(a background shell job starts with SIGINT ignored)")
    def test_interrupt_during_a_wait_closes_the_connection(self):
        """Ctrl-C while waiting for an answer cannot skip the call (no ids): the connection is closed."""
        server, client = self.pair()
        server["slow"] = lambda x: (time.sleep(1.5), x)[1]
        nc = client["c"]
        killer = threading.Timer(0.3, os.kill, args=(os.getpid(), signal.SIGINT))
        killer.start()
        try:
            with self.assertRaises(KeyboardInterrupt):
                nc.call(KGRemoteFnCall(sym("slow"), [1]))
        finally:
            killer.cancel()
        self.assertTrue(wait_until(lambda: not nc.is_open(), 5))

    def test_clit_sets_and_removes_the_limit(self):
        server, client = self.pair()
        self.assertEqual(self.kg(client, ".clit(c;2.5)"), 1)
        self.assertEqual(client["c"].call_timeout, 2.5)
        self.assertEqual(self.kg(client, ".clit(c;0)"), 1)
        self.assertIsNone(client["c"].call_timeout)
        self.kg(client, "d::.clid(c)")
        self.assertEqual(self.kg(client, ".clit(d;4)"), 1)           # a dict handle shares the connection
        self.assertEqual(client["c"].call_timeout, 4.0)
        self.assertEqual(self.kg(client, ".clit(5;1)"), 0)           # not a handle
        with self.assertRaises(KlongException):
            self.kg(client, '.clit(c;"x")')
        with self.assertRaises(KlongException):
            self.kg(client, ".clit(c;-1)")

    def test_async_timeout_closes_the_connection(self):
        server, client = self.pair()
        server["slow"] = lambda x: (time.sleep(1.5), x)[1]
        self.kg(client, ".clit(c;0.4);res::0;cb::{res::x};af::.async(c;cb)")
        self.kg(client, "af(:slow,,3)")
        nc = client["c"]
        self.assertTrue(wait_until(lambda: not nc.is_open(), 5))
        self.assertEqual(self.kg(client, "res"), 0)

    def test_silent_peer_fails_the_handshake_in_bounded_time(self):
        peer = FakeKdbServer(reply_handshake=False)
        self.addCleanup(peer.stop)
        client = self.repl()
        with patch("klongpy.ipc.endpoint.HANDSHAKE_TIMEOUT", 0.4):
            t0 = time.monotonic()
            with self.assertRaises(KlongException) as cm:
                self.kg(client, f".cli({peer.port})", 8)
        self.assertLess(time.monotonic() - t0, 5)
        self.assertIn("timed out", str(cm.exception))


class TestAuth(IPCCase):
    def serve(self, hook=None):
        server = self.repl()
        if hook:
            self.kg(server, hook)
        port = free_port()
        self.kg(server, f".srv({port})")
        return server, port

    def test_no_hook_accepts_any_credentials(self):
        server, port = self.serve()
        client = self.repl()
        self.kg(client, f'c::.cli("127.0.0.1:{port}:anyone:anything")')
        self.assertEqual(self.kg(client, 'c("1+1")'), 2)

    def test_auth_accepts(self):
        server, port = self.serve('.srv.auth::{:[(x="bob")&(y="s3cret");1;0]}')
        client = self.repl()
        self.kg(client, f'c::.cli("127.0.0.1:{port}:bob:s3cret")')
        self.assertEqual(self.kg(client, 'c("1+1")'), 2)

    def test_auth_rejects(self):
        server, port = self.serve('.srv.auth::{:[(x="bob")&(y="s3cret");1;0]}')
        client = self.repl()
        for addr in (f'"127.0.0.1:{port}:bob:wrong"', f'"127.0.0.1:{port}:eve:s3cret"', str(port)):
            t0 = time.monotonic()
            with self.assertRaises(KlongException) as cm:
                self.kg(client, f".cli({addr})", 8)
            self.assertLess(time.monotonic() - t0, 4)
            self.assertIn("handshake", str(cm.exception))
        # a rejected client leaves the server fine for the right one
        self.kg(client, f'c::.cli("127.0.0.1:{port}:bob:s3cret")')
        self.assertEqual(self.kg(client, 'c("1+1")'), 2)

    def test_auth_that_fails_rejects(self):
        server, port = self.serve(".srv.auth::{x+1}")
        client = self.repl()
        with self.assertRaises(KlongException):
            self.kg(client, f".cli({port})", 8)

    def test_auth_gets_user_and_password_as_strings(self):
        server, port = self.serve('seen::"";.srv.auth::{seen::x,"/",y;1}')
        client = self.repl()
        self.kg(client, f'c::.cli("127.0.0.1:{port}:al:pa:ss")')            # the password may contain ':'
        self.assertEqual(self.kg(server, "seen"), "al/pa:ss")

    def test_address_forms(self):
        self.assertEqual(parse_address(8888), ("localhost", 8888, "", ""))
        self.assertEqual(parse_address("8888"), ("localhost", 8888, "", ""))
        self.assertEqual(parse_address("h:9"), ("h", 9, "", ""))
        self.assertEqual(parse_address("h:9:u:p"), ("h", 9, "u", "p"))
        self.assertEqual(parse_address("h:9:u:p:q"), ("h", 9, "u", "p:q"))
        self.assertEqual(parse_address(":9"), ("localhost", 9, "", ""))
        with self.assertRaises(KlongException):
            parse_address("h:notaport")


class TestBytesOverIPC(IPCCase):
    def test_bytes_cross_and_arrive_as_integers(self):
        server, client = self.pair()
        self.kg(server, "idf::{x}")
        for b in (b"", b"\x00\xff", bytearray(b"hi")):
            client["v"] = b
            r = self.kg(client, "c(:idf,,v)")
            self.assertEqual([int(x) for x in r], list(b))
        client["v"] = [1, b"\x07\x08", {"k": 1}]
        r = self.kg(client, "c(:idf,,v)")
        self.assertEqual([int(x) for x in r[1]], [7, 8])

    def test_bytes_as_a_dictionary_value_and_large(self):
        server, client = self.pair()
        self.kg(server, "idf::{x}")
        client["v"] = {"k": b"\x01\x02"}
        r = self.kg(client, "c(:idf,,v)")
        self.assertEqual([int(x) for x in r["k"]], [1, 2])
        big = bytes(range(256)) * 4096
        client["v"] = big
        r = self.kg(client, "c(:idf,,v)", 30)
        self.assertEqual(bytes(np.asarray(client._backend.to_numpy(r)).astype(np.uint8)), big)

    def test_large_bytes_over_the_limit_close_the_connection(self):
        server, client = self.pair(ipc_max_message=100000)
        client["v"] = bytes(200000)
        with self.assertRaises(Exception):
            self.kg(client, "c(:idf,,v)", 8)
        self.assertTrue(wait_until(lambda: not client["c"].is_open(), 5))

    def test_compressed_bytes(self):
        server, client = self.pair()
        self.kg(server, "idf::{x}")
        client.ipc_compress = server.ipc_compress = True
        client["v"] = b"abcd" * 20000
        sent = []
        real_send = NetworkClient._send

        def spy(nc, data):
            sent.append(data)
            return real_send(nc, data)
        with patch.object(NetworkClient, "_send", spy), patch("klongpy.ipc.endpoint._is_loopback", return_value=False):
            r = self.kg(client, "c(:idf,,v)", 20)
        self.assertTrue(any(d[2] == 1 for d in sent))
        self.assertEqual(bytes(np.asarray(client._backend.to_numpy(r)).astype(np.uint8)), b"abcd" * 20000)

    def test_a_set_is_still_a_type_error(self):
        server, client = self.pair()
        client["v"] = {1, 2}
        with self.assertRaises(Exception) as cm:
            self.kg(client, "c(:idf,,v)")
        self.assertIn("set", str(cm.exception))


class TestLimits(IPCCase):
    def test_oversize_request_is_refused_and_the_connection_closed(self):
        server, client = self.pair(ipc_max_message=2000)
        self.assertEqual(self.kg(client, 'c("1+1")'), 2)
        big = "x" * 5000
        self.kg(client, f'big::"{big}"')
        with self.assertRaises(Exception) as cm:
            self.kg(client, "c(big)", 8)
        self.assertIsInstance(cm.exception, (KlongIPCException, KlongException))
        self.assertTrue(wait_until(lambda: not client["c"].is_open(), 5))
        # the server is still serving
        self.kg(client, f"c2::.cli({self.port})")
        self.assertEqual(self.kg(client, 'c2("1+1")'), 2)

    def test_oversize_response_is_refused_by_the_client(self):
        server = self.repl()
        client = self.repl()
        client['.system']['ipc_max_message'] = 2000
        port = free_port()
        self.kg(server, f".srv({port})")
        self.kg(server, 'bigfn::{5000#"a"}')
        self.kg(client, f"c::.cli({port})")
        with self.assertRaises(Exception):
            self.kg(client, "c(:bigfn,,0)", 8)
        self.assertTrue(wait_until(lambda: not client["c"].is_open(), 5))

    def test_garbage_from_a_client_does_not_hurt_the_server(self):
        server = self.repl()
        port = free_port()
        self.kg(server, f".srv({port})")
        for junk in (b"\xff" * 64, b"abc:def\x09\x00", b"\x00" * 10, b"x" * 300):
            s = socket.create_connection(("127.0.0.1", port), timeout=3)
            s.sendall(junk)
            s.settimeout(6)
            try:
                s.recv(1)                    # closed (b"") or reset, never a hang
            except OSError:
                pass
            s.close()
        client = self.repl()
        self.kg(client, f"c::.cli({port})")
        self.assertEqual(self.kg(client, 'c("1+1")'), 2)


class TestCompression(IPCCase):
    def test_off_by_default_and_on_for_a_remote_peer_only(self):
        server, client = self.pair()
        self.kg(server, "big::50000#[7]")
        sent = []
        real_send = NetworkClient._send

        def spy(nc, data):
            sent.append(data)
            return real_send(nc, data)
        with patch.object(NetworkClient, "_send", spy):
            r = self.kg(client, "c(:big)")
            self.assertEqual(len(r), 50000)
            self.assertTrue(all(d[2] == 0 for d in sent))            # nothing compressed by default
            client.ipc_compress = server.ipc_compress = True
            sent.clear()
            self.assertEqual(len(self.kg(client, "c(:big)")), 50000)
            self.assertTrue(all(d[2] == 0 for d in sent))            # loopback peers never get compressed messages
            with patch("klongpy.ipc.endpoint._is_loopback", return_value=False):
                sent.clear()
                r = self.kg(client, "c(:big)")
                self.assertEqual(list(r[:3]), [7, 7, 7])
                self.assertEqual(len(r), 50000)
                self.assertEqual(self.kg(client, "+/c(:big)"), 350000)
                self.kg(client, "up::50000#[3]")
                self.kg(client, "d::.clid(c)")
                self.kg(client, "d,:upl,,up")                         # a large request, compressed on the way out
                self.assertEqual(self.kg(server, "+/upl"), 150000)
                self.assertTrue(any(d[2] == 1 for d in sent), "no compressed message was sent")
        self.assertEqual(self.kg(client, 'c("1+1")'), 2)


def old_client_frame(command="1+1"):
    """What a 0.7 client sends first: a 16-byte id, a 4-byte length and a pickle."""
    data = pickle.dumps(command)
    return uuid.uuid4().bytes + struct.pack("!I", len(data)) + data


class Marker:
    """Unpickling this creates ``path``: it shows whether a peer's bytes were unpickled."""

    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (open, (self.path, "w"))


class TestOldProtocolRemoved(IPCCase):
    """The pickle protocol of 0.7 is gone: there is one protocol."""

    def setUp(self):
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("KLONGPY_IPC", None)

    def closed_quickly(self, port, payload):
        """Send ``payload``; True when the server closes the connection without a byte in answer."""
        s = socket.create_connection(("127.0.0.1", port), timeout=5)
        self.addCleanup(s.close)
        s.sendall(payload)
        t0 = time.monotonic()
        try:
            data = s.recv(100)
        except ConnectionResetError:
            data = b""
        self.assertLess(time.monotonic() - t0, 3)
        return data == b""

    def test_legacy_mode_is_a_start_up_error(self):
        os.environ["KLONGPY_IPC"] = "legacy"
        server = self.repl()
        with self.assertRaises(KlongException) as cm:
            self.kg(server, f".srv({free_port()})")
        self.assertIn("removed in 2.0", str(cm.exception))
        with self.assertRaises(KlongException) as cm:
            self.kg(server, f".cli({free_port()})")
        self.assertIn("removed in 2.0", str(cm.exception))
        self.assertEqual(self.kg(server, "1+1"), 2)

    def test_kdb_and_auto_are_accepted_and_other_values_are_errors(self):
        for value in ("kdb", "auto", " KDB ", ""):
            os.environ["KLONGPY_IPC"] = value
            server, client = self.pair_no_env()
            self.assertEqual(self.kg(client, 'c("1+1")'), 2)
        os.environ["KLONGPY_IPC"] = "bogus"
        server = self.repl()
        with self.assertRaises(KlongException) as cm:
            self.kg(server, f".srv({free_port()})")
        self.assertIn("bogus", str(cm.exception))

    def pair_no_env(self):
        server = self.repl()
        port = free_port()
        self.kg(server, f".srv({port})")
        client = self.repl()
        self.kg(client, f"c::.cli({port})")
        return server, client

    def test_an_old_client_frame_is_refused_and_not_unpickled(self):
        server = self.repl()
        port = free_port()
        self.kg(server, f".srv({port})")
        with tempfile.TemporaryDirectory() as d:
            marker = os.path.join(d, "unpickled")
            with self.assertLogs(level="WARNING") as logs:
                self.assertTrue(self.closed_quickly(port, old_client_frame(Marker(marker))))
            self.assertFalse(os.path.exists(marker))
        self.assertTrue(any("pickle protocol of KlongPy 0.7" in m and "removed in 2.0" in m for m in logs.output),
                        logs.output)
        # the server is still serving kdb+ clients
        client = self.repl()
        self.kg(client, f"c::.cli({port})")
        self.assertEqual(self.kg(client, 'c("2+3")'), 5)

    def test_a_first_frame_of_any_old_client_is_refused(self):
        server = self.repl()
        port = free_port()
        self.kg(server, f".srv({port})")
        for _ in range(5):
            self.assertTrue(self.closed_quickly(port, old_client_frame()))

    def test_legacy_frames_are_refused_with_srv_auth_and_without(self):
        server = self.repl()
        port = free_port()
        self.kg(server, f".srv({port})")
        self.assertTrue(self.closed_quickly(port, old_client_frame()))
        self.kg(server, ".srv.auth::{1}")
        self.assertTrue(self.closed_quickly(port, old_client_frame()))
        self.kg(server, ".srv.auth::{0}")
        self.assertTrue(self.closed_quickly(port, old_client_frame()))

    def test_legacy_frames_are_refused_whatever_the_size_limit(self):
        server = self.repl()
        server['.system']['ipc_max_message'] = 100
        port = free_port()
        self.kg(server, f".srv({port})")
        self.assertTrue(self.closed_quickly(port, old_client_frame()))
        self.assertTrue(self.closed_quickly(port, old_client_frame("1+1" + " " * 500)))

    def test_a_server_with_srv_auth_still_checks_kdb_clients(self):
        server = self.repl()
        port = free_port()
        self.kg(server, f".srv({port})")
        self.kg(server, '.srv.auth::{(x="bob")&(y="pw")}')
        client = self.repl()
        self.kg(client, f'c::.cli("127.0.0.1:{port}:bob:pw")')
        self.assertEqual(self.kg(client, 'c("1+1")'), 2)
        with self.assertRaises(KlongException):
            self.kg(client, f'.cli("127.0.0.1:{port}:bob:wrong")', 12)

    def test_a_client_of_an_old_server_fails_at_the_handshake(self):
        """A 0.7 server reads a 16-byte id first and answers nothing to a kdb+ hello."""
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        received = []
        conns = []

        def old_server():
            c, _ = listener.accept()
            conns.append(c)
            c.settimeout(5)
            try:
                received.append(c.recv(16))     # waits for its 16-byte message id, never replies
            except OSError:
                pass

        t = threading.Thread(target=old_server, daemon=True)
        t.start()
        self.addCleanup(listener.close)
        self.addCleanup(lambda: [c.close() for c in conns])
        client = self.repl()
        t0 = time.monotonic()
        with patch("klongpy.ipc.endpoint.HANDSHAKE_TIMEOUT", 1.0):
            with self.assertRaises(KlongException) as cm:
                self.kg(client, f".cli({port})", 8)
        self.assertLess(time.monotonic() - t0, 4)
        self.assertIn("handshake timed out", str(cm.exception))
        self.assertIn("0.7", str(cm.exception))
        t.join(6)

    def test_detect_handshake(self):
        d = detect_handshake
        self.assertEqual(d(b":\x03\x00"), "kdb")
        self.assertEqual(d(b"user:pass\x03\x00"), "kdb")
        self.assertIsNone(d(b"user:pa"))                             # need more
        self.assertIsNone(d(b"user:pass\x03"))
        self.assertEqual(d(old_client_frame()[:9]), "pickle")
        self.assertEqual(d(b"\x81"), "pickle")
        self.assertEqual(d(b"abc\x00"), "pickle")                    # a zero not preceded by 1..6
        self.assertEqual(d(b"ab\x03\x00"), "pickle")                 # no ':' before it
        self.assertEqual(d(b"a" * 300), "bad")
        # a version-4 UUID always has byte 8 >= 0x80, so 9 bytes decide every old frame
        for _ in range(2000):
            self.assertEqual(d(uuid.uuid4().bytes[:9]), "pickle")


class TestRawQClient(IPCCase):
    """A q client played by hand over a raw socket: the stand-in for a real q process."""

    def vectors(self):
        with open(VECTORS) as f:
            return {v["name"]: v for v in json.load(f)}

    def connect(self, port, greeting=b":\x03\x00"):
        s = socket.create_connection(("127.0.0.1", port), timeout=5)
        self.addCleanup(s.close)
        s.sendall(greeting)
        self.assertEqual(self.read_exact(s, 1), b"\x03")
        return s

    @staticmethod
    def read_exact(s, n):
        buf = b""
        while len(buf) < n:
            chunk = s.recv(n - len(buf))
            if not chunk:
                raise AssertionError(f"connection closed after {len(buf)} of {n} bytes")
            buf += chunk
        return buf

    def read_message(self, s):
        head = self.read_exact(s, 8)
        (n,) = struct.unpack("<I", head[4:8])
        return head + self.read_exact(s, n - 8)

    def test_handshake_then_sync_1_plus_1(self):
        vec = self.vectors()
        server = self.repl()
        port = free_port()
        self.kg(server, f".srv({port})")
        s = self.connect(port)
        s.sendall(bytes.fromhex(vec["string-1+1-sync"]["hex"]))
        self.assertEqual(self.read_message(s), bytes.fromhex(vec["response-long-2"]["hex"]))

    def test_handshake_with_credentials_and_older_capability(self):
        server = self.repl()
        port = free_port()
        self.kg(server, f'.srv({port});seen::"";.srv.auth::{{seen::x,"/",y;1}}')
        s = socket.create_connection(("127.0.0.1", port), timeout=5)
        self.addCleanup(s.close)
        s.sendall(b"kdb:pass\x02\x00")                              # an older client: capability 2
        self.assertEqual(self.read_exact(s, 1), b"\x02")            # we answer min(peer, 3)
        self.assertEqual(self.kg(server, "seen"), "kdb/pass")

    def test_call_with_symbol_function_and_error_reply(self):
        vec = self.vectors()
        server = self.repl()
        port = free_port()
        self.kg(server, f".srv({port});add::{{x+y}}")
        s = self.connect(port)
        s.sendall(bytes.fromhex(vec["call-symbol-fn"]["hex"]))       # (`add; 1; 2)
        self.assertEqual(self.read_message(s), bytes.fromhex("0102000011000000f90300000000000000"))
        s.sendall(bytes.fromhex(vec["call-string-fn"]["hex"]))       # ("add"; 1; 2)
        self.assertEqual(self.read_message(s), bytes.fromhex("0102000011000000f90300000000000000"))
        # an error comes back as the error object (type byte 0x80, text, NUL), not as a closed connection
        bad = Session.client()
        bad.feed(b"\x03")                                            # not needed for framing only
        body = encode("nosuchfn(1)")
        s.sendall(b"\x01\x01\x00\x00" + struct.pack("<I", 8 + len(body)) + body)
        reply = self.read_message(s)
        self.assertEqual(reply[:2], b"\x01\x02")
        self.assertEqual(reply[8], 0x80)
        self.assertEqual(reply[-1], 0)
        text = reply[9:-1].decode()
        self.assertRegex(text, r"^(name|type|parse|index|internal): ")
        # and the connection still serves
        s.sendall(bytes.fromhex(vec["string-1+1-sync"]["hex"]))
        self.assertEqual(self.read_message(s), bytes.fromhex(vec["response-long-2"]["hex"]))

    def test_async_message_gets_no_reply(self):
        vec = self.vectors()
        server = self.repl()
        port = free_port()
        self.kg(server, f".srv({port});seen::0")
        s = self.connect(port)
        body = encode("seen::41+1")
        s.sendall(b"\x01\x00\x00\x00" + struct.pack("<I", 8 + len(body)) + body)   # type 0: async
        self.assertTrue(wait_until(lambda: self.kg(server, "seen") == 42, 5))
        s.sendall(bytes.fromhex(vec["string-1+1-sync"]["hex"]))
        # the next message on the wire is the answer to the sync request, nothing was sent for the async one
        self.assertEqual(self.read_message(s), bytes.fromhex(vec["response-long-2"]["hex"]))

    def test_symbol_lookup_and_set_by_a_q_client(self):
        server = self.repl()
        port = free_port()
        self.kg(server, f".srv({port});foo::5")
        s = self.connect(port)

        def sync(value, call=False):
            body = encode_call_or_value(value, call)
            s.sendall(b"\x01\x01\x00\x00" + struct.pack("<I", 8 + len(body)) + body)
            reply = self.read_message(s)
            return decode(reply, reply[0] == 1, 64, offset=8)

        self.assertEqual(sync(sym("foo")), 5)                        # h`foo
        self.assertIs(sync(["set", sym("bar"), 7], call=True), decode(b"\x65\x00"))   # h("set";`bar;7) -> (::)
        self.assertEqual(self.kg(server, "bar"), 7)
        r = sync(sym("nosuch"))
        self.assertTrue(type(r).__name__ == "KdbError" and r.text.startswith("name: "))

    def test_q_function_value_is_opaque_to_klong(self):
        """A lambda with a q context from a q peer stays an opaque value, sendable back unchanged."""
        lam = KdbLambda("", "{x+y}")
        self.assertIs(lambda_value(KlongInterpreter(), lam), lam)
        self.assertEqual(decode(encode(lam)), lam)


def encode_call_or_value(value, call):
    from klongpy.ipc import encode_call
    return encode_call(value) if call else encode(value)


class TestExampleFlows(IPCCase):
    """The flows of examples/ipc/ over the kdb+ protocol, with the port made dynamic."""

    def load(self, klong, name, port, extra=""):
        with open(os.path.join(EXAMPLES, name)) as f:
            text = f.read().replace("8888", str(port))
        with tempfile.NamedTemporaryFile("w", suffix=".kg", delete=False) as t:
            t.write(text)
        self.addCleanup(os.unlink, t.name)
        self.kg(klong, f'.l("{t.name}")')

    def test_simple_rpc(self):
        server, client = self.repl(), self.repl()
        port = free_port()
        self.load(server, "simple_server.kg", port)
        chan = client[".sys.cout"]
        old, chan.raw = chan.raw, io.StringIO()
        try:
            self.load(client, "simple_client.kg", port)
            printed = chan.raw.getvalue()
        finally:
            chan.raw = old
        # the example prints the square of 5 called through a proxy and through the dictionary
        self.assertEqual([l for l in printed.splitlines() if l.strip() == "25"], ["25", "25"])

    def test_pubsub(self):
        server, client = self.repl(), self.repl()
        port = free_port()
        self.load(server, "srv_pubsub.kg", port)
        self.load(client, "cli_pubsub.kg", port)
        # replace the example's printing callback with one that records
        self.kg(client, "got::[];update::{got::got,,x}")
        self.assertEqual(self.kg(server, "#clients"), 1)             # .srv.o registered the client, subscribe added its tickers
        self.assertTrue(wait_until(lambda: self.kg(client, "#got") >= 1, 6), "no update arrived")
        first = self.kg(client, "got@0")
        self.assertTrue(len(first) >= 1)

    def test_broadcast(self):
        """The server evaluates a string on each connected client once a second; the client program closes itself later."""
        server, client = self.repl(), self.repl()
        port = free_port()
        self.load(server, "srv_broadcast.kg", port)
        chan = client[".sys.cout"]
        old, chan.raw = chan.raw, io.StringIO()
        try:
            self.load(client, "cli_broadcast.kg", port)
            self.assertTrue(wait_until(lambda: self.kg(server, "#clients") == 1, 5))
            self.assertTrue(wait_until(lambda: "hello from server" in chan.raw.getvalue(), 6), chan.raw.getvalue())
        finally:
            seen = chan.raw.getvalue()
            chan.raw = old
        self.assertEqual(self.kg(client, ".clic(cli)"), 1)           # the example hangs up after 10 s; do it now
        self.assertTrue(wait_until(lambda: self.kg(server, "#clients") == 0, 5))


class TestPureFunctions(unittest.TestCase):
    def test_call_items(self):
        obj = lambda *x: np.array(x, dtype=object)  # noqa: E731
        self.assertEqual(call_items(obj(sym("f"), 1, 2)), [sym("f"), 1, 2])
        self.assertEqual(call_items(obj("f", 1)), ["f", 1])
        self.assertEqual(call_items(obj(sym("f"), sym("x"))), [sym("f"), sym("x")])     # symbol vector, two items
        self.assertIsNone(call_items(obj(sym("f"))))                                     # one item: data
        self.assertIsNone(call_items(np.array([1, 2, 3])))
        self.assertIsNone(call_items(obj(1, sym("f"))))
        self.assertIsNone(call_items("f"))
        self.assertIsNone(call_items(5))
        self.assertEqual(len(call_items(obj(KdbLambda("kg", "{1}")))), 1)               # a kg lambda alone: nilad call
        self.assertIsNone(call_items(obj(KdbLambda("", "{x}"), 1)))                       # q lambda: not ours to run

    def test_display_arity(self):
        self.assertEqual([display_arity(t) for t in (":nilad", ":monad", ":dyad", ":triad", ":fn/7")], [0, 1, 2, 3, 7])
        self.assertIsNone(display_arity("{x+1}"))
        self.assertIsNone(display_arity(":fn/x"))

    def test_lambda_value(self):
        k = KlongInterpreter()
        self.assertEqual(str(lambda_value(k, KdbLambda("kg", ":dyad"))), ":dyad")
        fn = lambda_value(k, KdbLambda("kg", "{x+1}"))
        self.assertEqual(fn.arity, 1)
        self.assertIsInstance(lambda_value(k, KdbLambda("kg", ":monad"), object(), sym("f")), KGRemoteFnProxy)
        # text that is not exactly one function literal is never run
        for text in ('.p("pwned")', "{x}(1)", "1+1", "{x}{y}", ""):
            self.assertIsInstance(lambda_value(k, KdbLambda("kg", text)), KdbLambda, text)

    def test_wire_error_text(self):
        self.assertEqual(wire_error_text(KeyError(sym("zz"))), "name: symbol not found: zz")
        self.assertTrue(wire_error_text(IndexError("bad")).startswith("index: "))
        self.assertTrue(wire_error_text(KlongException("x must be a list")).startswith("type: "))
        self.assertTrue(wire_error_text(RuntimeError("odd")).startswith("internal: "))
        self.assertNotIn("\x00", wire_error_text(KlongException("a\x00b")))
        self.assertEqual(KlongRemoteException("type: bad", "p").wire_name, "type")
        self.assertIsNone(KlongRemoteException("'length", "p").wire_name)             # a kdb+ error text


if __name__ == "__main__":
    unittest.main()


class TestFunctionValues(IPCCase):
    """A function literal crosses the wire as its source text and is callable on arrival."""

    def test_function_argument_runs_on_the_server(self):
        server, client = self.pair()
        self.kg(server, "ap::{x(y)}")
        self.kg(client, "ap::c(:ap)")
        self.assertEqual(self.kg(client, "ap({x+1};10)"), 11)

    def test_function_set_through_a_remote_dictionary(self):
        server, client = self.pair()
        self.kg(client, "d::.clid(c)")
        self.kg(client, "d,:fn,{x+1}")
        self.assertEqual(self.kg(server, "fn(10)"), 11)
        self.assertEqual(self.kg(client, "t::d?:fn;t(10)"), 11)

    def test_projection_goes_as_its_display_form(self):
        from klongpy import KlongInterpreter
        from klongpy.ipc.endpoint import function_source_text
        k = KlongInterpreter()
        self.assertEqual(function_source_text(k("{x+1}")), "{x+1}")
        self.assertIsNone(function_source_text(k("f::{x*y};f(2;)")))

    def test_niladic_function_text_is_not_run_on_arrival(self):
        server, client = self.pair()
        self.kg(server, "n::0")
        self.kg(client, "d::.clid(c)")
        self.kg(client, "d,:bump,{n::n+1}")
        self.assertEqual(self.kg(server, "n"), 0)
        self.assertEqual(self.kg(server, "bump()"), 1)
