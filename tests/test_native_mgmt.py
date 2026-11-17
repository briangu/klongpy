"""
The management service (klongpy/lib/mgmt.kg), on the host: a management process fronting three
node processes over localhost TCP, driven from its own prompt and by a desktop KlongPy client.

Skipped unless the `klong` binary is built:

    cd native && cargo build --release

Every node is the native runtime in a child process (see test_native_ipc.py). The management process is
the native runtime too, except in TestKlongPyAsManagementNode, where KlongPy loads the same file.
A killed process stands for a pulled cable; a node whose function waits stands for a slow node.
"""
import concurrent.futures
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import unittest

import numpy as np

from ipc_helpers import free_port, wait_until
from test_native_ipc import BINARY, NEED_NATIVE, TIMEOUT, Case, Native, NativeError, as_np, norm

from klongpy.core import KGSym
from klongpy.errors import (KlongArityError, KlongError, KlongInternalError, KlongMemoryError, KlongNameError,
                            KlongRangeError, KlongRemoteError, KlongShapeError, KlongSyntaxError, KlongSystemError,
                            KlongTypeError)

S = KGSym

#: what every node defines; `work` is replaced on a node that is to be slow
NODE = ('sq::{x*x};add::{x+y};nil::{42};idf::{x};total::{+/x};fail::{1+"a"};answer::42;'
        'wait::{[t];t::.pc()+x;{.pc()<t}{x}:~0;x};work::{x*2};raise::{.err(x;"boom")}')

#: the kinds a node can fail with, and the class KlongPy raises for each
KINDS = [("syntax", KlongSyntaxError), ("name", KlongNameError), ("arity", KlongArityError), ("type", KlongTypeError),
         ("shape", KlongShapeError), ("range", KlongRangeError), ("system", KlongSystemError),
         ("memory", KlongMemoryError), ("internal", KlongInternalError), ("remote", KlongRemoteError)]

#: errors the runtime raises by itself: request, kind, part of the message
NATURAL = [('"1+""a"""', "type", "cannot apply + (Plus) to integer and string"),
           ('"fail(1)"', "type", "cannot apply + (Plus)"),
           ('"1+"', "syntax", "missing right operand"),
           ('"[1 2 3]@9"', "range", "out of range"),
           ('"nosuch(1)"', "name", "nosuch"),
           ('".timer(1)"', "arity", None),
           ('"[1 2 3]+[1 2]"', "shape", None),
           ('".ic(""no-such-file"")"', "system", None)]


#: an error as the prompt shows it; one raised inside a loaded file has the file's name before the position
ERROR_LINE = re.compile(r"^(?:(?:[\w.<>/-]+:)?\d+:\d+: )?"
                        r"(syntax|name|arity|type|shape|range|system|memory|internal|remote|interrupt) error: (.*)$")


class Node(Native):
    """A native runtime in a directory that outlives the process, so that it can be started again there."""

    def __init__(self, directory=None, args=()):
        self.dir = directory or tempfile.mkdtemp(prefix="klong-mgmt-")
        self.proc = subprocess.Popen([BINARY, *args], cwd=self.dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
        self.lines = queue.Queue()
        self.log = []
        threading.Thread(target=self._read, daemon=True).start()

    def stop(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait(5)
        for f in (self.proc.stdin, self.proc.stdout):
            try:
                f.close()
            except OSError:
                pass

    def run(self, line, timeout=TIMEOUT):
        self.send(line)
        out = self.output(timeout)
        # what the event loop logged meanwhile (a request that was out on a link when it failed) is not the answer
        self.log += [x for x in out if x.startswith((".async: ", "IPC: "))]
        out = [x for x in out if not x.startswith((".async: ", "IPC: "))]
        for i, text in enumerate(out):
            m = ERROR_LINE.match(text)
            if m:
                raise NativeError(m.group(1), m.group(2), "\n".join(out[i:]))
        return out

    def kill(self):
        """A pulled cable: the process is gone at once."""
        self.proc.kill()
        self.proc.wait(5)


class Cluster(Case):
    """Three nodes n1 n2 n3, each serving on its own port."""

    NAMES = ("n1", "n2", "n3")

    def start_node(self, name, directory=None, args=(), setup=NODE):
        n = Node(directory, args)
        self.addCleanup(n.stop)
        if directory is None:
            self.addCleanup(shutil.rmtree, n.dir, True)
        if not args:
            n.run(setup)
            self.assertEqual(n.value(f'.srv("127.0.0.1:{self.ports[name]}")'), "1")
        self.nodes[name] = n
        return n

    def start_cluster(self):
        self.ports = {name: free_port() for name in self.NAMES}
        self.nodes = {}
        for name in self.NAMES:
            self.start_node(name)

    def restart(self, name, args=()):
        """Start the node again in its directory and on its port."""
        old = self.nodes[name]
        old.kill()
        return self.start_node(name, old.dir, args)

    def adds(self):
        return ";".join(f'mgmt.add("{n}";"127.0.0.1:{self.ports[n]}")' for n in self.NAMES)


class NativeManagement(Cluster):
    """The cluster behind a native management process, with a desktop KlongPy client connected to it."""

    def setUp(self):
        self.start_cluster()
        self.m = Node()
        self.addCleanup(self.m.stop)
        self.addCleanup(shutil.rmtree, self.m.dir, True)
        self.m.run('.l("mgmt")')
        self.m.run(self.adds())
        self.mport = free_port()
        self.assertEqual(self.m.value(f'.srv("127.0.0.1:{self.mport}")'), "1")

    def desktop(self):
        """KlongPy as the desktop helper: mgmt::.cli("host:port"), then mgmt(:mgmt.call,...)."""
        klong = self.repl()
        self.kg(klong, f'mgmt::.cli("127.0.0.1:{self.mport}")')
        return klong

    def state(self, name):
        return self.m.value(f'(mgmt.nodes?"{name}")?"state"')

    def states(self):
        return [self.state(n) for n in self.NAMES]


# ----------------------------------------------------------------------------------------------------------
# registry and relay
# ----------------------------------------------------------------------------------------------------------

@NEED_NATIVE
class TestRegistryAndRelay(NativeManagement):
    def test_registry(self):
        m = self.m
        self.assertEqual(m.value("mgmt.trap,#mgmt.names()"), "[1 3]")        # this runtime has .try
        self.assertEqual(m.value("mgmt.names()"), '["n1" "n2" "n3"]')
        self.assertEqual(self.states(), ['"up"'] * 3)
        self.assertEqual(m.value('(mgmt.nodes?"n2")?"addr"'), f'"127.0.0.1:{self.ports["n2"]}"')
        self.assertEqual(m.value('((mgmt.nodes?"n2")?"ver")?"platform"'), '"host"')
        self.assertEqual(m.value('((mgmt.nodes?"n1")?"seen")>0'), "1")
        self.assertEqual(m.value('{((x@1)?"state"),((x@1)?"age")<5}\'mgmt.status()'), '[["up" 1] ["up" 1] ["up" 1]]')
        # connecting gave each node three names and nothing else of the library
        self.assertEqual(self.nodes["n1"].value("mgmt.r,($mgmt.run),,$mgmt.job"), '[0 ":dyad" ":triad"]')
        self.assertEqual(self.nodes["n1"].value("mgmt.call"), ":mgmt.call")
        # a name that is added again gets the new address, and its old link is closed
        self.nodes["n1"].run("opened::0;closed::0;.srv.o::{opened::opened+1};.srv.c::{closed::closed+1}")
        self.assertEqual(m.value(f'mgmt.add("n1";"localhost:{self.ports["n1"]}");mgmt.names()'), '["n1" "n2" "n3"]')
        self.assertEqual(m.value('mgmt.call("n1";[:sq]@0;[4])'), "16")
        self.assertTrue(wait_until(lambda: self.nodes["n1"].value("opened,closed") == "[1 1]"))

    def test_relay_of_values(self):
        m = self.m
        self.assertEqual(m.value('mgmt.call("n1";[:sq]@0;[7])'), "49")       # a symbol: the node's function
        self.assertEqual(m.value('mgmt.call("n2";"sq";[7])'), "49")          # a string: source the node evaluates
        self.assertEqual(m.value('mgmt.call("n3";{x+y};[3 4])'), "7")        # a function: sent as its source
        self.assertEqual(m.value('mgmt.call("n1";[:add]@0;[3 4])'), "7")
        self.assertEqual(m.value('mgmt.call("n1";[:nil]@0;[])'), "42")
        self.assertEqual(m.value('mgmt.call("n1";"answer";[])'), "42")       # with no arguments: the value
        self.assertEqual(m.value('mgmt.call("n1";"1+2";[])'), "3")
        self.assertEqual(m.value('mgmt.call("n2";[:total]@0;,!101)'), "5050")
        for lit in ["5", "-7", "1.5", "0ca", '"abc"', '""', '"héllo 世"', ":abc", "[1 2 3]", "[1.5 2.5]", "[:a :b]",
                    '["ab" "cd"]', '[1 :a "s"]', "[[1 2] [3 4]]", "[[1 2] [3]]", '[1 ["two" [3.5 :four]]]', "[]",
                    ":{[:a 1] [:b 2]}", ':{[1 "one"] [2 "two"]}', "[1 5]", "0", '[0 "type" "not an error"]', '["" 5]',
                    "0.1", "[0.1 0.2]"]:
            self.assertEqual(m.value(f'v::{lit};mgmt.call("n3";[:idf]@0;,v)~v'), "1", lit)
        self.assertEqual(m.value('big::!200000;r::mgmt.call("n1";[:idf]@0;,big);(r~big),#r', 60), "[1 200000]")
        # state held on a node stays there
        m.run('mgmt.call("n1";"counter::10";[])')
        self.assertEqual(m.value('mgmt.call("n1";"counter::counter+1";[])'), "11")
        self.assertEqual(self.nodes["n1"].value("counter"), "11")
        self.assertEqual(self.nodes["n2"].value("counter"), ":counter")

    def test_errors_of_a_node_keep_their_kind_and_name_the_node(self):
        for kind, _ in KINDS:
            e = self.native_error(self.m, f'mgmt.call("n2";[:raise]@0;,"{kind}")', kind)
            self.assertEqual(e.message, "n2: boom")
        for request, kind, text in NATURAL:
            e = self.native_error(self.m, f'mgmt.call("n3";{request};[])', kind, text)
            self.assertTrue(e.message.startswith("n3: "), e.message)
        e = self.native_error(self.m, 'mgmt.call("n1";[:nosuch]@0;[1])', "name", "nosuch")
        self.assertTrue(e.message.startswith("n1: "), e.message)
        # the links are as good as before
        self.assertEqual(self.states(), ['"up"'] * 3)
        self.assertEqual(self.m.value('mgmt.call("n2";[:sq]@0;[5])'), "25")

    def test_errors_of_the_management_node_are_its_own(self):
        e = self.native_error(self.m, 'mgmt.call("n9";[:sq]@0;[5])', "name")
        self.assertEqual(e.message, "mgmt: no such node: n9")
        e = self.native_error(self.m, 'mgmt.timeout("n9";1)', "name")
        self.assertEqual(e.message, "mgmt: no such node: n9")

    def test_a_link_that_fails_is_a_system_error_naming_the_node(self):
        self.nodes["n2"].kill()
        e = self.native_error(self.m, 'mgmt.call("n2";[:sq]@0;[5])', "system")
        self.assertTrue(e.message.startswith("n2: "), e.message)
        self.assertEqual(self.states(), ['"up"', '"down"', '"up"'])
        e = self.native_error(self.m, 'mgmt.call("n2";[:sq]@0;[5])', "system", "cannot connect")
        self.assertTrue(e.message.startswith("n2: "), e.message)
        self.assertEqual(self.m.value('mgmt.call("n3";[:sq]@0;[5])'), "25")
        # the node comes back: the next call connects again
        self.restart("n2")
        self.assertEqual(self.m.value('mgmt.call("n2";[:sq]@0;[5])'), "25")
        self.assertEqual(self.states(), ['"up"'] * 3)

    def test_a_call_that_times_out_is_a_system_error_naming_the_node(self):
        self.assertEqual(self.m.value('mgmt.timeout("n3";0.5)'), "0.5")
        t0 = time.monotonic()
        e = self.native_error(self.m, 'mgmt.call("n3";[:wait]@0;[3])', "system", "timed out")
        self.assertLess(time.monotonic() - t0, 2)
        self.assertTrue(e.message.startswith("n3: "), e.message)
        self.assertEqual(self.states(), ['"up"', '"up"', '"down"'])
        self.assertEqual(self.m.value('mgmt.call("n1";[:wait]@0;[0.7])'), "0.7")      # n1 still has 30 seconds
        # once n3 is done waiting it answers again
        self.assertEqual(self.m.value('mgmt.timeout("n3";10);mgmt.call("n3";[:sq]@0;[5])'), "25")

    def test_hooks_a_board_may_provide(self):
        e = self.native_error(self.m, 'mgmt.reset("n1")', "name")
        self.assertEqual(e.message, "mgmt: this board has no node.reset")
        self.native_error(self.m, 'mgmt.power("n1";0)', "name", "this board has no node.power")
        self.m.run("did::[];node.reset::{did::did,,x;1};node.power::{did::did,,x,,y;y}")
        self.assertEqual(self.m.value('mgmt.reset("n1")'), "1")
        self.assertEqual(self.m.value('mgmt.power("n2";0)'), "0")
        self.assertEqual(self.m.value("did"), '["n1" ["n2" 0]]')
        self.native_error(self.m, 'mgmt.reset("n9")', "name", "mgmt: no such node: n9")


# ----------------------------------------------------------------------------------------------------------
# fan-out
# ----------------------------------------------------------------------------------------------------------

@NEED_NATIVE
class TestFanOut(NativeManagement):
    def test_all_nodes_answer(self):
        m = self.m
        self.assertEqual(m.value("mgmt.all([:sq]@0;[7])"), ':{["n1" 49] ["n2" 49] ["n3" 49]}')
        self.assertEqual(m.value('mgmt.each(["n3" "n1"];"add";[3 4])'), ':{["n3" 7] ["n1" 7]}')
        self.assertEqual(m.value('mgmt.all({x,.pc()>0};,"v")'), ':{["n1" ["v" 1]] ["n2" ["v" 1]] ["n3" ["v" 1]]}')
        self.assertEqual(m.value('mgmt.all("answer";[])'), ':{["n1" 42] ["n2" 42] ["n3" 42]}')
        self.assertEqual(m.value("mgmt.each([];[:sq]@0;[7])"), ":{}")
        self.assertEqual(m.value('r::mgmt.all([:total]@0;,!1001);r?"n2"'), "500500")
        # each node worked on its own state
        for i, name in enumerate(self.NAMES):
            self.nodes[name].run(f"base::{i * 100}")
        self.assertEqual(m.value('mgmt.all("base+1";[])'), ':{["n1" 1] ["n2" 101] ["n3" 201]}')

    def test_the_nodes_work_at_the_same_time(self):
        t0 = time.monotonic()
        self.assertEqual(self.m.value("mgmt.all([:wait]@0;[0.6])"), ':{["n1" 0.6] ["n2" 0.6] ["n3" 0.6]}')
        self.assertLess(time.monotonic() - t0, 1.5)          # three waits of 0.6 s in a row would be 1.8 s

    def test_an_error_on_one_node_is_its_error_value(self):
        self.nodes["n2"].run('work::{x+"a"}')
        r = self.m.value("r::mgmt.all([:work]@0;[21])")
        self.assertEqual(r, ':{["n1" 42] ["n2" :{["error" "type"] ["node" "n2"] ["message" '
                            '"n2: cannot apply + (Plus) to integer and string"]}] ["n3" 42]}')
        for kind, _ in KINDS:
            self.nodes["n3"].run(f'work::{{.err("{kind}";"boom")}}')
            self.assertEqual(self.m.value('((mgmt.all([:work]@0;[21]))?"n3")?"error"'), f'"{kind}"')
        self.assertEqual(self.m.value('(mgmt.each(["n1" "n9"];[:sq]@0;[2]))?"n9"'),
                         ':{["error" "name"] ["node" "mgmt"] ["message" "mgmt: no such node: n9"]}')
        self.assertEqual(self.states(), ['"up"'] * 3)

    def test_one_node_killed_and_one_slow(self):
        m = self.m
        m.run("mgmt.start(60)")                              # the check runs (and is not due again in this test)
        self.nodes["n3"].run("work::{wait(4);x}")
        m.run('mgmt.timeout("n3";1)')
        self.nodes["n2"].kill()
        t0 = time.monotonic()
        m.run("r::mgmt.all([:work]@0;[21])")
        took = time.monotonic() - t0
        self.assertEqual(m.value('r?"n1"'), "42")
        self.assertEqual(m.value('((r?"n2")?"error"),,(r?"n2")?"node"'), '["system" "n2"]')
        self.assertEqual(m.value('((r?"n3")?"error"),,(r?"n3")?"node"'), '["system" "n3"]')
        self.assertIn("n3: call timed out", m.value('(r?"n3")?"message"'))
        self.assertTrue(m.value('(r?"n2")?"message"').startswith('"n2: '))
        # the dead node cost nothing and the slow one its own second; n1's 30 seconds were never waited for
        self.assertGreater(took, 0.9)
        self.assertLess(took, 2.0)
        self.assertEqual(self.states(), ['"up"', '"down"', '"down"'])
        # the same again: both are without a link, which the check will mend; nothing is waited for
        t0 = time.monotonic()
        m.run("r::mgmt.all([:work]@0;[4])")
        self.assertLess(time.monotonic() - t0, 0.5)
        self.assertEqual(m.value('(r?"n1"),(,(r?"n2")?"error"),,(r?"n3")?"error"'), '[8 "system" "system"]')
        self.assertEqual(m.value('(r?"n2")?"message"'), '"n2: not connected"')
        # with no check running the fan-out connects by itself: n2 is still gone, then it is back
        m.run("mgmt.stop()")
        self.assertIn("n2: cannot connect", m.value('((mgmt.each(["n1" "n2"];[:work]@0;[4]))?"n2")?"message"'))
        self.restart("n2")
        self.assertEqual(m.value('mgmt.each(["n1" "n2"];[:work]@0;[4])'), ':{["n1" 8] ["n2" 8]}')

    def test_two_slow_nodes_cost_one_timeout_not_two(self):
        m = self.m
        for name in ("n1", "n3"):
            self.nodes[name].run("work::{wait(4);x}")
            m.run(f'mgmt.timeout("{name}";1)')
        t0 = time.monotonic()
        m.run("r::mgmt.all([:work]@0;[21])")
        took = time.monotonic() - t0
        self.assertEqual(m.value('(,(r?"n1")?"error"),(r?"n2"),,(r?"n3")?"error"'), '["system" 42 "system"]')
        self.assertLess(took, 1.8)

    def test_a_node_with_a_longer_timeout_is_waited_for(self):
        m = self.m
        self.nodes["n1"].run("work::{wait(1.2);x}")
        self.nodes["n3"].run("work::{wait(4);x}")
        m.run('mgmt.timeout("n1";3);mgmt.timeout("n3";0.5)')
        t0 = time.monotonic()
        m.run("r::mgmt.all([:work]@0;[21])")
        took = time.monotonic() - t0
        self.assertEqual(m.value('(r?"n1"),(r?"n2"),,(r?"n3")?"error"'), '[21 42 "system"]')
        self.assertLess(took, 2.2)


# ----------------------------------------------------------------------------------------------------------
# liveness
# ----------------------------------------------------------------------------------------------------------

@NEED_NATIVE
class TestLiveness(NativeManagement):
    INTERVAL = 0.2

    def test_a_killed_node_is_down_and_up_again_after_it_restarts(self):
        m = self.m
        self.assertEqual(m.value(f"mgmt.start({self.INTERVAL})"), "1")
        time.sleep(3 * self.INTERVAL)
        self.assertEqual(self.states(), ['"up"'] * 3)
        seen = float(m.value('(mgmt.nodes?"n1")?"seen"'))
        self.nodes["n2"].kill()
        t0 = time.monotonic()
        self.assertTrue(wait_until(lambda: self.state("n2") == '"down"', 5))
        self.assertLess(time.monotonic() - t0, 2 * self.INTERVAL + 0.5)     # within the ping interval, and some slack
        self.assertEqual(self.states(), ['"up"', '"down"', '"up"'])
        self.assertEqual(m.value('(mgmt.nodes?"n2")?"h"'), "0")
        # down stays down, the others keep answering, and a fan-out says which is which
        time.sleep(4 * self.INTERVAL)
        self.assertEqual(self.states(), ['"up"', '"down"', '"up"'])
        self.assertGreater(float(m.value('(mgmt.nodes?"n1")?"seen"')), seen)
        self.assertEqual(m.value('r::mgmt.all([:sq]@0;[3]);(r?"n1"),((r?"n2")?"error"),,r?"n3"'), '[9 "system" 9]')
        self.assertEqual(m.value('((mgmt.status())?"n2")?"state"'), '"down"')
        # the node comes back on its address: connected again at a following check
        self.restart("n2")
        self.assertTrue(wait_until(lambda: self.state("n2") == '"up"', 5))
        self.assertEqual(m.value('mgmt.call("n2";[:sq]@0;[6])'), "36")
        self.assertEqual(m.value("mgmt.all([:sq]@0;[3])"), ':{["n1" 9] ["n2" 9] ["n3" 9]}')
        # errors of the check are caught, not logged (the runtime itself logs a probe that was out when the link fell)
        self.assertEqual(m.run("1"), ["1"])
        self.assertEqual([x for x in m.log if "remote call failed" not in x], [])
        # stopped: a node that dies now is not noticed until it is used
        self.assertEqual(m.value("mgmt.stop()"), "1")
        self.nodes["n3"].kill()
        time.sleep(3 * self.INTERVAL)
        self.assertEqual(self.state("n3"), '"up"')
        self.assertEqual(m.value("mgmt.stop()"), "0")

    def test_a_node_that_is_busy_is_down_until_it_answers(self):
        m = self.m
        m.run(f"mgmt.start({self.INTERVAL})")
        self.nodes["n3"].send("wait(1.5)")                   # busy at its own prompt: it serves nobody meanwhile
        self.assertTrue(wait_until(lambda: self.state("n3") == '"down"', 5))
        self.assertEqual(self.states()[:2], ['"up"', '"up"'])
        self.assertTrue(wait_until(lambda: self.state("n3") == '"up"', 5))
        self.nodes["n3"].output()
        self.assertEqual(m.value('mgmt.call("n3";[:sq]@0;[6])'), "36")

    def test_a_node_added_before_it_is_up(self):
        m = self.m
        port = free_port()
        self.ports["n4"] = port
        self.assertEqual(m.value(f'mgmt.add("n4";"127.0.0.1:{port}")'), '"n4"')
        self.assertEqual(self.state("n4"), '"down"')
        m.run(f"mgmt.start({self.INTERVAL})")
        time.sleep(2 * self.INTERVAL)
        self.assertEqual(self.state("n4"), '"down"')
        self.start_node("n4")
        self.assertTrue(wait_until(lambda: self.state("n4") == '"up"', 5))
        self.assertEqual(m.value('(mgmt.all([:sq]@0;[2]))?"n4"'), "4")


# ----------------------------------------------------------------------------------------------------------
# deploy
# ----------------------------------------------------------------------------------------------------------

@NEED_NATIVE
class TestDeploy(NativeManagement):
    def test_a_file_is_stored_and_the_node_uses_it(self):
        m = self.m
        text = 'cube::{x*x*x}\ngreet::{"hi ",x}\n'
        m.run('src::"cube::{x*x*x}",(:#10),"greet::{""hi "",x}",:#10')
        self.assertEqual(m.value('mgmt.put("n1";"lib1.kg";src)'), str(len(text)))
        with open(os.path.join(self.nodes["n1"].dir, "lib1.kg"), encoding="utf-8") as f:
            self.assertEqual(f.read(), text)
        self.assertFalse(os.path.exists(os.path.join(self.nodes["n2"].dir, "lib1.kg")))
        m.run('mgmt.load("n1";"lib1")')
        self.assertEqual(m.value('mgmt.call("n1";[:cube]@0;[3])'), "27")
        self.assertEqual(m.value('mgmt.call("n1";[:greet]@0;,"bob")'), '"hi bob"')
        # the node's output still goes to its console
        self.assertEqual(self.nodes["n1"].run('.p("here")'), ["here", '"here"'])
        # a file that cannot be written is the node's system error
        e = self.native_error(m, 'mgmt.put("n1";"no-such-dir/f.kg";src)', "system")
        self.assertTrue(e.message.startswith("n1: "), e.message)
        self.assertEqual(self.nodes["n1"].run('.p("here")'), ["here", '"here"'])

    def test_boot_kg_is_run_when_the_node_starts_again(self):
        m = self.m
        port = self.ports["n2"]
        boot = f'deployed::{{x+1000}}\nmarker::"from boot.kg"\n.srv("127.0.0.1:{port}")\n'
        klong = self.desktop()
        klong["boot"] = boot
        self.assertEqual(self.kg(klong, 'mgmt(:mgmt.boot,"n2",,boot)'), len(boot))
        with open(os.path.join(self.nodes["n2"].dir, "boot.kg"), encoding="utf-8") as f:
            self.assertEqual(f.read(), boot)
        self.native_error(m, 'mgmt.call("n2";[:deployed]@0;[1])', "name")      # stored, not loaded
        # the node starts again and runs its boot.kg, as a board does at power-up (the host binary is given the file)
        self.restart("n2", args=("boot.kg",))
        self.assertTrue(wait_until(lambda: self._answers("n2"), 10))
        self.assertEqual(m.value('mgmt.call("n2";[:deployed]@0;[1])'), "1001")
        self.assertEqual(self.kg(klong, 'mgmt(:mgmt.call,"n2","marker",,[])'), "from boot.kg")
        self.assertEqual(m.value('(mgmt.all("marker";[]))?"n2"'), '"from boot.kg"')

    def _answers(self, name):
        try:
            return self.m.value(f'mgmt.call("{name}";"1+1";[])') == "2"
        except NativeError:
            return False


# ----------------------------------------------------------------------------------------------------------
# the desktop: KlongPy through .cli
# ----------------------------------------------------------------------------------------------------------

@NEED_NATIVE
class TestDesktop(NativeManagement):
    def setUp(self):
        super().setUp()
        self.klong = self.desktop()

    def k(self, text, timeout=TIMEOUT):
        return self.kg(self.klong, text, timeout)

    def test_calls_on_a_named_node(self):
        self.assertEqual(self.k('mgmt(:mgmt.call,"n1",:sq,,[7])'), 49)
        self.assertEqual(self.k('mgmt(:mgmt.call,"n2","add",,[3 4])'), 7)
        self.assertEqual(self.k('mgmt(:mgmt.call,"n3","1+2",,[])'), 3)
        self.assertEqual(self.k('mgmt(:mgmt.call,"n3",{x*10},,[4])'), 40)
        self.klong["data"] = np.arange(1000) * 0.5
        self.assertEqual(self.k('mgmt(:mgmt.call,"n1",:total,,,data)'), float(np.sum(np.arange(1000) * 0.5)))
        np.testing.assert_array_equal(as_np(self.k('mgmt(:mgmt.call,"n1",:idf,,,data)')), np.arange(1000) * 0.5)
        self.assertSame(self.k('mgmt(:mgmt.call,"n1",:idf,,,[1 :a "s"])'), [1, S("a"), "s"])
        self.assertSame(self.k('mgmt(:mgmt.call,"n1",:idf,,,:{[:a 1] [:b 2]})'), {S("a"): 1, S("b"): 2})

    def test_an_array_goes_to_every_node_and_the_results_come_back_as_a_dictionary(self):
        self.klong["data"] = np.arange(100)
        for i, name in enumerate(self.NAMES):
            self.nodes[name].run(f"scale::{{x*{i + 1}}}")
        r = self.k('mgmt(:mgmt.all,"{+/scale(x)}",,,data)')
        self.assertSame(r, {"n1": 4950, "n2": 9900, "n3": 14850})
        r = self.k('mgmt(:mgmt.all,:scale,,,data)')
        self.assertEqual(list(r), ["n1", "n2", "n3"])
        for i, name in enumerate(self.NAMES):
            np.testing.assert_array_equal(as_np(r[name]), np.arange(100) * (i + 1))
        r = self.k('mgmt(:mgmt.each,(,["n3" "n1"]),:sq,,[4])')
        self.assertSame(r, {"n3": 16, "n1": 16})

    def test_each_kind_of_error_arrives_as_its_class_with_the_node_named(self):
        for kind, cls in KINDS:
            self.klong["kind"] = kind
            with self.assertRaises(cls) as cm:
                self.k('mgmt(:mgmt.call,"n2",:raise,,,kind)')
            self.assertEqual(cm.exception.kind, kind)
            self.assertIn("n2: boom", str(cm.exception))
        with self.assertRaises(KlongTypeError) as cm:
            self.k('mgmt(:mgmt.call,"n3",:fail,,[1])')
        self.assertIn("n3: cannot apply + (Plus) to integer and string", str(cm.exception))
        with self.assertRaises(KlongNameError) as cm:
            self.k('mgmt(:mgmt.call,"n1",:nosuch,,[1])')
        self.assertIn("n1: ", str(cm.exception))
        # the management node's own
        with self.assertRaises(KlongNameError) as cm:
            self.k('mgmt(:mgmt.call,"n9",:sq,,[1])')
        self.assertIn("mgmt: no such node: n9", str(cm.exception))
        self.assertEqual(self.k('mgmt(:mgmt.call,"n2",:sq,,[5])'), 25)       # every link is as it was

    def test_fan_out_with_a_dead_and_a_slow_node(self):
        self.nodes["n3"].run("work::{wait(4);x}")
        self.k('mgmt(:mgmt.timeout,"n3",,1)')
        self.nodes["n2"].kill()
        t0 = time.monotonic()
        r = self.k('mgmt(:mgmt.all,:work,,[21])')
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertEqual(r["n1"], 42)
        for name in ("n2", "n3"):
            self.assertSame(r[name]["error"], "system")
            self.assertSame(r[name]["node"], name)
            self.assertTrue(r[name]["message"].startswith(name + ": "), r[name])
        self.assertIn("timed out", r["n3"]["message"])
        with self.assertRaises(KlongSystemError) as cm:
            self.k('mgmt(:mgmt.call,"n2",:sq,,[1])')
        self.assertIn("n2: ", str(cm.exception))
        self.assertEqual(self.k('mgmt(:mgmt.call,"n1",:sq,,[5])'), 25)

    def test_status_and_liveness_seen_from_the_desktop(self):
        self.k("mgmt(:mgmt.start,,0.2)")
        st = self.k('mgmt("mgmt.status()")')
        self.assertEqual(list(st), ["n1", "n2", "n3"])
        self.assertEqual([st[n]["state"] for n in self.NAMES], ["up"] * 3)
        self.assertEqual(st["n1"]["ver"]["platform"], "host")
        self.assertEqual(st["n1"]["addr"], f"127.0.0.1:{self.ports['n1']}")
        self.nodes["n1"].kill()
        self.assertTrue(wait_until(lambda: self.k('mgmt("mgmt.status()")')["n1"]["state"] == "down", 5))
        st = self.k('mgmt("mgmt.status()")')
        self.assertEqual([st[n]["state"] for n in self.NAMES], ["down", "up", "up"])
        self.restart("n1")
        self.assertTrue(wait_until(lambda: self.k('mgmt("mgmt.status()")')["n1"]["state"] == "up", 5))
        self.assertEqual(self.k('mgmt(:mgmt.call,"n1",:sq,,[5])'), 25)

    def test_deploy_from_the_desktop(self):
        self.klong["src"] = "dbl::{x*2}\n"
        self.assertEqual(self.k('mgmt(:mgmt.put,"n3","dbl.kg",,src)'), 11)
        self.k('mgmt(:mgmt.load,"n3",,"dbl")')
        self.assertEqual(self.k('mgmt(:mgmt.call,"n3",:dbl,,[21])'), 42)

    def test_two_desktops(self):
        other = self.desktop()
        self.assertEqual(self.k('mgmt(:mgmt.call,"n1",:sq,,[3])'), 9)
        self.assertEqual(self.kg(other, 'mgmt(:mgmt.call,"n1",:sq,,[4])'), 16)
        self.assertSame(self.kg(other, 'mgmt(:mgmt.all,:sq,,[2])'), {"n1": 4, "n2": 4, "n3": 4})
        self.assertSame(self.k('mgmt(:mgmt.all,:sq,,[3])'), {"n1": 9, "n2": 9, "n3": 9})


# ----------------------------------------------------------------------------------------------------------
# KlongPy as the management process
# ----------------------------------------------------------------------------------------------------------

@NEED_NATIVE
class TestKlongPyAsManagementNode(Cluster):
    """The same file loaded by KlongPy, which has no .try and no .err: what the header says works there."""

    def setUp(self):
        self.start_cluster()
        self.klong = self.repl()
        self.k('.l("mgmt")')
        self.k(self.adds())

    def k(self, text, timeout=TIMEOUT):
        """Evaluate on the Klong loop, as the kgpy prompt, a served request and the liveness check all do."""
        box = concurrent.futures.Future()

        def run():
            try:
                box.set_result(self.klong(text))
            except BaseException as e:  # noqa: BLE001 - raised in the test thread
                box.set_exception(e)
        self.klong[".system"]["klongloop"].call_soon_threadsafe(run)
        return box.result(timeout)

    def state(self, name):
        return self.k(f'(mgmt.nodes?"{name}")?"state"')

    def test_registry_and_relay(self):
        self.assertEqual(self.k("mgmt.trap"), 0)                                   # KlongPy has no .try
        self.assertEqual(list(self.k("mgmt.names()")), ["n1", "n2", "n3"])
        self.assertEqual([self.state(n) for n in self.NAMES], ["down"] * 3)        # not connected until used
        self.assertEqual(self.k('mgmt.call("n1";:sq;[7])'), 49)
        self.assertEqual(self.k('mgmt.call("n2";"add";[3 4])'), 7)
        self.assertEqual(self.k('mgmt.call("n3";"1+2";[])'), 3)
        self.assertEqual(self.k('mgmt.call("n3";{x*10};[4])'), 40)
        self.klong["data"] = np.arange(1000) * 0.5
        np.testing.assert_array_equal(as_np(self.k('mgmt.call("n1";:idf;,data)')), np.arange(1000) * 0.5)
        self.assertEqual([self.state(n) for n in self.NAMES], ["up"] * 3)
        st = self.k("mgmt.status()")
        self.assertEqual(st["n1"]["ver"]["platform"], "host")

    def test_errors_keep_their_kind_but_not_the_name(self):
        for kind, cls in KINDS:
            self.klong["kind"] = kind
            with self.assertRaises(cls) as cm:
                self.k('mgmt.call("n2";:raise;,kind)')
            self.assertEqual(cm.exception.kind, kind)
            self.assertIn("boom", str(cm.exception))
            self.assertNotIn("n2: ", str(cm.exception))
        self.assertEqual(self.k('mgmt.call("n2";:sq;[5])'), 25)
        with self.assertRaises(KlongNameError) as cm:
            self.k('mgmt.call("n9";:sq;[5])')
        self.assertIn("mgmt.error", str(cm.exception))

    def test_fan_out(self):
        self.k('mgmt.call("n1";"1";[]);mgmt.call("n2";"1";[]);mgmt.call("n3";"1";[])')
        self.klong["data"] = np.arange(100)
        self.assertSame(self.k("mgmt.all(:total;,data)"), {"n1": 4950, "n2": 4950, "n3": 4950})
        t0 = time.monotonic()
        r = self.k("mgmt.all(:wait;[0.6])")                  # (0.6 as the backend holds it: 32 bits on torch)
        self.assertEqual({n: round(float(v), 5) for n, v in r.items()}, {"n1": 0.6, "n2": 0.6, "n3": 0.6})
        self.assertLess(time.monotonic() - t0, 1.5)
        # an error on a node is an error value: the node caught it
        self.nodes["n2"].run('work::{x+"a"}')
        r = self.k("mgmt.all(:work;[21])")
        self.assertEqual((r["n1"], r["n3"]), (42, 42))
        self.assertSame(r["n2"], {"error": "type", "node": "n2",
                                  "message": "n2: cannot apply + (Plus) to integer and string"})
        # a link that fails is not caught here: the whole call fails
        self.nodes["n3"].kill()
        with self.assertRaises(KlongError):
            self.k("mgmt.all(:sq;[2])")

    def test_a_node_that_was_never_connected_is_an_error_value(self):
        r = self.k("mgmt.all(:sq;[2])")
        self.assertSame(r["n1"], {"error": "system", "node": "n1", "message": "n1: not connected"})

    def test_deploy(self):
        self.klong["src"] = "dbl::{x*2}\n"
        self.assertEqual(self.k('mgmt.put("n3";"dbl.kg";src)'), 11)
        with open(os.path.join(self.nodes["n3"].dir, "dbl.kg"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "dbl::{x*2}\n")
        self.k('mgmt.load("n3";"dbl")')
        self.assertEqual(self.k('mgmt.call("n3";:dbl;[21])'), 42)

    def test_liveness_marks_a_killed_node_down(self):
        self.k('mgmt.call("n1";"1";[]);mgmt.call("n2";"1";[]);mgmt.call("n3";"1";[])')
        self.assertEqual(self.k("mgmt.start(0.2)"), 1)
        time.sleep(0.6)
        self.assertEqual([self.state(n) for n in self.NAMES], ["up"] * 3)
        self.assertGreater(self.k('(mgmt.nodes?"n1")?"seen"'), 0)
        self.nodes["n2"].kill()
        self.assertTrue(wait_until(lambda: self.state("n2") == "down", 5))
        time.sleep(0.6)
        self.assertEqual([self.state(n) for n in self.NAMES], ["up", "down", "up"])       # and the timer lives on
        # not connected again by the check (see the header); by hand it is
        self.restart("n2")
        time.sleep(0.6)
        self.assertEqual(self.state("n2"), "down")
        self.k('mgmt.open("n2")')
        self.assertEqual(self.state("n2"), "up")
        self.assertEqual(self.k('mgmt.call("n2";:sq;[5])'), 25)
        self.assertEqual(self.k("mgmt.stop()"), 1)

    def test_a_desktop_through_a_klongpy_management_node(self):
        self.k('mgmt.call("n1";"1";[]);mgmt.call("n2";"1";[]);mgmt.call("n3";"1";[])')
        port = free_port()
        self.assertEqual(self.k(f".srv({port})"), 1)
        client = self.native(f"m::.cli({port})")
        self.assertEqual(client.value('m([:mgmt.call "n1" :sq [7]])'), "49")
        self.assertEqual(client.value('m([:mgmt.all :sq [3]])'), ':{["n1" 9] ["n2" 9] ["n3" 9]}')
        self.native_error(client, 'm([:mgmt.call "n1" :fail [1]])', "type", "cannot apply + (Plus)")


if __name__ == "__main__":
    unittest.main()
