"""Python references stay alive through Klong values and are collected afterward."""
from concurrent.futures import ThreadPoolExecutor
import gc
from threading import Event
from weakref import ref

import numpy as np
import pytest

from klongpy import KlongInterpreter
from klongpy.native_ext import native
from klongpy.types import KGLambda, KGSym


class Payload:
    pass


def assert_collected(*references):
    gc.collect()
    assert all(reference() is None for reference in references)


def test_temporary_hosted_sessions_are_collectible():
    references = []
    for _ in range(6):
        engine = KlongInterpreter(backend="numpy")
        engine("data::!2000")
        references.append(ref(engine))
        del engine
    assert_collected(*references)


def test_custom_klambda_keeps_its_originating_context():
    class ReadBias(KGLambda):
        def __init__(self, name):
            self.name = name

        def get_arity(self):
            return 1

        def __call__(self, klong, ctx):
            return klong(self.name) + ctx[KGSym("x")]

    source = KlongInterpreter()
    target = KlongInterpreter()
    source("bias::40;d:::{}")
    source["read"] = ReadBias("bias")
    assert source("read(2)") == 42
    source('d,"read",,read')
    target["d"] = source["d"]
    target('bias::100;run()::{[read];read::d@"read";read(2)}')
    source_ref = ref(source)
    del source
    gc.collect()
    assert target("run()") == 42
    del target["d"]
    assert_collected(source_ref)


def test_discarded_consumers_do_not_follow_live_source():
    source = KlongInterpreter(backend="numpy")
    source('bias::10;f::{x+bias};d:::{;"n";5};t::.table(:{;"n";[1 2]})')
    references = []
    for _ in range(6):
        consumer = KlongInterpreter(backend="numpy")
        consumer("bias::20")
        consumer["f"], consumer["d"], consumer["t"] = source["f"], source["d"], source["t"]
        assert consumer('f(d@"n")') == 25
        references.append(ref(consumer))
        del consumer
    assert_collected(*references)
    assert source("f(1)") == 11


def test_callbacks_added_after_transfer_follow_shared_values():
    source = KlongInterpreter(backend="numpy")
    consumer = KlongInterpreter(backend="numpy")
    run = 'run(key)::{[cb];cb::d@key;cb()}'
    source('bias::10;d:::{;"n";5};t::.table(:{;"n";[1 2]});' + run)
    consumer("bias::20;" + run)
    consumer["d"], consumer["t"] = source["d"], source["t"]

    def independent():
        return 42

    def from_source(engine=source):
        return engine("bias")

    def from_consumer(engine=consumer):
        return engine("bias")

    source["d"]["independent"] = independent
    source["d"]["fromSource"] = from_source
    consumer["d"]["fromConsumer"] = from_consumer
    assert consumer["d"]["independent"] is independent
    assert source["d"]["fromConsumer"] is from_consumer
    assert source('run("fromConsumer")') == 20
    consumer(".insert(t;,[3])")
    np.testing.assert_array_equal(source('t@"n"'), [1, 2, 3])

    source_ref, consumer_ref = ref(source), ref(consumer)
    callbacks = ref(independent), ref(from_source), ref(from_consumer)
    del source, independent, from_source, from_consumer
    gc.collect()
    assert consumer('run("fromSource")') == 10
    del consumer["d"]["fromSource"]
    gc.collect()
    assert consumer('run("independent")') == 42

    del consumer["d"], consumer["t"]
    assert_collected(source_ref, *callbacks)
    assert consumer("bias") == 20
    del consumer
    assert_collected(consumer_ref)


def test_consumer_callback_survives_without_its_interpreter():
    source = KlongInterpreter(backend="numpy")
    consumer = KlongInterpreter(backend="numpy")
    source('d:::{};run(key)::{[cb];cb::d@key;cb()}')
    consumer("bias::20")
    consumer["d"] = source["d"]

    def independent():
        return 42

    def capturing(engine=consumer):
        return engine("bias")

    consumer["d"]["independent"] = independent
    consumer["d"]["capturing"] = capturing
    consumer_ref, independent_ref, capturing_ref = ref(consumer), ref(independent), ref(capturing)
    del consumer, independent, capturing
    gc.collect()
    assert source('run("capturing")') == 20

    del source["d"]["capturing"]
    assert_collected(consumer_ref, capturing_ref)
    assert source('run("independent")') == 42
    del source["d"]["independent"]
    assert_collected(independent_ref)


def test_payload_added_to_nested_dictionary_in_shared_table():
    source = KlongInterpreter(backend="numpy")
    consumer = KlongInterpreter(backend="numpy")
    source("row:::{}")
    table = source('.table(:{;"cells";,row})')
    row = source["row"]
    consumer["table"] = table
    del source["row"], table

    payload = Payload()
    reference = ref(payload)
    row["payload"] = payload
    del payload, row, source
    gc.collect()
    assert reference() is not None
    assert consumer('((table@"cells")@0)@"payload"') is reference()
    del consumer["table"]
    assert_collected(reference)


def test_finalizer_can_move_shared_value_before_releasing_its_payload():
    class Transfer:
        def __init__(self, target, value):
            self.target = target
            self.value = value

        def __del__(self):
            self.target["received"] = self.value

    source = KlongInterpreter(backend="numpy")
    target = KlongInterpreter(backend="numpy")
    view = source(":{}")
    payload = Payload()
    view["payload"] = payload
    transfer = Transfer(target, view)
    source["trigger"] = transfer
    payload_ref, transfer_ref = ref(payload), ref(transfer)
    del payload, transfer, view
    gc.collect()
    assert payload_ref() is not None and transfer_ref() is not None

    del source["trigger"]
    assert_collected(transfer_ref)
    assert payload_ref() is not None
    assert target('received@"payload"') is payload_ref()
    del target["received"]
    assert_collected(payload_ref)


def test_finalizer_can_register_callback_already_held_in_another_session():
    class Transfer:
        def __init__(self, target, view, callback):
            self.target, self.view, self.callback = target, view, callback

        def __del__(self):
            self.target.register("received", self.callback, 255)

    def identity(x):
        return x

    source, target = native().Klong(2), native().Klong(2)
    view = source.eval(":{}")
    view["callback"] = identity
    transfer = Transfer(target, view, identity)
    source["trigger"] = transfer
    callback_ref, transfer_ref = ref(identity), ref(transfer)
    del identity, transfer, view
    gc.collect()
    assert callback_ref() is not None and transfer_ref() is not None

    del source["trigger"]
    assert_collected(transfer_ref)
    assert target["received"] is callback_ref()
    assert target.eval("received(7)") == 7
    del target["received"]
    assert_collected(callback_ref)


def test_resurrected_native_session_retains_old_and_new_values():
    rescued = []

    class Resurrects:
        def __init__(self, engine):
            self.engine = engine

        def __del__(self):
            rescued.append(self.engine)

    engine = native().Klong(0)
    engine["cycle"] = Resurrects(engine)
    del engine
    gc.collect()
    assert len(rescued) == 1
    engine = rescued.pop()
    assert engine["cycle"].engine is engine

    payload = Payload()
    payload_ref = ref(payload)
    engine["new"] = payload
    del payload
    gc.collect()
    assert payload_ref() is not None
    assert engine["new"] is payload_ref()
    del engine
    assert_collected(payload_ref)


@pytest.mark.parametrize("kind", ["projection", "dict", "table"])
def test_escaped_value_keeps_its_native_state(kind):
    engine = native().Klong(0)
    marker = Payload()
    engine["marker"] = marker
    engine.eval('offset::10;f(a;b)::{a+b+offset};d:::{;"value";5};t::.table(:{;"n";[2 3]})')
    escaped = engine.eval({"projection": "f(1;)", "dict": "d", "table": "t"}[kind])
    marker_ref = ref(marker)
    del engine, marker
    gc.collect()
    if kind == "projection":
        assert escaped(31) == 42
    elif kind == "dict":
        escaped["value"] = escaped["value"] + 1
        assert escaped["value"] == 6
    else:
        np.testing.assert_array_equal(escaped["n"], [2, 3])
    del escaped
    assert_collected(marker_ref)


def test_gc_during_interleaved_native_callbacks():
    first, second = native().Klong(0), native().Klong(0)
    payload = Payload()
    payload.first, payload.second = first, second

    def ask_second(x):
        return second.call("askFirst", x)

    def ask_first(x):
        first["payload"] = payload
        gc.collect()
        return first.eval("40") + x

    first.register("askSecond", ask_second, 1)
    second.register("askFirst", ask_first, 1)
    assert first["askSecond"] is ask_second
    assert second["askFirst"] is ask_first
    assert first.eval("askSecond(2)") == 42
    references = ref(payload), ref(ask_first), ref(ask_second)
    del first, second, payload, ask_first, ask_second
    assert_collected(*references)


def test_blocked_callback_does_not_retain_an_unrelated_session():
    waiting, release = Event(), Event()
    long_lived = native().Klong(0)

    def wait_for_release():
        waiting.set()
        assert release.wait(timeout=10)
        return 42

    long_lived.register("waitForRelease", wait_for_release, 0)
    with ThreadPoolExecutor(max_workers=1) as pool:
        call = pool.submit(long_lived.eval, "waitForRelease()")
        try:
            assert waiting.wait(timeout=10)
            engine = KlongInterpreter(backend="numpy")
            payload = Payload()
            payload.engine = engine
            engine["payload"] = payload
            references = ref(engine), ref(payload)
            del engine, payload
            assert_collected(*references)
        finally:
            release.set()
        assert call.result(timeout=10) == 42


def test_host_array_cycle_is_collectible():
    engine = KlongInterpreter(backend="numpy")
    array = np.ma.array([1, 2, 3], mask=[False, True, False])
    array.engine = engine
    engine["payload"] = {"array": array}
    assert engine["payload"]["array"] is array
    references = ref(engine), ref(array)
    del engine, array
    assert_collected(*references)


def test_attached_static_program_keeps_its_runtime():
    engine = native().Klong(2)
    marker = Payload()
    engine["marker"] = marker
    source = 'f::{x+1};.entry("f";"int -> int")'
    engine.run(source)
    program = native().StaticProgram("<static>", source, klong=engine)
    marker_ref = ref(marker)
    del engine, marker
    gc.collect()
    assert program.call_entry("f", [41]) == 42
    del program
    assert_collected(marker_ref)


def test_static_extern_can_capture_its_program():
    holder = Payload()

    def callback(x, keep=holder):
        return x + 1 if keep.program else 0

    source = 'adc::.extern("adc";"int -> int");f::{adc(x)};.entry("f";"int -> int")'
    holder.program = native().StaticProgram("<static>", source, externs={"adc": callback})
    assert holder.program.call_entry("f", [41]) == 42
    references = ref(holder), ref(callback)
    del holder, callback
    assert_collected(*references)
