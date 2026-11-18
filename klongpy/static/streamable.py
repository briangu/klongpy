"""
``kgpy check --streamable`` (docs/hardware.md): which constructs of a
program inside the static profile keep a hardware kernel from taking one record per clock cycle.

    check_streamable(path, int_bits=32, text=None) -> [StaticDiagnostic]

The program is checked and lowered as for kgpy compile c, and the typed IR of every function an entry reaches is searched.
Each finding carries the source span of the construct, the rule id (klongpy.static.rules.STREAM_RULES), what is
wrong and what to change. A program without findings can reach one record per cycle in the interval the tool
reports; it is a statement about the program's dependences, not a promise about the tool.

The rules

    stream-loop    a While loop, or an Iterate whose count is not a constant: the call has no fixed number of cycles
    stream-state   a state variable whose new value is computed from its old value by a multi-cycle operation (a float
                   operation, an integer or fixed-point division by a value that is not a constant): the next record
                   needs the result
    stream-table   a data-dependent index (Index, Gather or Amend) into an array of more than TABLE_MAX items
    stream-block   an entry with more than BLOCK_MAX element operations per call that is not a sliding window
"""
from . import ir
from .compile import compile_program
from .diagnostics import StaticDiagnostic
from .emit_c import HLS_WORK_LIMIT, hls_work, match_window
from .rules import STREAM_RULES

__all__ = ["check_streamable", "TABLE_MAX", "BLOCK_MAX"]

TABLE_MAX = 1024
BLOCK_MAX = HLS_WORK_LIMIT

_FLOAT_OPS = ("add", "sub", "mul", "div", "rem", "pow")


def _reachable(entry):
    """The FnSpecs an entry reaches, the entry's own first."""
    out, todo, seen = [], [entry.fn], set()
    while todo:
        f = todo.pop(0)
        if id(f) in seen:
            continue
        seen.add(id(f))
        out.append(f)
        for n in ir.walk(f.body):
            todo.extend(ir.fnspecs(n))
            if isinstance(n, ir.Call):
                todo.append(n.fn)
    return out


def _reads_state(node, name, memo):
    for n in ir.walk(node):
        if isinstance(n, ir.GlobalRef) and n.state and n.name == name:
            return True
        for g in ir.fnspecs(n) + ([n.fn] if isinstance(n, ir.Call) else []):
            key = (id(g), name)
            if key not in memo:
                memo[key] = False
                memo[key] = _reads_state(g.body, name, memo)
            if memo[key]:
                return True
    return False


def _slow_operation(node, memo):
    """
    The first node under ``node`` (through the functions it calls) that takes several cycles: a float operation, a fixed-point
    or integer division or remainder by a value that is not a constant (a serial divider, 35 cycles in the tool's estimate).
    """
    for n in ir.walk(node):
        if isinstance(n, ir.Binary) and n.op in _FLOAT_OPS and n.a.ty.elem == "float":
            return n, "float " + n.op
        if isinstance(n, ir.Unary) and n.op == "recip":
            return n, "float reciprocal"
        if isinstance(n, ir.Binary) and n.op in ("idiv", "rem", "div") and n.a.ty.elem != "float" and n.b.cval is None:
            return n, f"{'fixed-point' if n.a.ty.elem != 'int' else 'integer'} {n.op} by a value that is not a constant"
        for g in ir.fnspecs(n) + ([n.fn] if isinstance(n, ir.Call) else []):
            if id(g) not in memo:
                memo[id(g)] = None
                memo[id(g)] = _slow_operation(g.body, memo)
            if memo[id(g)] is not None:
                return memo[id(g)]
    return None


def check_streamable(path, int_bits=32, text=None, externs=None):
    """The StaticDiagnostics of the streamable tier. Raises KlongStaticError when the program is outside the profile."""
    compiled = compile_program(path, text=text, int_bits=int_bits, externs=externs)
    program = compiled.program
    out, seen = [], set()

    def add(rule, node, fn, message):
        sp = tuple(node.span) if getattr(node, "span", None) is not None else (fn.span or (0, 0))
        key = (rule, sp, message)
        if key in seen:
            return
        seen.add(key)
        out.append(StaticDiagnostic(rule, sp, fn.source or compiled.report.source, message, STREAM_RULES[rule].hint))

    for e in program.entries:
        fns = _reachable(e)
        window = match_window(program, e)
        for f in fns:
            for n in ir.walk(f.body):
                if isinstance(n, ir.Loop):
                    if n.kind == "while":
                        add("stream-loop", n, f, "a While loop: its number of iterations depends on the data, so the call has "
                                                 "no fixed number of cycles")
                    elif n.count.cval is None or n.count.ty.shape or ir.effects(n.count):
                        add("stream-loop", n, f, "an Iterate whose count is not a compile-time constant: the number of "
                                                 "iterations depends on the data")
                if isinstance(n, (ir.Index, ir.Gather, ir.Amend)) and n.a.ty.shape and n.a.ty.size > TABLE_MAX:
                    idx = n.i if isinstance(n, ir.Index) else n.idx
                    if idx.cval is None:
                        add("stream-table", n, f, f"a data-dependent index into an array of {n.a.ty.size} items "
                                                  f"({n.a.ty}): more than {TABLE_MAX}")
                if isinstance(n, ir.Assign) and isinstance(n.target, ir.GlobalRef) and n.target.state:
                    name = n.target.name
                    if _reads_state(n.value, name, {}):
                        slow = _slow_operation(n.value, {})
                        if slow is not None:
                            add("stream-state", n, f, f"the state {name} is read and rewritten through a {slow[1]}: a multi-cycle "
                                                      "operation inside the loop that carries it from one record to the next")
        work = hls_work(program, e)
        if work > BLOCK_MAX and window is None:
            add("stream-block", e.fn.body, e.fn, f"entry {e.decl.name} does about {work} element operations per call "
                                                 f"(more than {BLOCK_MAX}) and is not a sliding window")
    out.sort(key=lambda d: (d.span[0], d.span[1], d.rule))
    return out
