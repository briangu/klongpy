"""Generate docs/system-functions.md from the eval_sys_* docstrings.

Run from the repository root with a Python that has the optional extras
(duckdb, aiohttp, websockets) installed:

    python -W ignore gen_system_functions.py > docs/system-functions.md

Examples are run while generating; the output shown is what the interpreter
returned. A function without an entry in DESC falls back to the first sentence
of its docstring, and a function without a group goes to "Other".
"""
import contextlib
import importlib
import inspect
import io
import re
import sys
import textwrap
import warnings

warnings.simplefilter("ignore")

from klongpy.repl import create_repl, cleanup_repl
from klongpy.writer import kg_write

MODULES = [
    "klongpy.sys_fn", "klongpy.sys_var", "klongpy.sys_fn_autograd", "klongpy.sys_fn_timer",
    "klongpy.sys_fn_ipc", "klongpy.sys_fn_static", "klongpy.sys_fn_hw", "klongpy.db.sys_fn_db", "klongpy.db.sys_fn_kvs",
    "klongpy.web.sys_fn_web", "klongpy.ws.sys_fn_ws",
]
HEADER = re.compile(r"^(?P<head>.*?)\s{2,}\[(?P<title>[^\]]+)\]\s*$")

GROUPS = [
    ("Evaluation and output", "", ".E .d .p .pf .fmt .w .bkd .bkp .x .l .module .pc .rn .sys"),
    ("Reading and files", "", ".r .rl .rls .rs .mi .ic .oc .ac .cc .fc .tc .fl .df"),
    ("Python and backend", "", ".py .pyf .pya .pyc .bkf"),
    ("Strict mode", "", ".strict .unsafe"),
    ("System variables", "", ".a .cin .cout .cerr .cols .e .edit .fastpow .h"),
    ("Timers", "Defined in the base interpreter. See [Timers](timer.md).", ".timer .timerc"),
    ("IPC", "Defined in the base interpreter. See [IPC Capabilities](ipc_capabilities.md).",
     ".srv .cli .clid .clic .async"),
    ("Autograd", "Defined in the base interpreter; most need the PyTorch backend. See [PyTorch Backend](torch_backend.md).",
     ".jacobian .gradcheck .compile .compilex .cmodes .export"),
    ("Embedded programs", "Defined in the base interpreter. See [Compiling Klong to C](embedded.md).", ".extern .entry .param .fixed .float"),
    ("Hardware (experimental)", "Defined in the base interpreter. See [Klong functions as FPGA hardware](hardware.md).",
     ".hw .hw.lanes .hw.set .hw.put .hw.get .hw.free .hw.reset .hw.info"),
    ("Database", 'Load with `.py("klongpy.db")`. See [Fast Columnar Database](fast_columnar_database.md).',
     ".db .table .insert .index .rindex .schema"),
    ("Key-value and table storage", 'Load with `.py("klongpy.db")`. See [Table and Key-Value Stores](table_and_key_value_stores.md).',
     ".kvs .tables"),
    ("Web server", 'Load with `.py("klongpy.web")`. See [Web Server](web_server.md).', ".web .webc"),
    ("WebSockets", 'Load with `.py("klongpy.ws")`. See [WebSockets](websockets.md).', ".ws .wsc"),
]

DESC = {
    ".E": "Evaluate a string of Klong at global scope and return the result.",
    ".d": "Display a value without a newline (strings print without quotes).",
    ".p": "Display a value followed by a newline.",
    ".pf": "Fill a template like `.fmt`, print the result and a newline, and return the string.",
    ".fmt": "Fill a template string with `{}` placeholders and return the string.",
    ".w": "Write a value in readable form (strings keep their quotes), without a newline.",
    ".bkd": "Display a value in the backend's own format (a torch tensor stays a tensor).",
    ".bkp": "Print a value in the backend's own format, followed by a newline.",
    ".x": "Exit the interpreter with a success or failure status.",
    ".l": "Load and run a Klong source file, searching `KLONGPATH`.",
    ".module": "Mark the start or end of a module; `.module(:name)` opens one and `.module(0)` closes it.",
    ".pc": "Return the process time used so far, in seconds.",
    ".rn": "Return a random real number in 0 <= x < 1.",
    ".sys": "Run a shell command and return its exit code.",
    ".r": "Read one data object from the current input channel.",
    ".rl": "Read one line from the current input channel.",
    ".rls": "Read all remaining lines from the current input channel as a list of strings.",
    ".rs": "Read a data object from a string (the inverse of `.w`).",
    ".mi": "Return 1 if the input channel has more input, otherwise 0.",
    ".ic": "Open a file for reading and return an input channel.",
    ".oc": "Open a file for writing (truncating it) and return an output channel.",
    ".ac": "Open a file for appending and return an output channel.",
    ".cc": "Close a channel.",
    ".fc": "Select the input channel, or restore standard input with a false value.",
    ".tc": "Select the output channel, or restore standard output with a false value.",
    ".fl": "Flush the current output channel.",
    ".df": "Delete a file; an error if it cannot be deleted.",
    ".py": "Import a Python module and make its functions callable from Klong.",
    ".pyf": "Import named functions from a Python module.",
    ".pya": "Return an attribute of a Python object, such as a function in an imported module.",
    ".pyc": "Call a Python function or method with positional and keyword arguments.",
    ".bkf": "Import functions from the active backend's array module (NumPy or torch).",
    ".strict": "Turn on strict mode (level 1): a function may not create a global by assignment.",
    ".unsafe": "Turn strict mode off (level 0).",
    ".a": "The command line arguments of a Klong program.",
    ".cin": "Standard input channel.",
    ".cout": "Standard output channel.",
    ".cerr": "Standard error channel.",
    ".cols": "Screen width in columns.",
    ".e": "Epsilon: `1+.e` is 1.0 and `.e` is 1e-18 in KlongPy.",
    ".edit": "Line editor switch.",
    ".fastpow": "Switch for compiling `x^y` to a faster power function.",
    ".h": "Symbol naming the host system (`:unix` or `:plan9`).",
    ".timer": "Call a function every `y` seconds (fractions allowed) until it returns 0 or the timer is cancelled.",
    ".timerc": "Cancel a timer.",
    ".srv": "Start the IPC server on a port or address; raises if the bind fails; `.srv(0)` stops it.",
    ".cli": "Connect to a KlongPy IPC server and return a function that evaluates code remotely.",
    ".clid": "Connect to a KlongPy IPC server and return a dictionary backed by its variables.",
    ".clic": "Close a connection opened by `.cli` or `.clid`.",
    ".async": "Wrap a function so that it runs asynchronously and calls a completion function.",
    ".jacobian": "Jacobian of a function at a point.",
    ".gradcheck": "Check autograd gradients against numeric ones (PyTorch backend).",
    ".compile": "Compile a function with `torch.compile` (PyTorch backend).",
    ".compilex": "Compile a function with a chosen mode and backend (PyTorch backend).",
    ".cmodes": "List the available `torch.compile` modes and backends (PyTorch backend).",
    ".export": "Export a function's computation graph to a file (PyTorch backend).",
    ".extern": "Declare a function the host provides, with a typed signature, for a program that is compiled.",
    ".entry": "Mark a function as an entry point of a compiled program and fix its signature.",
    ".param": "Declare a global of a compiled program as a parameter the host sets between runs (a register in hardware).",
    ".fixed": "Convert a number to a fixed-point type, `fixed<W,I>`, in a compiled program.",
    ".float": "Convert a fixed-point or integer value to a float in a compiled program.",
    ".hw": "Bind the FPGA kernels of a built program (or the simulator's) and return a dictionary of functions.",
    ".hw.lanes": "Bind the kernels of a program built with several lanes: copies of the program side by side.",
    ".hw.set": "Set a parameter of a bound kernel between runs.",
    ".hw.put": "Upload the arguments of calls of a kernel to the card once and return a buffer.",
    ".hw.get": "Download a buffer made by `.hw.put`.",
    ".hw.free": "Free a buffer made by `.hw.put`.",
    ".hw.reset": "Make the next call of a kernel start the program from its initial state.",
    ".hw.info": "A dictionary describing the hardware in use and the loaded image.",
    ".db": "Create a DuckDB database from a dictionary of tables.",
    ".table": "Create a table from column definitions.",
    ".insert": "Insert rows into a table.",
    ".index": "Create an index on columns of a table.",
    ".rindex": "Remove the indexes of a table.",
    ".schema": "Return the columns of a table, or the schemas of a database.",
    ".kvs": "Create a file-backed key-value store at a path.",
    ".tables": "Create file-backed table storage at a path.",
    ".web": "Start an HTTP server with GET and POST routes and return its handle.",
    ".webc": "Stop a web server.",
    ".ws": "Connect to a WebSocket server.",
    ".wsc": "Close a WebSocket connection.",
}

# Examples run at generation time: name -> list of expressions (one block per name,
# one interpreter per block, so later lines may use earlier ones).
EXAMPLES = {
    ".E": ['.E("a::123");a'],
    ".p": ['.p([1 2 3])'],
    ".fmt": ['.fmt("Spot: {}";3.5)', '.fmt("{1} {0}";[;1;2])', '.fmt("{:8}";"ab")',
             '.fmt("{:-8}";"ab")', '.fmt("{:5.2}";3.0)', '.fmt("{{}}";1)', '.fmt("{}{}";1)'],
    ".pf": ['.pf("a={}";1)'],
    ".sys": ['.sys("exit 3")', '.sys("true")', '.sys("false")'],
    ".rn": ['.rn()<1'],
    ".rs": ['.rs("[1 2 3]")'],
    ".pyf": ['.pyf("numpy";"sqrt");sqrt(4)'],
    ".bkf": ['.bkf("exp");exp(0.0)'],
    ".e": ['1+.e', '.e'],
    ".strict": ['.strict()', 'f::{newvar::42}', 'f()', 'g::{.E("newvar::42")}', 'g()', 'newvar'],
    ".unsafe": ['.unsafe()', 'f::{newvar::42}', 'f()', 'newvar'],
    ".timer": ['cb::{0}', '.timer("t";-1;cb)', '.timer("t";"x";cb)', '.timer("t";0.5;3)'],
}

# Variables documented in the Klong manual that KlongPy does not define.
UNDEFINED_VARS = {".a", ".cols", ".edit", ".fastpow", ".h"}

NOTES = {
    ".e": (
        "KlongPy numbers are IEEE doubles, so the statement above about the smallest "
        "difference does not hold. `.e` is 1e-18, `1+.e` is 1.0, and the spacing "
        "between neighbouring doubles is about 1.4e-17 at 0.1 and 1.1e-16 just below 1."
    ),
    ".sys": (
        "The exit code is returned as an integer. Output of the command goes to the "
        "terminal of the interpreter process, not to the Klong output channel, so "
        "`.tc` does not capture it."
    ),
    ".strict": (
        "KlongPy has no `global:` prefix. A function that needs to create a global "
        "uses `.E`, which evaluates its string at global scope."
    ),
    ".timer": (
        "The interval `y` may be fractional. A negative, `NaN` or non-numeric `y`, "
        "or a `z` that is not a function, raises an error instead of returning a "
        "message string. If the callback raises, the error is logged and the timer "
        "is cancelled."
    ),
    ".srv": (
        "When the port cannot be bound (for example it is in use), `.srv` raises an "
        "error such as `cannot listen on 0.0.0.0:8888: ... address already in use`. "
        "A call over `.cli` or `.clid` that gets no reply within 30 seconds raises an "
        "error containing `timed out`, closes the connection, and fails any other "
        "call waiting on it."
    ),
}

# Docstring text that is wrong today. The generator shows the corrected text; each
# (old, new) is applied when `old` is found, so the page stays right once the
# docstring itself is fixed.
FIXES = {
    ".fmt": [
        ('.fmt("{:5.2}";3)           -->  " 3.00"', '.fmt("{:5.2}";3)           -->  "    3.00"'),
    ],
    ".strict": [
        ("In strict mode, functions cannot create new global variables\nunless they are explicitly declared with the global: prefix.",
         "In strict mode, a function cannot create a global variable by assignment;\nassigning to a name that is not a parameter, a declared local or an existing\nglobal is an error. There is no global: prefix. To create a global from a\nfunction, use .E."),
        ('    counter::0\n    increment::{counter::counter+1}  :" ERROR"\n    increment::{[global:counter];counter::counter+1}  :" OK"',
         '    counter::0\n    increment::{counter::counter+1}  :" OK - counter already exists"\n    f::{newvar::42}\n    f()  :" ERROR - newvar is not defined"'),
    ],
    ".unsafe": [
        ('    f::{newvar::42}  :" OK - creates global newvar"\n    f()',
         '    f::{newvar::42}\n    f()  :" newvar is local to the call; no global is created"'),
        ("In unsafe mode, functions can create new global variables\nwithout declaration (legacy behavior).",
         "Strict mode is off. Assigning to a new name inside a function still makes\nit local to the call; it does not create a global. Use .E for that."),
    ],
    ".E": [("persist as globals \u2014 even", "persist as globals, even")],
    ".e": [
        ("On a 9-digit implementation, it would be 0.000000001 (1e-9).",
         "In KlongPy it is 1e-18."),
    ],
}

_KLONG_BACKSLASH = {"\x07": "\\a", "\x08": "\\b"}
_KLONG_BACKSLASH.update({chr(i): "\\%d" % i for i in range(1, 7)})


def unescape(doc):
    return "".join(_KLONG_BACKSLASH.get(c, c) for c in doc)


def collect():
    items = {}
    for m in MODULES:
        mod = importlib.import_module(m)
        for n, f in inspect.getmembers(mod, inspect.isfunction):
            if not (n.startswith("eval_sys_") and f.__module__ == m):
                continue
            doc = textwrap.dedent(unescape(f.__doc__ or "")).strip("\n")
            lines = doc.splitlines()
            first = next((l for l in lines if l.strip()), "")
            mm = HEADER.match(first.strip())
            if not mm:
                raise SystemExit(f"{n}: docstring has no header line")
            head = mm.group("head").strip()
            name = re.match(r"\.\w+(?:\.\w+)*", head).group(0)
            body = "\n".join(lines[lines.index(first) + 1:]).strip("\n")
            items[name] = dict(name=name, head=head, title=mm.group("title"), body=body, module=m)
    return items


def first_sentence(body):
    para = body.strip().split("\n\n")[0] if body.strip() else ""
    para = " ".join(para.split())
    m = re.match(r"(.+?\.)(\s|$)", para)
    return (m.group(1) if m else para) or ""


def run_examples(exprs):
    """Run expressions in one fresh REPL interpreter; return [(expr, shown output)]."""
    buf = io.StringIO()
    out = []
    with contextlib.redirect_stdout(buf):
        klong, loops = create_repl()
        try:
            for e in exprs:
                buf.seek(0)
                buf.truncate()
                try:
                    r = klong(e)
                    shown = buf.getvalue() + kg_write(r, klong._backend, display=False)
                except Exception as ex:
                    shown = buf.getvalue() + "error: " + str(ex).split("\n")[0]
                out.append((e, shown))
        finally:
            cleanup_repl(loops)
    return out


def fix_body(name, body):
    for old, new in FIXES.get(name, []):
        if old in body:
            body = body.replace(old, new, 1)
    return body


def main():
    items = collect()
    grouped = set()
    out = []
    w = out.append
    w("# System Functions and Variables\n")
    w("This page lists every system function and variable that KlongPy defines, with a one-line "
      "description and the full text of its docstring. It is generated from the "
      "`eval_sys_*` docstrings, and the examples were run when the page was generated; "
      "the output shown is what the interpreter printed. The REPL command `]h .name` shows the docstring.\n")
    w("An error is shown as `error: message`. The REPL adds its own prefix to errors.\n")
    groups = list(GROUPS)
    listed = {n for _, _, names in groups for n in names.split()}
    extra = sorted(set(items) - listed)
    if extra:
        groups.append(("Other", "", " ".join(extra)))
    for gname, gnote, names in groups:
        names = [n for n in names.split() if n in items]
        if not names:
            continue
        w(f"## {gname}\n")
        if gnote:
            w(gnote + "\n")
        w("| Name | Title | Description |")
        w("|------|-------|-------------|")
        for n in names:
            it = items[n]
            d = DESC.get(n) or first_sentence(it["body"])
            w(f"| `{it['head']}` | {it['title']} | {d} |")
        w("")
        for n in names:
            it = items[n]
            grouped.add(n)
            w(f"### `{n}`\n")
            w(f"`{it['head']}` [{it['title']}]\n")
            if n in UNDEFINED_VARS:
                w("**Not defined in KlongPy.** The name is documented in the Klong manual; in KlongPy it "
                  f"evaluates to the symbol `{n}` and has no effect.\n")
            body = fix_body(n, it["body"])
            if body.strip():
                w("```text")
                w(body)
                w("```\n")
            if n in NOTES:
                w(NOTES[n] + "\n")
            if n in EXAMPLES:
                w("```kgpy")
                for e, shown in run_examples(EXAMPLES[n]):
                    w(f"?> {e}")
                    w(shown)
                w("```\n")
    sys.stdout.write("\n".join(out).rstrip() + "\n")


if __name__ == "__main__":
    main()
