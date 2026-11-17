"""
The operator table of the static profile, as plain data
(docs/static-profile.md).

    OPTABLE              every operator and adverb of Klong, one Op per glyph and valence
    FORMS                the allowed forms that are not operators (literals, control, calls)
    lookup(glyph, valence) -> Op or None
    OPERAND_KINDS, TYPE_RULES, SHAPE_RULES     the closed vocabularies used by the columns

This is the single source for the checker (klongpy/static/checker.py applies
the rule named in each row), for the generated section of the profile
document (klongpy/static/profile_doc.py) and for the per-row tests. Lowering
and the C expansion are attached to the rows elsewhere.

Columns of an Op

    id          stable name of the row ("add", "take", "each-pair")
    glyph       the operator or adverb as written
    valence     "monad", "dyad", "adverb1" (f'a: one operand) or "adverb2" (a f'b)
    name        Klong's name for it
    status      "allowed" or "rejected"
    operands    what the operands must be: a tuple of OPERAND_KINDS names, in the
                order written (left, right); for adverbs the verb is not listed
    type_rule   element type of the result: a TYPE_RULES name
    shape_rule  shape of the result: a SHAPE_RULES name
    rule        rejected rows: the id in klongpy.static.rules of the error
    note        one line for the document
    examples    allowed rows: tuples (body, signature[, inputs]); ``f::{body}``
                entered with that signature is inside the profile and has the
                result type the signature names. inputs, when given, are the
                Python arguments the tests call it with (indices must be in
                range); otherwise the tests make some up.

Every operator the interpreter knows has a row (a test compares the table
with the interpreter's own tables), so a new operator cannot slip into
compiled programs unnoticed: a glyph without a row is rejected as
unsupported-op.
"""
from collections import namedtuple

Op = namedtuple("Op", "id glyph valence name status operands type_rule shape_rule rule note examples")
Form = namedtuple("Form", "id form note examples")

OPERAND_KINDS = {
    "num": "an int or a float, scalar or array",
    "array": "an array (rank 1 to 4)",
    "rank1": "a rank-1 array",
    "rank2": "a rank-2 array",
    "int": "an int, scalar or array",
    "int-scalar": "an int scalar",
    "const-int": "an int scalar that is a compile-time constant",
    "const-shape": "a compile-time constant: a positive int, or a list literal of 1 to 4 positive ints",
    "index": "an int scalar or an int array; constants are range-checked",
    "same-shape": "a value of the same shape as the other operand",
    "amend-pair": "written v,i: a scalar value v and a scalar index or a rank-1 index list of known length",
    "predicate": "a monadic function (a name or a literal) that returns a scalar",
}

TYPE_RULES = {
    "promote": "float if either operand is float, else int",
    "float": "always float",
    "int": "always int",
    "int-only": "both operands must be int; the result is int",
    "power": "int when both operands are int and the exponent is a non-negative constant, else float",
    "same": "the element type of the operand",
    "right": "the element type of the right operand",
    "left": "the element type of the left operand",
    "verb": "the element type the verb returns",
    "acc": "the accumulator: the fixed point of acc = f(acc, item), with at most one int to float promotion",
    "state": "the type of the starting value, which the verb must return unchanged",
}

SHAPE_RULES = {
    "prefix": "scalar with scalar is a scalar; scalar with array is the array's shape; "
              "of two arrays the shorter shape must be a prefix of the longer, which is the result",
    "same": "the shape of the operand",
    "scalar": "a scalar",
    "item": "the operand's shape without its first dimension (a scalar stays a scalar)",
    "index": "a scalar index drops the first dimension; an index array of shape S gives S followed by the rest",
    "take": "[|n|] (the items cycle when |n| is larger than the length; negative n takes from the end)",
    "drop": "the first dimension shrinks by |n|, which must leave at least one item",
    "join": "scalar,scalar is [2]; rank 1 with a scalar is [n+1]; "
            "two arrays of the same rank whose items have the same shape are [n+m, ...]",
    "reshape": "the constant shape; the items cycle",
    "transpose": "[m,n] for an operand of shape [n,m]",
    "enumerate": "[n]",
    "wrap": "[1] followed by the operand's shape",
    "rank-vector": "[rank of the operand]",
    "amend": "the shape of the array",
    "each": "[n] followed by the shape the verb returns",
    "each-pair": "[n-1] followed by the shape the verb returns",
    "acc": "the shape of the accumulator",
    "scan": "[n] followed by the shape of the accumulator",
    "scan-seed": "[n+1] followed by the shape of the accumulator (the seed is the first item)",
    "state": "the shape of the starting value",
}


def _ok(id, glyph, valence, name, operands, type_rule, shape_rule, note, *examples):
    return Op(id, glyph, valence, name, "allowed", tuple(operands.split()), type_rule, shape_rule, None, note, examples)


def _no(id, glyph, valence, name, rule, note):
    return Op(id, glyph, valence, name, "rejected", (), None, None, rule, note, ())


OPTABLE = (
    # -- atomic dyads ---------------------------------------------------------
    _ok("add", "+", "dyad", "Plus", "num num", "promote", "prefix", "",
        ("x+y", "int int -> int"), ("x+y", "int[3] float -> float[3]"), ("x+y", "int[2,3] int[2] -> int[2,3]")),
    _ok("sub", "-", "dyad", "Minus", "num num", "promote", "prefix", "",
        ("x-y", "float int[3] -> float[3]")),
    _ok("mul", "*", "dyad", "Times", "num num", "promote", "prefix", "",
        ("x*y", "int[3] int[3] -> int[3]")),
    _ok("div", "%", "dyad", "Divide", "num num", "float", "prefix",
        "always float; a constant int divided by the constant 0 is an error",
        ("x%y", "int int -> float"), ("x%y", "int[3] int -> float[3]")),
    _ok("idiv", ":%", "dyad", "Integer-Divide", "int int", "int-only", "prefix",
        "truncates toward zero; KlongPy also accepts floats here, the profile does not",
        ("x:%y", "int int -> int"), ("x:%2", "int[3] -> int[3]")),
    _ok("rem", "!", "dyad", "Remainder", "num num", "promote", "prefix",
        "the result has the sign of the dividend (C's %, fmod for floats)",
        ("x!y", "int int -> int"), ("x!y", "float int -> float")),
    _ok("pow", "^", "dyad", "Power", "num num", "power", "prefix",
        "KlongPy returns an int whenever the result is whole; the profile cannot depend on values",
        ("x^2", "int -> int"), ("x^y", "int int -> float"), ("x^2", "float[3] -> float[3]")),
    _ok("max", "|", "dyad", "Max/Or", "num num", "promote", "prefix", "",
        ("x|y", "int float -> float")),
    _ok("min", "&", "dyad", "Min/And", "num num", "promote", "prefix", "",
        ("x&y", "int[3] int[3] -> int[3]")),
    _ok("lt", "<", "dyad", "Less", "num num", "int", "prefix", "",
        ("x<y", "float[3] float -> int[3]")),
    _ok("gt", ">", "dyad", "More", "num num", "int", "prefix", "",
        ("x>y", "int int -> int")),
    _ok("eq", "=", "dyad", "Equal", "num num", "int", "prefix", "",
        ("x=y", "int[3] int[3] -> int[3]")),
    _ok("match", "~", "dyad", "Match", "num same-shape", "int", "scalar",
        "operands of different shapes are an error (KlongPy answers 0)",
        ("x~y", "int[3] int[3] -> int"), ("x~y", "float float -> int")),
    # -- structural dyads -----------------------------------------------------
    _ok("index", "@", "dyad", "At/Index", "array index", "left", "index",
        "a negative index counts from the end; f@x (apply) is not in the profile",
        ("x@y", "float[4] int -> float", ([1.5, 2.5, 3.5, 4.5], 2)),
        ("x@y", "float[4] int[2,3] -> float[2,3]", ([1.5, 2.5, 3.5, 4.5], [[0, 1, 2], [3, 0, 1]])),
        ("x@y", "int[3,2] int -> int[2]", ([[1, 2], [3, 4], [5, 6]], 1)),
        ("x@[0 2]", "int[3] -> int[2]")),
    _ok("take", "#", "dyad", "Take", "const-int rank1", "right", "take",
        "rank 1 only in version 1; -2#a is -(2#a): write (-2)#a",
        ("2#x", "int[4] -> int[2]"), ("(-2)#x", "float[4] -> float[2]"), ("6#x", "int[4] -> int[6]")),
    _ok("drop", "_", "dyad", "Drop", "const-int array", "right", "drop",
        "negative n drops from the end",
        ("1_x", "int[4] -> int[3]"), ("(-1)_x", "int[3,2] -> int[2,2]")),
    _ok("join", ",", "dyad", "Join", "num num", "promote", "join",
        "joining ints with floats gives floats",
        ("x,y", "int[2] float[3] -> float[5]"), ("x,y", "int int -> int[2]"),
        ("x,y", "int[2,3] int[1,3] -> int[3,3]"), ("x,y", "float int[2] -> float[3]")),
    _ok("rotate", ":+", "dyad", "Rotate", "int-scalar rank1", "right", "same",
        "rank 1 only (KlongPy rotates the flattened data of a matrix); negative n rotates left",
        ("y:+x", "int[4] int -> int[4]"), ("(-1):+x", "float[3] -> float[3]")),
    _ok("reshape", ":^", "dyad", "Reshape", "const-shape num", "right", "reshape",
        "the row-major data cycles; a scalar n takes a scalar or a rank-1 array only (write [n]:^a for a matrix)",
        ("[2 3]:^x", "int[6] -> int[2,3]"), ("4:^x", "int[2] -> int[4]"), ("[2 2]:^x", "float -> float[2,2]")),
    _ok("amend", ":=", "dyad", "Amend", "rank1 amend-pair", "promote", "amend",
        "g::g:=v,i on a global is an in-place store",
        ("x:=y,0", "float[4] float -> float[4]"), ("x:=y,[0 2]", "int[4] int -> int[4]"),
        ("x:=y,z", "int[4] int int -> int[4]", ([1, 2, 3, 4], 9, 1))),
    _ok("define", "::", "dyad", "Define", "", "right", "same",
        "name::expr assigns a local or a global and is the value assigned; see the forms table", ),
    # -- monads -----------------------------------------------------------------
    _ok("neg", "-", "monad", "Negate", "num", "same", "same",
        "a leading - negates the whole expression to its right",
        ("-x", "int[3] -> int[3]")),
    _ok("floor", "_", "monad", "Floor", "num", "int", "same", "rounds toward minus infinity",
        ("_x", "float[3] -> int[3]")),
    _ok("recip", "%", "monad", "Reciprocal", "num", "float", "same", "",
        ("%x", "int -> float")),
    _ok("not", "~", "monad", "Not", "num", "int", "same", "",
        ("~x", "int[3] -> int[3]")),
    _ok("size", "#", "monad", "Size", "array", "int", "scalar",
        "the first dimension, a compile-time constant",
        ("#x", "int[3,2] -> int")),
    _ok("shape", "^", "monad", "Shape", "array", "int", "rank-vector", "a compile-time constant",
        ("^x", "int[3,2] -> int[2]")),
    _ok("first", "*", "monad", "First", "num", "same", "item", "",
        ("*x", "int[3,2] -> int[2]"), ("*x", "float[3] -> float")),
    _ok("reverse", "|", "monad", "Reverse", "num", "same", "same", "reverses along the first dimension",
        ("|x", "int[3] -> int[3]"), ("|x", "int[3,2] -> int[3,2]")),
    _ok("transpose", "+", "monad", "Transpose", "rank2", "same", "transpose", "",
        ("+x", "int[2,3] -> int[3,2]")),
    _ok("enumerate", "!", "monad", "Enumerate", "const-int", "int", "enumerate", "n must be a positive constant",
        ("x+!3", "int -> int[3]")),
    _ok("list", ",", "monad", "List", "num", "same", "wrap", "",
        (",x", "int -> int[1]"), (",x", "int[3] -> int[1,3]")),
    # -- adverbs ------------------------------------------------------------------
    _ok("each", "'", "adverb1", "Each", "array", "verb", "each", "f is called on each item along the first dimension",
        ("{x*2}'x", "int[3] -> int[3]"), ("-'x", "float[3] -> float[3]"), ("{x,x}'x", "int[3] -> int[3,2]")),
    _ok("each2", "'", "adverb2", "Each-2", "array array", "verb", "each",
        "both operands must have the same length (KlongPy stops at the shorter)",
        ("x{x+y}'y", "int[3] float[3] -> float[3]")),
    _ok("over", "/", "adverb1", "Over", "array", "acc", "acc", "the accumulator starts from the first item",
        ("+/x", "int[4] -> int"), ("+/x", "float[3,2] -> float[2]"), ("{x+y%2}/x", "int[4] -> float")),
    _ok("over-seed", "/", "adverb2", "Over-Neutral", "num array", "acc", "acc", "the accumulator starts from the seed",
        ("0.0+/x", "int[4] -> float"), ("y+/x", "int[3] int -> int")),
    _ok("scan", "\\", "adverb1", "Scan-Over", "array", "acc", "scan", "",
        ("+\\x", "int[4] -> int[4]")),
    _ok("scan-seed", "\\", "adverb2", "Scan-Over-Neutral", "num array", "acc", "scan-seed", "returns n+1 items",
        ("0+\\x", "int[4] -> int[5]")),
    _ok("each-pair", ":'", "adverb1", "Each-Pair", "array", "verb", "each-pair", "needs at least two items",
        ("-:'x", "int[4] -> int[3]")),
    _ok("iterate", ":*", "adverb2", "Iterate", "int-scalar num", "state", "state", "n f:*a applies f n times",
        ("3{x*2}:*x", "int -> int"), ("y{x*0.5}:*x", "float int -> float", (8.0, 2))),
    _ok("while", ":~", "adverb2", "While", "predicate num", "state", "state", "p f:~a applies f while p holds",
        ("{x<100}{x*2}:~x", "int -> int", (3,))),
    # -- rejected ---------------------------------------------------------------------
    _no("where", "&", "monad", "Expand/Where", "where", "result length depends on the data"),
    _no("find", "?", "dyad", "Find", "find", "result length depends on the data"),
    _no("range", "?", "monad", "Range", "range", "result length depends on the data"),
    _no("group", "=", "monad", "Group", "group", "result is ragged"),
    _no("grade-up", "<", "monad", "Grade-Up", "grade", "fixed length; left out of version 1 (needs a sort routine)"),
    _no("grade-down", ">", "monad", "Grade-Down", "grade", "fixed length; left out of version 1 (needs a sort routine)"),
    _no("split", ":#", "dyad", "Split", "split", "result is ragged"),
    _no("cut", ":_", "dyad", "Cut", "cut", "result is ragged"),
    _no("converge", ":~", "adverb1", "Converge", "converge", "number of steps unknown"),
    _no("scan-converge", "\\~", "adverb1", "Scan-Converging", "scan-dynamic", "result length unknown"),
    _no("scan-while", "\\~", "adverb2", "Scan-While", "scan-dynamic", "result length unknown"),
    _no("scan-iterate", "\\*", "adverb2", "Scan-Iterating", "scan-dynamic",
        "fixed length for a constant count, but not in the design's list; left out of version 1"),
    _no("format", "$", "monad", "Format", "string", "makes a string"),
    _no("format2", "$", "dyad", "Format2", "string", "makes a string"),
    _no("form", ":$", "dyad", "Form", "string", "reads a string"),
    _no("char", ":#", "monad", "Char", "string", "makes a character"),
    _no("atom", "@", "monad", "Atom", "unsupported-op", "always a constant in a typed program"),
    _no("undefined", ":_", "monad", "Undefined", "unsupported-op", ":undefined is not a run-time value"),
    _no("amend-in-depth", ":-", "dyad", "Amend-in-Depth", "unsupported-op", "not in the design's list"),
    _no("index-in-depth", ":@", "dyad", "Index-in-Depth", "unsupported-op", "not in the design's list; write (a@i)@j"),
    _no("each-left", ":\\", "adverb2", "Each-Left", "unsupported-op", "not in the design's list; write {a f x}'b"),
    _no("each-right", ":/", "adverb2", "Each-Right", "unsupported-op", "not in the design's list; write {x f b}'a"),
    _no("each-index", "@'", "adverb1", "Each-Index", "unsupported-op", "not in the design's list"),
    _no("track", "˙", "monad", "Track", "unsupported-op", "autograd"),
    _no("grad", "∇", "monad", "Grad", "unsupported-op", "autograd"),
    _no("grad2", "∇", "dyad", "Grad", "unsupported-op", "autograd"),
    _no("jacobian", "∂", "dyad", "Jacobian", "unsupported-op", "autograd"),
    _no("autograd", ":>", "dyad", "Autograd", "unsupported-op", "autograd"),
)

_BY_KEY = {(op.glyph, op.valence): op for op in OPTABLE}
BY_ID = {op.id: op for op in OPTABLE}

assert len(_BY_KEY) == len(OPTABLE), "two rows for one glyph and valence"
assert len(BY_ID) == len(OPTABLE), "duplicate row id"
for _op in OPTABLE:
    assert all(k in OPERAND_KINDS for k in _op.operands), _op.id
    assert _op.status == "rejected" or (_op.type_rule in TYPE_RULES and _op.shape_rule in SHAPE_RULES), _op.id


def lookup(glyph, valence):
    """The row for a glyph used as "monad", "dyad", "adverb1" or "adverb2"; None if there is none."""
    return _BY_KEY.get((glyph, valence))


# ---------------------------------------------------------------------------
# The C expansion column: row id -> (strategy, int form, float form).
#
#   "expr"   an elementwise C expression; {a} and {b} are the operands (already of the
#            element type the operation works in), {line} the Klong source line. A tree of
#            these over one shape is one loop nest with no intermediate array.
#   "view"   an index remapping: the result reads its operand through other indices and
#            has no storage of its own (a fusible leaf).
#   "const"  folded at compile time.
#   "loop"   a loop that fills a slot (or a scalar accumulator).
#   "stmt"   a statement.
#
# Every allowed row has an entry and no rejected row has one; the emitter
# (klongpy/static/emit_c.py) handles exactly these, and a test holds the three
# sets equal, so the checker cannot accept an operator the emitter lacks.
# ---------------------------------------------------------------------------

C_EXPANSION = {
    "add": ("expr", "kg_iadd({a}, {b})", "({a} + {b})"),
    "sub": ("expr", "kg_isub({a}, {b})", "({a} - {b})"),
    "mul": ("expr", "kg_imul({a}, {b})", "({a} * {b})"),
    "div": ("expr", None, "kg_fdiv({a}, {b}, {line})"),
    "idiv": ("expr", "kg_idiv({a}, {b}, {line})", None),
    "rem": ("expr", "kg_imod({a}, {b}, {line})", "kg_fmod({a}, {b}, {line})"),
    "pow": ("expr", "kg_ipow({a}, {b})", "kg_pow({a}, {b})"),
    "max": ("expr", "kg_imax({a}, {b})", "kg_fmax({a}, {b})"),
    "min": ("expr", "kg_imin({a}, {b})", "kg_fmin({a}, {b})"),
    "lt": ("expr", "kg_ilt({a}, {b})", "kg_flt({a}, {b})"),
    "gt": ("expr", "kg_igt({a}, {b})", "kg_fgt({a}, {b})"),
    "eq": ("expr", "kg_ieq({a}, {b})", "kg_feq({a}, {b})"),
    "match": ("loop", "kg_match((kg_float){a}, (kg_float){b})", "kg_match({a}, {b})"),
    "index": ("view", None, None),
    "take": ("view", None, None),
    "drop": ("view", None, None),
    "join": ("view", None, None),
    "rotate": ("view", None, None),
    "reshape": ("view", None, None),
    "amend": ("loop", None, None),
    "define": ("stmt", None, None),
    "neg": ("expr", "kg_ineg({a})", "(-{a})"),
    "floor": ("expr", None, "kg_floor({a})"),
    "recip": ("expr", None, "kg_fdiv((kg_float)1, {a}, {line})"),
    "not": ("expr", "(kg_int)({a} == 0)", "(kg_int)({a} == 0)"),
    "size": ("const", None, None),
    "shape": ("const", None, None),
    "first": ("view", None, None),
    "reverse": ("view", None, None),
    "transpose": ("view", None, None),
    "enumerate": ("view", None, None),
    "list": ("view", None, None),
    "each": ("loop", None, None),
    "each2": ("loop", None, None),
    "over": ("loop", None, None),
    "over-seed": ("loop", None, None),
    "scan": ("loop", None, None),
    "scan-seed": ("loop", None, None),
    "each-pair": ("loop", None, None),
    "iterate": ("loop", None, None),
    "while": ("loop", None, None),
}

assert set(C_EXPANSION) == {op.id for op in OPTABLE if op.status == "allowed"}, "the C column covers exactly the allowed rows"


# ---------------------------------------------------------------------------
# Allowed forms that are not operators. Each example is (program, signature):
# the program defines f, and f entered with that signature is inside the
# profile with the result type the signature names.
# ---------------------------------------------------------------------------

FORMS = (
    Form("number", "`1`, `2.5`, `1e3`", "an integer literal is int; a literal with `.` or `e` is float",
         (("f::{x+1.5}", "int -> float"),)),
    Form("list", "`[1 2 3]`, `[[1 2] [3 4]]`", "a rectangular list of numbers; any float makes it float",
         (("f::{x+[1 2 3]}", "int -> int[3]"), ("f::{x*[[1 2.5] [3 4]]}", "int -> float[2,2]"))),
    Form("evaluated-list", "`[;a;b;c]`", "elements of the same shape; ints are promoted when any element is float",
         (("f::{[;x;y;x+y]}", "int int -> int[3]"), ("f::{[;x;1.5]}", "int -> float[2]"))),
    Form("conditional", "`:[c;a;b]`, `:[c1;a:|c2;b;else]`",
         "the condition is a scalar; the branches have the same shape; int with float gives float",
         (("f::{:[x>0;x;0-x]}", "int -> int"), ("f::{:[x>0;1:|x<0;2.5;0]}", "int -> float"))),
    Form("sequence", "`{a;b;c}`", "statements run in order; the last is the result",
         (("f::{[t];t::x*2;t+1}", "int -> int"),)),
    Form("local", "`{[a b];a::...}`", "a local's type is the join of its assignments (int with float is float); its shape is fixed",
         (("f::{[t];t::0;t::x%2;t}", "int -> float"),)),
    Form("parameters", "`x y z`, or `f(a;b;c;d)::{...}`", "named parameters allow more than three",
         (("f(a;b;c;d)::{(a+b)*c-d}", "int int float int -> float"),)),
    Form("destructure", "`[a b]::v`", "v is an array with exactly one item per name; as the first statement the names are locals",
         (("f::{[a b]::x;a+b}", "int[2] -> int"),)),
    Form("global", "`name::expr` to a top-level name", "the type and shape of a global never change",
         (("acc::0\nf::{acc::acc+x;acc}", "int -> int"),)),
    Form("call", "`g(a;b)`, `a g b`", "a compiled function, with every argument supplied; specialised per argument types",
         (("sq::{x*x}\nf::{sq(x)+sq(1.5)}", "int -> float"),)),
    Form("extern-call", "`read.adc(0)`", "a name declared with .extern; a str parameter takes a string literal",
         (('.extern("read.adc";"int -> float")\nf::{read.adc(x)*2}', "int -> float"),
          ('.extern("log.msg";"str int -> void")\nf::{log.msg("tick";x);x}', "int -> int"))),
    Form("verb", "`f'a` with f an operator, a function name or `{...}`",
         "a function literal is allowed only as the verb of an adverb or the predicate of While; "
         "it may read, but not assign, the locals of the function it is written in",
         (("sq::{x*x}\nf::{[k];k::2;(sq'x)+{x*k}'x}", "int[3] -> int[3]"),)),
)
