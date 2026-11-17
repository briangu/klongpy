"""
The rules of the static profile, as data: one stable id per rejection.

    RULES   id -> Rule(id, group, title, section, hint)

A profile violation is reported as a KlongStaticError whose ``rule`` attribute
is one of these ids (docs/static-profile.md; ``section`` is the
section of that document the rule comes from, "+" marks a rule this checker
adds because the design implies it without naming it). The ids are part of
the contract with the reject fixtures (tests/static/reject/<id>.kg) and with
the profile document; do not rename one without renaming both.
"""
from collections import namedtuple

Rule = namedtuple("Rule", "id group title section hint")

GROUPS = (
    ("program", "Program structure and declarations"),
    ("global", "Compile-time values"),
    ("value", "Values a run-time function may not hold"),
    ("length", "Results whose length depends on run-time data"),
    ("form", "Forms outside the profile"),
    ("type", "Types and shapes"),
    ("stream", "The streamable tier (--streamable)"),
)

_R = (
    # -- program structure -------------------------------------------------
    ("strict-syntax", "program", "the program does not parse at strict level 2", "1",
     None),
    ("toplevel-effect", "program", "a top-level expression that is not a definition or a declaration", "1",
     "a compiled program has no top-level effects: define a function and mark it with .entry"),
    ("decl-literal", "program", "an argument of .extern, .entry or .l that is not a string literal", "1, 9",
     "declarations are read at compile time: write the name and the signature as string literals"),
    ("signature", "program", "a malformed signature", "9",
     "a signature is \"params -> result\", for example \"int float[64] -> float\" or \"-> void\""),
    ("extern", "program", "an extern that cannot be declared (bad or reserved name, declared twice with different signatures)", "9",
     None),
    ("entry", "program", "an entry that cannot be declared (unknown function, wrong number of parameters, reserved name, "
     "result that does not match the function)", "9",
     None),
    ("load", "program", "a file named by .l that cannot be found or read", "1",
     None),
    ("redefinition", "program", "a function or extern name defined more than once, or assigned", "1+",
     "a compiled function is one definition: give the second one another name"),
    # -- compile-time values -----------------------------------------------
    ("global-value", "global", "a global whose compile-time value is a string, char, symbol, dictionary, function or :undefined", "2",
     "a global is an int, a float or a rectangular array of them"),
    ("global-array", "global", "a global whose compile-time value is a ragged, mixed or empty array, or has more than 4 dimensions", "2",
     "every row must have the same length and hold only numbers"),
    ("int-range", "global", "an integer constant outside the range of kg_int", "2",
     "the default kg_int is int32_t; --int64 widens it"),
    # -- run-time values -----------------------------------------------------
    ("string", "value", "a string or a character as a run-time value", "3",
     "a string literal may only be passed directly to an extern parameter of type str"),
    ("symbol", "value", "a symbol as a run-time value", "3",
     "a name inside [...] is a symbol: write [;a;b] to build a list from variables"),
    ("dict", "value", "a dictionary as a run-time value", "3",
     None),
    ("list-literal", "value", "a list that is not a rectangular list of numbers", "3",
     "every element must be a number or a list of the same shape"),
    ("fn-value", "value", "a function used as a value", "3",
     "a function may only be called by name with all its arguments, or stand directly as the verb of an adverb"),
    ("projection", "value", "a projection: a call with fewer arguments than the function takes", "3",
     "supply every argument, or write a function that does: {f(1;x)}"),
    ("void-value", "value", "the result of a void extern used as a value", "9",
     "an extern declared \"-> void\" has no result"),
    # -- result length ---------------------------------------------------------
    ("where", "length", "&a (Expand/Where): the result length depends on the data", "3", None),
    ("find", "length", "a?b (Find): the result length depends on the data", "3", None),
    ("range", "length", "?a (Range): the result length depends on the data", "3", None),
    ("group", "length", "=a (Group): the result length depends on the data", "3", None),
    ("grade", "length", "<a and >a (Grade): left out of version 1 (needs a sort routine in the runtime)", "3, 13", None),
    ("split", "length", "a:#b (Split): the result is ragged", "3", None),
    ("cut", "length", "a:_b (Cut): the result is ragged", "3", None),
    ("const-count", "length", "take, drop, reshape or enumerate with a count that is not a compile-time constant", "3",
     "the count must be a literal, #a or ^a, or a global that no function assigns"),
    ("converge", "length", "f:~a (Converge): the number of steps is not known", "3",
     "use n f:*a (Iterate) or p f:~a (While)"),
    ("scan-dynamic", "length", "the \\~ and \\* scan forms: the result length is not known", "3", None),
    # -- forms ---------------------------------------------------------------
    ("take-rank", "form", "take from anything but a rank-1 array", "3",
     "version 1 takes from rank-1 arrays only; index the rows instead: a@!n"),
    ("amend-form", "form", "an amend other than a:=v,i on a rank-1 array with one scalar value", "3",
     "write a:=v,i with a scalar v and either a scalar index i or an index list of known length"),
    ("size-of-atom", "form", "# or ^ of an atom", "3",
     "Size and Shape take arrays"),
    ("unsupported-op", "form", "an operator, adverb or adverb chain that is not in the operator table", "3+",
     None),
    ("recursion", "form", "recursion, direct or mutual, including .f", "3",
     "storage is planned ahead of time, so the call graph must have no cycle: use n f:*a or p f:~a"),
    ("sys-fn", "form", "a system function or variable in a run-time function", "3",
     "only names declared with .extern exist on the target; system functions run at compile time, in a top-level definition"),
    ("undefined-name", "form", "a name that is not a parameter, a local, a global, a function or an extern", "3+",
     None),
    ("unset-local", "form", "a local read before it is assigned", "3+",
     None),
    ("closure-assign", "form", "a nested function literal that assigns a local of the enclosing function", "3+",
     "return the value from the literal and assign it outside"),
    ("arity", "form", "a call or an adverb with the wrong number of arguments for the function", "3+",
     None),
    # -- types and shapes --------------------------------------------------------
    ("operand", "type", "an operand of the wrong type or rank for the operator", "4",
     None),
    ("shape-mismatch", "type", "operands whose shapes do not agree", "4",
     "the shorter shape must be a prefix of the longer"),
    ("shape-change", "type", "a variable whose type or shape differs between two assignments", "3, 4",
     None),
    ("cond-shape", "type", "a conditional whose branches have different shapes", "3, 4",
     None),
    ("cond-scalar", "type", "a condition that is not a scalar", "4",
     "reduce it first, for example with &/ or |/"),
    ("div-zero", "type", "division of an integer constant by the constant zero", "3",
     None),
    ("index-range", "type", "a constant index out of range", "4",
     None),
    ("array-shape", "type", "an array with more than 4 dimensions or with an empty dimension", "2",
     "every dimension is a positive integer and the rank is 1 to 4"),
    ("loop-state", "type", "an accumulator or loop state whose type or shape changes", "4",
     "start from a value of the type the function returns (0.0 instead of 0)"),
    ("extern-call", "type", "a call of an extern that does not match its signature", "9",
     None),
    ("fixed", "type", "a fixed-point number used in a way that is not defined: mixed with a float or another fixed type, "
     "an operator it does not have, a conversion that is not exact", "6",
     "convert explicitly with .fixed(\"fixed<W,I>\";x) or .float(x); an int converts implicitly only as a constant that fits"),
    ("param", "program", "a .param that cannot be declared, a parameter that a compiled function assigns, or a parameter used as a shape", "6",
     "a parameter is a scalar register the host writes between runs; a function only reads it"),
)

RULES = {r[0]: Rule(*r) for r in _R}

assert len(RULES) == len(_R), "duplicate rule id"


# ---------------------------------------------------------------------------
# The streamable tier (kgpy --check-static --streamable): not violations of the
# profile, but constructs that keep a hardware kernel from taking one record per clock cycle.
# ---------------------------------------------------------------------------

STREAM_RULES = {r[0]: Rule(*r) for r in (
    ("stream-loop", "stream", "a loop whose number of iterations depends on the data (While, or Iterate with a count that is "
     "not a constant)", "5",
     "a pipelined kernel needs a bounded loop: use n f:*a with a constant n, and stop early by holding the value"),
    ("stream-state", "stream", "state that is read and written back through a multi-cycle operation (a float operation, or a "
     "division or remainder by a value that is not a constant)", "5",
     "a float add, multiply or divide, or a divider, takes several cycles, so the next record has to wait for it: keep that state "
     "as an int or a fixed<W,I> updated by adds, multiplies and constant moduli, or let the host do the accumulation"),
    ("stream-table", "stream", "a data-dependent index into an array of more than 1024 items", "5",
     "a random access into a big table needs a memory with one port per access; make the table smaller, or index it "
     "with a constant window (x@idx with idx[a][b] = c + a + b is a shift register)"),
    ("stream-block", "stream", "an entry that does more than 4096 element operations per call and is not a sliding window", "5",
     "such a call is a block program: the kernel can start a new record only when the previous one has gone through its "
     "loops. Write the filter as a window, +/taps*x@idx with the index table k+!w, to stream it"),
)}
