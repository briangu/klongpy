//! The rules, the operator table and the forms of the static profile: the source of truth (the Python package reads them through the extension).

pub struct Rule {
    pub id: &'static str,
    pub group: &'static str,
    pub title: &'static str,
    pub section: &'static str,
    pub hint: Option<&'static str>,
}

/// Every rule: the profile's (the profile rules) and then the streamable tier's.
pub static RULES: &[Rule] = &[
    Rule { id: "strict-syntax", group: "program", title: "the program does not parse at strict level 2", section: "1", hint: None },
    Rule { id: "toplevel-effect", group: "program", title: "a top-level expression that is not a definition or a declaration", section: "1", hint: Some("a compiled program has no top-level effects: define a function and mark it with .entry") },
    Rule { id: "decl-literal", group: "program", title: "an argument of .extern, .entry or .l that is not a string literal", section: "1, 9", hint: Some("declarations are read at compile time: write the name and the signature as string literals") },
    Rule { id: "signature", group: "program", title: "a malformed signature", section: "9", hint: Some("a signature is \"params -> result\", for example \"int float[64] -> float\" or \"-> void\"") },
    Rule { id: "extern", group: "program", title: "an extern that cannot be declared (bad or reserved name, declared twice with different signatures)", section: "9", hint: None },
    Rule { id: "entry", group: "program", title: "an entry that cannot be declared (unknown function, wrong number of parameters, reserved name, result that does not match the function)", section: "9", hint: None },
    Rule { id: "load", group: "program", title: "a file named by .l that cannot be found or read", section: "1", hint: None },
    Rule { id: "redefinition", group: "program", title: "a function or extern name defined more than once, or assigned", section: "1+", hint: Some("a compiled function is one definition: give the second one another name") },
    Rule { id: "global-value", group: "global", title: "a global whose compile-time value is a string, char, symbol, dictionary, function or :undefined", section: "2", hint: Some("a global is an int, a float or a rectangular array of them") },
    Rule { id: "global-array", group: "global", title: "a global whose compile-time value is a ragged, mixed or empty array, or has more than 4 dimensions", section: "2", hint: Some("every row must have the same length and hold only numbers") },
    Rule { id: "int-range", group: "global", title: "an integer constant outside the range of kg_int", section: "2", hint: Some("the default kg_int is int32_t; --int64 widens it") },
    Rule { id: "string", group: "value", title: "a string or a character as a run-time value", section: "3", hint: Some("a string literal may only be passed directly to an extern parameter of type str") },
    Rule { id: "symbol", group: "value", title: "a symbol as a run-time value", section: "3", hint: Some("a name inside [...] is a symbol: write [;a;b] to build a list from variables") },
    Rule { id: "dict", group: "value", title: "a dictionary as a run-time value", section: "3", hint: None },
    Rule { id: "list-literal", group: "value", title: "a list that is not a rectangular list of numbers", section: "3", hint: Some("every element must be a number or a list of the same shape") },
    Rule { id: "fn-value", group: "value", title: "a function used as a value", section: "3", hint: Some("a function may only be called by name with all its arguments, or stand directly as the verb of an adverb") },
    Rule { id: "projection", group: "value", title: "a projection: a call with fewer arguments than the function takes", section: "3", hint: Some("supply every argument, or write a function that does: {f(1;x)}") },
    Rule { id: "void-value", group: "value", title: "the result of a void extern used as a value", section: "9", hint: Some("an extern declared \"-> void\" has no result") },
    Rule { id: "where", group: "length", title: "&a (Expand/Where): the result length depends on the data", section: "3", hint: None },
    Rule { id: "find", group: "length", title: "a?b (Find): the result length depends on the data", section: "3", hint: None },
    Rule { id: "range", group: "length", title: "?a (Range): the result length depends on the data", section: "3", hint: None },
    Rule { id: "group", group: "length", title: "=a (Group): the result length depends on the data", section: "3", hint: None },
    Rule { id: "grade", group: "length", title: "<a and >a (Grade): not in the profile (it needs a sort routine in the runtime)", section: "3, 13", hint: None },
    Rule { id: "split", group: "length", title: "a:#b (Split): the result is ragged", section: "3", hint: None },
    Rule { id: "cut", group: "length", title: "a:_b (Cut): the result is ragged", section: "3", hint: None },
    Rule { id: "const-count", group: "length", title: "take, drop, reshape or enumerate with a count that is not a compile-time constant", section: "3", hint: Some("the count must be a literal, #a or ^a, or a global that no function assigns") },
    Rule { id: "converge", group: "length", title: "f:~a (Converge): the number of steps is not known", section: "3", hint: Some("use n f:*a (Iterate) or p f:~a (While)") },
    Rule { id: "scan-dynamic", group: "length", title: "the \\~ and \\* scan forms: the result length is not known", section: "3", hint: None },
    Rule { id: "take-rank", group: "form", title: "take from anything but a rank-1 array", section: "3", hint: Some("take works on rank-1 arrays only; index the rows instead: a@!n") },
    Rule { id: "amend-form", group: "form", title: "an amend other than a:=v,i on a rank-1 array with one scalar value", section: "3", hint: Some("write a:=v,i with a scalar v and either a scalar index i or an index list of known length") },
    Rule { id: "size-of-atom", group: "form", title: "# or ^ of an atom", section: "3", hint: Some("Size and Shape take arrays") },
    Rule { id: "unsupported-op", group: "form", title: "an operator, adverb or adverb chain that is not in the operator table", section: "3+", hint: None },
    Rule { id: "recursion", group: "form", title: "recursion, direct or mutual, including .f", section: "3", hint: Some("storage is planned ahead of time, so the call graph must have no cycle: use n f:*a or p f:~a") },
    Rule { id: "sys-fn", group: "form", title: "a system function or variable in a run-time function", section: "3", hint: Some("only names declared with .extern exist on the target; system functions run at compile time, in a top-level definition") },
    Rule { id: "undefined-name", group: "form", title: "a name that is not a parameter, a local, a global, a function or an extern", section: "3+", hint: None },
    Rule { id: "unset-local", group: "form", title: "a local read before it is assigned", section: "3+", hint: None },
    Rule { id: "closure-assign", group: "form", title: "a nested function literal that assigns a local of the enclosing function", section: "3+", hint: Some("return the value from the literal and assign it outside") },
    Rule { id: "arity", group: "form", title: "a call or an adverb with the wrong number of arguments for the function", section: "3+", hint: None },
    Rule { id: "operand", group: "type", title: "an operand of the wrong type or rank for the operator", section: "4", hint: None },
    Rule { id: "shape-mismatch", group: "type", title: "operands whose shapes do not agree", section: "4", hint: Some("the shorter shape must be a prefix of the longer") },
    Rule { id: "shape-change", group: "type", title: "a variable whose type or shape differs between two assignments", section: "3, 4", hint: None },
    Rule { id: "cond-shape", group: "type", title: "a conditional whose branches have different shapes", section: "3, 4", hint: None },
    Rule { id: "cond-scalar", group: "type", title: "a condition that is not a scalar", section: "4", hint: Some("reduce it first, for example with &/ or |/") },
    Rule { id: "div-zero", group: "type", title: "division of an integer constant by the constant zero", section: "3", hint: None },
    Rule { id: "index-range", group: "type", title: "a constant index out of range", section: "4", hint: None },
    Rule { id: "array-shape", group: "type", title: "an array with more than 4 dimensions or with an empty dimension", section: "2", hint: Some("every dimension is a positive integer and the rank is 1 to 4") },
    Rule { id: "loop-state", group: "type", title: "an accumulator or loop state whose type or shape changes", section: "4", hint: Some("start from a value of the type the function returns (0.0 instead of 0)") },
    Rule { id: "extern-call", group: "type", title: "a call of an extern that does not match its signature", section: "9", hint: None },
    Rule { id: "fixed", group: "type", title: "a fixed-point number used in a way that is not defined: mixed with a float or another fixed type, an operator it does not have, a conversion that is not exact", section: "6", hint: Some("convert explicitly with .fixed(\"fixed<W,I>\";x) or .float(x); an int converts implicitly only as a constant that fits") },
    Rule { id: "param", group: "program", title: "a .param that cannot be declared, a parameter that a compiled function assigns, or a parameter used as a shape", section: "6", hint: Some("a parameter is a scalar register the host writes between runs; a function only reads it") },
    Rule { id: "stream-loop", group: "stream", title: "a loop whose number of iterations depends on the data (While, or Iterate with a count that is not a constant)", section: "5", hint: Some("a pipelined kernel needs a bounded loop: use n f:*a with a constant n, and stop early by holding the value") },
    Rule { id: "stream-state", group: "stream", title: "state that is read and written back through a multi-cycle operation (a float operation, or a division or remainder by a value that is not a constant)", section: "5", hint: Some("a float add, multiply or divide, or a divider, takes several cycles, so the next record has to wait for it: keep that state as an int or a fixed<W,I> updated by adds, multiplies and constant moduli, or let the host do the accumulation") },
    Rule { id: "stream-table", group: "stream", title: "a data-dependent index into an array of more than 1024 items", section: "5", hint: Some("a random access into a big table needs a memory with one port per access; make the table smaller, or index it with a constant window (x@idx with idx[a][b] = c + a + b is a shift register)") },
    Rule { id: "stream-block", group: "stream", title: "an entry that does more than 4096 element operations per call and is not a sliding window", section: "5", hint: Some("such a call is a block program: the kernel can start a new record only when the previous one has gone through its loops. Write the filter as a window, +/taps*x@idx with the index table k+!w, to stream it") },
];

/// The streamable tier's rules alone (the streamable tier); they are also the tail of `RULES`.
pub static STREAM_RULES: &[Rule] = &[
    Rule { id: "stream-loop", group: "stream", title: "a loop whose number of iterations depends on the data (While, or Iterate with a count that is not a constant)", section: "5", hint: Some("a pipelined kernel needs a bounded loop: use n f:*a with a constant n, and stop early by holding the value") },
    Rule { id: "stream-state", group: "stream", title: "state that is read and written back through a multi-cycle operation (a float operation, or a division or remainder by a value that is not a constant)", section: "5", hint: Some("a float add, multiply or divide, or a divider, takes several cycles, so the next record has to wait for it: keep that state as an int or a fixed<W,I> updated by adds, multiplies and constant moduli, or let the host do the accumulation") },
    Rule { id: "stream-table", group: "stream", title: "a data-dependent index into an array of more than 1024 items", section: "5", hint: Some("a random access into a big table needs a memory with one port per access; make the table smaller, or index it with a constant window (x@idx with idx[a][b] = c + a + b is a shift register)") },
    Rule { id: "stream-block", group: "stream", title: "an entry that does more than 4096 element operations per call and is not a sliding window", section: "5", hint: Some("such a call is a block program: the kernel can start a new record only when the previous one has gone through its loops. Write the filter as a window, +/taps*x@idx with the index table k+!w, to stream it") },
];

/// The groups of the error table, in order: (id, title).
pub static GROUPS: &[(&str, &str)] = &[
    ("program", "Program structure and declarations"),
    ("global", "Compile-time values"),
    ("value", "Values a run-time function may not hold"),
    ("length", "Results whose length depends on run-time data"),
    ("form", "Forms outside the profile"),
    ("type", "Types and shapes"),
    ("stream", "The streamable tier (--streamable)"),
];

pub static OPERAND_KINDS: &[(&str, &str)] = &[
    ("num", "an int or a float, scalar or array"),
    ("array", "an array (rank 1 to 4)"),
    ("rank1", "a rank-1 array"),
    ("rank2", "a rank-2 array"),
    ("int", "an int, scalar or array"),
    ("int-scalar", "an int scalar"),
    ("const-int", "an int scalar that is a compile-time constant"),
    ("const-shape", "a compile-time constant: a positive int, or a list literal of 1 to 4 positive ints"),
    ("index", "an int scalar or an int array; constants are range-checked"),
    ("same-shape", "a value of the same shape as the other operand"),
    ("amend-pair", "written v,i: a scalar value v and a scalar index or a rank-1 index list of known length"),
    ("predicate", "a monadic function (a name or a literal) that returns a scalar"),
];

pub static TYPE_RULES: &[(&str, &str)] = &[
    ("promote", "float if either operand is float, else int"),
    ("float", "always float"),
    ("int", "always int"),
    ("int-only", "both operands must be int; the result is int"),
    ("power", "int when both operands are int and the exponent is a non-negative constant, else float"),
    ("same", "the element type of the operand"),
    ("right", "the element type of the right operand"),
    ("left", "the element type of the left operand"),
    ("verb", "the element type the verb returns"),
    ("acc", "the accumulator: the fixed point of acc = f(acc, item), with at most one int to float promotion"),
    ("state", "the type of the starting value, which the verb must return unchanged"),
];

pub static SHAPE_RULES: &[(&str, &str)] = &[
    ("prefix", "scalar with scalar is a scalar; scalar with array is the array's shape; of two arrays the shorter shape must be a prefix of the longer, which is the result"),
    ("same", "the shape of the operand"),
    ("scalar", "a scalar"),
    ("item", "the operand's shape without its first dimension (a scalar stays a scalar)"),
    ("index", "a scalar index drops the first dimension; an index array of shape S gives S followed by the rest"),
    ("take", "[|n|] (the items cycle when |n| is larger than the length; negative n takes from the end)"),
    ("drop", "the first dimension shrinks by |n|, which must leave at least one item"),
    ("join", "scalar,scalar is [2]; rank 1 with a scalar is [n+1]; two arrays of the same rank whose items have the same shape are [n+m, ...]"),
    ("reshape", "the constant shape; the items cycle"),
    ("transpose", "[m,n] for an operand of shape [n,m]"),
    ("enumerate", "[n]"),
    ("wrap", "[1] followed by the operand's shape"),
    ("rank-vector", "[rank of the operand]"),
    ("amend", "the shape of the array"),
    ("each", "[n] followed by the shape the verb returns"),
    ("each-pair", "[n-1] followed by the shape the verb returns"),
    ("acc", "the shape of the accumulator"),
    ("scan", "[n] followed by the shape of the accumulator"),
    ("scan-seed", "[n+1] followed by the shape of the accumulator (the seed is the first item)"),
    ("state", "the shape of the starting value"),
];

/// An example program: `f::{body}` (or the whole `body` for a form) entered with `signature` is inside the profile and has the result type the
/// signature names. `inputs` is a JSON array of the host arguments a test calls it with (indices in range); None: a test makes some up.
pub struct Example {
    pub body: &'static str,
    pub signature: &'static str,
    pub inputs: Option<&'static str>,
}

/// An allowed form that is not an operator.
pub struct Form {
    pub id: &'static str,
    pub form: &'static str,
    pub note: &'static str,
    pub examples: &'static [Example],
}

pub static FORMS: &[Form] = &[
    Form { id: "number", form: "`1`, `2.5`, `1e3`", note: "an integer literal is int; a literal with `.` or `e` is float", examples: &[Example { body: "f::{x+1.5}", signature: "int -> float", inputs: None }] },
    Form { id: "list", form: "`[1 2 3]`, `[[1 2] [3 4]]`", note: "a rectangular list of numbers; any float makes it float", examples: &[Example { body: "f::{x+[1 2 3]}", signature: "int -> int[3]", inputs: None }, Example { body: "f::{x*[[1 2.5] [3 4]]}", signature: "int -> float[2,2]", inputs: None }] },
    Form { id: "evaluated-list", form: "`[;a;b;c]`", note: "elements of the same shape; ints are promoted when any element is float", examples: &[Example { body: "f::{[;x;y;x+y]}", signature: "int int -> int[3]", inputs: None }, Example { body: "f::{[;x;1.5]}", signature: "int -> float[2]", inputs: None }] },
    Form { id: "conditional", form: "`:[c;a;b]`, `:[c1;a:|c2;b;else]`", note: "the condition is a scalar; the branches have the same shape; int with float gives float", examples: &[Example { body: "f::{:[x>0;x;0-x]}", signature: "int -> int", inputs: None }, Example { body: "f::{:[x>0;1:|x<0;2.5;0]}", signature: "int -> float", inputs: None }] },
    Form { id: "sequence", form: "`{a;b;c}`", note: "statements run in order; the last is the result", examples: &[Example { body: "f::{[t];t::x*2;t+1}", signature: "int -> int", inputs: None }] },
    Form { id: "local", form: "`{[a b];a::...}`", note: "a local's type is the join of its assignments (int with float is float); its shape is fixed", examples: &[Example { body: "f::{[t];t::0;t::x%2;t}", signature: "int -> float", inputs: None }] },
    Form { id: "parameters", form: "`x y z`, or `f(a;b;c;d)::{...}`", note: "named parameters allow more than three", examples: &[Example { body: "f(a;b;c;d)::{(a+b)*c-d}", signature: "int int float int -> float", inputs: None }] },
    Form { id: "destructure", form: "`[a b]::v`", note: "v is an array with exactly one item per name; as the first statement the names are locals", examples: &[Example { body: "f::{[a b]::x;a+b}", signature: "int[2] -> int", inputs: None }] },
    Form { id: "global", form: "`name::expr` to a top-level name", note: "the type and shape of a global never change", examples: &[Example { body: "acc::0
f::{acc::acc+x;acc}", signature: "int -> int", inputs: None }] },
    Form { id: "call", form: "`g(a;b)`, `a g b`", note: "a compiled function, with every argument supplied; specialised per argument types", examples: &[Example { body: "sq::{x*x}
f::{sq(x)+sq(1.5)}", signature: "int -> float", inputs: None }] },
    Form { id: "extern-call", form: "`adc(0)`", note: "a name assigned `.extern(c_symbol;signature)`; a str parameter takes a string literal", examples: &[Example { body: "adc::.extern(\"read_adc\";\"int -> float\")
f::{adc(x)*2}", signature: "int -> float", inputs: None }, Example { body: "msg::.extern(\"log_msg\";\"str int -> void\")
f::{msg(\"tick\";x);x}", signature: "int -> int", inputs: None }] },
    Form { id: "verb", form: "`f'a` with f an operator, a function name or `{...}`", note: "a function literal is allowed only as the verb of an adverb or the predicate of While; it may read, but not assign, the locals of the function it is written in", examples: &[Example { body: "sq::{x*x}
f::{[k];k::2;(sq'x)+{x*k}'x}", signature: "int[3] -> int[3]", inputs: None }] },
];

/// A row of the operator table: how one glyph is used (`valence`: monad, dyad, adverb1 or adverb2).
pub struct Op {
    pub id: &'static str,
    pub glyph: &'static str,
    pub valence: &'static str,
    pub name: &'static str,
    pub allowed: bool,
    pub operands: &'static [&'static str],
    pub type_rule: Option<&'static str>,
    pub shape_rule: Option<&'static str>,
    pub rule: Option<&'static str>,
    pub note: &'static str,
    pub examples: &'static [Example],
}

pub static OPTABLE: &[Op] = &[
    Op { id: "add", glyph: "+", valence: "dyad", name: "Plus", allowed: true, operands: &["num", "num"], type_rule: Some("promote"), shape_rule: Some("prefix"), rule: None, note: "", examples: &[Example { body: "x+y", signature: "int int -> int", inputs: None }, Example { body: "x+y", signature: "int[3] float -> float[3]", inputs: None }, Example { body: "x+y", signature: "int[2,3] int[2] -> int[2,3]", inputs: None }] },
    Op { id: "sub", glyph: "-", valence: "dyad", name: "Minus", allowed: true, operands: &["num", "num"], type_rule: Some("promote"), shape_rule: Some("prefix"), rule: None, note: "", examples: &[Example { body: "x-y", signature: "float int[3] -> float[3]", inputs: None }] },
    Op { id: "mul", glyph: "*", valence: "dyad", name: "Times", allowed: true, operands: &["num", "num"], type_rule: Some("promote"), shape_rule: Some("prefix"), rule: None, note: "", examples: &[Example { body: "x*y", signature: "int[3] int[3] -> int[3]", inputs: None }] },
    Op { id: "div", glyph: "%", valence: "dyad", name: "Divide", allowed: true, operands: &["num", "num"], type_rule: Some("float"), shape_rule: Some("prefix"), rule: None, note: "always float; a constant int divided by the constant 0 is an error", examples: &[Example { body: "x%y", signature: "int int -> float", inputs: None }, Example { body: "x%y", signature: "int[3] int -> float[3]", inputs: None }] },
    Op { id: "idiv", glyph: ":%", valence: "dyad", name: "Integer-Divide", allowed: true, operands: &["int", "int"], type_rule: Some("int-only"), shape_rule: Some("prefix"), rule: None, note: "truncates toward zero; KlongPy also accepts floats here, the profile does not", examples: &[Example { body: "x:%y", signature: "int int -> int", inputs: None }, Example { body: "x:%2", signature: "int[3] -> int[3]", inputs: None }] },
    Op { id: "rem", glyph: "!", valence: "dyad", name: "Remainder", allowed: true, operands: &["num", "num"], type_rule: Some("promote"), shape_rule: Some("prefix"), rule: None, note: "the result has the sign of the dividend (C's %, fmod for floats)", examples: &[Example { body: "x!y", signature: "int int -> int", inputs: None }, Example { body: "x!y", signature: "float int -> float", inputs: None }] },
    Op { id: "pow", glyph: "^", valence: "dyad", name: "Power", allowed: true, operands: &["num", "num"], type_rule: Some("power"), shape_rule: Some("prefix"), rule: None, note: "KlongPy returns an int whenever the result is whole; the profile cannot depend on values", examples: &[Example { body: "x^2", signature: "int -> int", inputs: None }, Example { body: "x^y", signature: "int int -> float", inputs: None }, Example { body: "x^2", signature: "float[3] -> float[3]", inputs: None }] },
    Op { id: "max", glyph: "|", valence: "dyad", name: "Max/Or", allowed: true, operands: &["num", "num"], type_rule: Some("promote"), shape_rule: Some("prefix"), rule: None, note: "", examples: &[Example { body: "x|y", signature: "int float -> float", inputs: None }] },
    Op { id: "min", glyph: "&", valence: "dyad", name: "Min/And", allowed: true, operands: &["num", "num"], type_rule: Some("promote"), shape_rule: Some("prefix"), rule: None, note: "", examples: &[Example { body: "x&y", signature: "int[3] int[3] -> int[3]", inputs: None }] },
    Op { id: "lt", glyph: "<", valence: "dyad", name: "Less", allowed: true, operands: &["num", "num"], type_rule: Some("int"), shape_rule: Some("prefix"), rule: None, note: "", examples: &[Example { body: "x<y", signature: "float[3] float -> int[3]", inputs: None }] },
    Op { id: "gt", glyph: ">", valence: "dyad", name: "More", allowed: true, operands: &["num", "num"], type_rule: Some("int"), shape_rule: Some("prefix"), rule: None, note: "", examples: &[Example { body: "x>y", signature: "int int -> int", inputs: None }] },
    Op { id: "eq", glyph: "=", valence: "dyad", name: "Equal", allowed: true, operands: &["num", "num"], type_rule: Some("int"), shape_rule: Some("prefix"), rule: None, note: "", examples: &[Example { body: "x=y", signature: "int[3] int[3] -> int[3]", inputs: None }] },
    Op { id: "match", glyph: "~", valence: "dyad", name: "Match", allowed: true, operands: &["num", "same-shape"], type_rule: Some("int"), shape_rule: Some("scalar"), rule: None, note: "operands of different shapes are an error (KlongPy answers 0)", examples: &[Example { body: "x~y", signature: "int[3] int[3] -> int", inputs: None }, Example { body: "x~y", signature: "float float -> int", inputs: None }] },
    Op { id: "index", glyph: "@", valence: "dyad", name: "At/Index", allowed: true, operands: &["array", "index"], type_rule: Some("left"), shape_rule: Some("index"), rule: None, note: "a negative index counts from the end; f@x (apply) is not in the profile", examples: &[Example { body: "x@y", signature: "float[4] int -> float", inputs: Some("[[1.5, 2.5, 3.5, 4.5], 2]") }, Example { body: "x@y", signature: "float[4] int[2,3] -> float[2,3]", inputs: Some("[[1.5, 2.5, 3.5, 4.5], [[0, 1, 2], [3, 0, 1]]]") }, Example { body: "x@y", signature: "int[3,2] int -> int[2]", inputs: Some("[[[1, 2], [3, 4], [5, 6]], 1]") }, Example { body: "x@[0 2]", signature: "int[3] -> int[2]", inputs: None }] },
    Op { id: "take", glyph: "#", valence: "dyad", name: "Take", allowed: true, operands: &["const-int", "rank1"], type_rule: Some("right"), shape_rule: Some("take"), rule: None, note: "rank 1 only; -2#a is -(2#a): write (-2)#a", examples: &[Example { body: "2#x", signature: "int[4] -> int[2]", inputs: None }, Example { body: "(-2)#x", signature: "float[4] -> float[2]", inputs: None }, Example { body: "6#x", signature: "int[4] -> int[6]", inputs: None }] },
    Op { id: "drop", glyph: "_", valence: "dyad", name: "Drop", allowed: true, operands: &["const-int", "array"], type_rule: Some("right"), shape_rule: Some("drop"), rule: None, note: "negative n drops from the end", examples: &[Example { body: "1_x", signature: "int[4] -> int[3]", inputs: None }, Example { body: "(-1)_x", signature: "int[3,2] -> int[2,2]", inputs: None }] },
    Op { id: "join", glyph: ",", valence: "dyad", name: "Join", allowed: true, operands: &["num", "num"], type_rule: Some("promote"), shape_rule: Some("join"), rule: None, note: "joining ints with floats gives floats", examples: &[Example { body: "x,y", signature: "int[2] float[3] -> float[5]", inputs: None }, Example { body: "x,y", signature: "int int -> int[2]", inputs: None }, Example { body: "x,y", signature: "int[2,3] int[1,3] -> int[3,3]", inputs: None }, Example { body: "x,y", signature: "float int[2] -> float[3]", inputs: None }] },
    Op { id: "rotate", glyph: ":+", valence: "dyad", name: "Rotate", allowed: true, operands: &["int-scalar", "rank1"], type_rule: Some("right"), shape_rule: Some("same"), rule: None, note: "rank 1 only (KlongPy rotates the flattened data of a matrix); negative n rotates left", examples: &[Example { body: "y:+x", signature: "int[4] int -> int[4]", inputs: None }, Example { body: "(-1):+x", signature: "float[3] -> float[3]", inputs: None }] },
    Op { id: "reshape", glyph: ":^", valence: "dyad", name: "Reshape", allowed: true, operands: &["const-shape", "num"], type_rule: Some("right"), shape_rule: Some("reshape"), rule: None, note: "the row-major data cycles; a scalar n takes a scalar or a rank-1 array only (write [n]:^a for a matrix)", examples: &[Example { body: "[2 3]:^x", signature: "int[6] -> int[2,3]", inputs: None }, Example { body: "4:^x", signature: "int[2] -> int[4]", inputs: None }, Example { body: "[2 2]:^x", signature: "float -> float[2,2]", inputs: None }] },
    Op { id: "amend", glyph: ":=", valence: "dyad", name: "Amend", allowed: true, operands: &["rank1", "amend-pair"], type_rule: Some("promote"), shape_rule: Some("amend"), rule: None, note: "g::g:=v,i on a global is an in-place store", examples: &[Example { body: "x:=y,0", signature: "float[4] float -> float[4]", inputs: None }, Example { body: "x:=y,[0 2]", signature: "int[4] int -> int[4]", inputs: None }, Example { body: "x:=y,z", signature: "int[4] int int -> int[4]", inputs: Some("[[1, 2, 3, 4], 9, 1]") }] },
    Op { id: "define", glyph: "::", valence: "dyad", name: "Define", allowed: true, operands: &[], type_rule: Some("right"), shape_rule: Some("same"), rule: None, note: "name::expr assigns a local or a global and is the value assigned; see the forms table", examples: &[] },
    Op { id: "neg", glyph: "-", valence: "monad", name: "Negate", allowed: true, operands: &["num"], type_rule: Some("same"), shape_rule: Some("same"), rule: None, note: "a leading - negates the whole expression to its right", examples: &[Example { body: "-x", signature: "int[3] -> int[3]", inputs: None }] },
    Op { id: "floor", glyph: "_", valence: "monad", name: "Floor", allowed: true, operands: &["num"], type_rule: Some("int"), shape_rule: Some("same"), rule: None, note: "rounds toward minus infinity", examples: &[Example { body: "_x", signature: "float[3] -> int[3]", inputs: None }] },
    Op { id: "recip", glyph: "%", valence: "monad", name: "Reciprocal", allowed: true, operands: &["num"], type_rule: Some("float"), shape_rule: Some("same"), rule: None, note: "", examples: &[Example { body: "%x", signature: "int -> float", inputs: None }] },
    Op { id: "not", glyph: "~", valence: "monad", name: "Not", allowed: true, operands: &["num"], type_rule: Some("int"), shape_rule: Some("same"), rule: None, note: "", examples: &[Example { body: "~x", signature: "int[3] -> int[3]", inputs: None }] },
    Op { id: "size", glyph: "#", valence: "monad", name: "Size", allowed: true, operands: &["array"], type_rule: Some("int"), shape_rule: Some("scalar"), rule: None, note: "the first dimension, a compile-time constant", examples: &[Example { body: "#x", signature: "int[3,2] -> int", inputs: None }] },
    Op { id: "shape", glyph: "^", valence: "monad", name: "Shape", allowed: true, operands: &["array"], type_rule: Some("int"), shape_rule: Some("rank-vector"), rule: None, note: "a compile-time constant", examples: &[Example { body: "^x", signature: "int[3,2] -> int[2]", inputs: None }] },
    Op { id: "first", glyph: "*", valence: "monad", name: "First", allowed: true, operands: &["num"], type_rule: Some("same"), shape_rule: Some("item"), rule: None, note: "", examples: &[Example { body: "*x", signature: "int[3,2] -> int[2]", inputs: None }, Example { body: "*x", signature: "float[3] -> float", inputs: None }] },
    Op { id: "reverse", glyph: "|", valence: "monad", name: "Reverse", allowed: true, operands: &["num"], type_rule: Some("same"), shape_rule: Some("same"), rule: None, note: "reverses along the first dimension", examples: &[Example { body: "|x", signature: "int[3] -> int[3]", inputs: None }, Example { body: "|x", signature: "int[3,2] -> int[3,2]", inputs: None }] },
    Op { id: "transpose", glyph: "+", valence: "monad", name: "Transpose", allowed: true, operands: &["rank2"], type_rule: Some("same"), shape_rule: Some("transpose"), rule: None, note: "", examples: &[Example { body: "+x", signature: "int[2,3] -> int[3,2]", inputs: None }] },
    Op { id: "enumerate", glyph: "!", valence: "monad", name: "Enumerate", allowed: true, operands: &["const-int"], type_rule: Some("int"), shape_rule: Some("enumerate"), rule: None, note: "n must be a positive constant", examples: &[Example { body: "x+!3", signature: "int -> int[3]", inputs: None }] },
    Op { id: "list", glyph: ",", valence: "monad", name: "List", allowed: true, operands: &["num"], type_rule: Some("same"), shape_rule: Some("wrap"), rule: None, note: "", examples: &[Example { body: ",x", signature: "int -> int[1]", inputs: None }, Example { body: ",x", signature: "int[3] -> int[1,3]", inputs: None }] },
    Op { id: "each", glyph: "'", valence: "adverb1", name: "Each", allowed: true, operands: &["array"], type_rule: Some("verb"), shape_rule: Some("each"), rule: None, note: "f is called on each item along the first dimension", examples: &[Example { body: "{x*2}'x", signature: "int[3] -> int[3]", inputs: None }, Example { body: "-'x", signature: "float[3] -> float[3]", inputs: None }, Example { body: "{x,x}'x", signature: "int[3] -> int[3,2]", inputs: None }] },
    Op { id: "each2", glyph: "'", valence: "adverb2", name: "Each-2", allowed: true, operands: &["array", "array"], type_rule: Some("verb"), shape_rule: Some("each"), rule: None, note: "both operands must have the same length (KlongPy stops at the shorter)", examples: &[Example { body: "x{x+y}'y", signature: "int[3] float[3] -> float[3]", inputs: None }] },
    Op { id: "over", glyph: "/", valence: "adverb1", name: "Over", allowed: true, operands: &["array"], type_rule: Some("acc"), shape_rule: Some("acc"), rule: None, note: "the accumulator starts from the first item", examples: &[Example { body: "+/x", signature: "int[4] -> int", inputs: None }, Example { body: "+/x", signature: "float[3,2] -> float[2]", inputs: None }, Example { body: "{x+y%2}/x", signature: "int[4] -> float", inputs: None }] },
    Op { id: "over-seed", glyph: "/", valence: "adverb2", name: "Over-Neutral", allowed: true, operands: &["num", "array"], type_rule: Some("acc"), shape_rule: Some("acc"), rule: None, note: "the accumulator starts from the seed", examples: &[Example { body: "0.0+/x", signature: "int[4] -> float", inputs: None }, Example { body: "y+/x", signature: "int[3] int -> int", inputs: None }] },
    Op { id: "scan", glyph: "\\", valence: "adverb1", name: "Scan-Over", allowed: true, operands: &["array"], type_rule: Some("acc"), shape_rule: Some("scan"), rule: None, note: "", examples: &[Example { body: "+\\x", signature: "int[4] -> int[4]", inputs: None }] },
    Op { id: "scan-seed", glyph: "\\", valence: "adverb2", name: "Scan-Over-Neutral", allowed: true, operands: &["num", "array"], type_rule: Some("acc"), shape_rule: Some("scan-seed"), rule: None, note: "returns n+1 items", examples: &[Example { body: "0+\\x", signature: "int[4] -> int[5]", inputs: None }] },
    Op { id: "each-pair", glyph: ":'", valence: "adverb1", name: "Each-Pair", allowed: true, operands: &["array"], type_rule: Some("verb"), shape_rule: Some("each-pair"), rule: None, note: "needs at least two items", examples: &[Example { body: "-:'x", signature: "int[4] -> int[3]", inputs: None }] },
    Op { id: "iterate", glyph: ":*", valence: "adverb2", name: "Iterate", allowed: true, operands: &["int-scalar", "num"], type_rule: Some("state"), shape_rule: Some("state"), rule: None, note: "n f:*a applies f n times", examples: &[Example { body: "3{x*2}:*x", signature: "int -> int", inputs: None }, Example { body: "y{x*0.5}:*x", signature: "float int -> float", inputs: Some("[8.0, 2]") }] },
    Op { id: "while", glyph: ":~", valence: "adverb2", name: "While", allowed: true, operands: &["predicate", "num"], type_rule: Some("state"), shape_rule: Some("state"), rule: None, note: "p f:~a applies f while p holds", examples: &[Example { body: "{x<100}{x*2}:~x", signature: "int -> int", inputs: Some("[3]") }] },
    Op { id: "where", glyph: "&", valence: "monad", name: "Expand/Where", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("where"), note: "result length depends on the data", examples: &[] },
    Op { id: "find", glyph: "?", valence: "dyad", name: "Find", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("find"), note: "result length depends on the data", examples: &[] },
    Op { id: "range", glyph: "?", valence: "monad", name: "Range", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("range"), note: "result length depends on the data", examples: &[] },
    Op { id: "group", glyph: "=", valence: "monad", name: "Group", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("group"), note: "result is ragged", examples: &[] },
    Op { id: "grade-up", glyph: "<", valence: "monad", name: "Grade-Up", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("grade"), note: "fixed length; not in the profile (it needs a sort routine in the runtime)", examples: &[] },
    Op { id: "grade-down", glyph: ">", valence: "monad", name: "Grade-Down", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("grade"), note: "fixed length; not in the profile (it needs a sort routine in the runtime)", examples: &[] },
    Op { id: "split", glyph: ":#", valence: "dyad", name: "Split", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("split"), note: "result is ragged", examples: &[] },
    Op { id: "cut", glyph: ":_", valence: "dyad", name: "Cut", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("cut"), note: "result is ragged", examples: &[] },
    Op { id: "converge", glyph: ":~", valence: "adverb1", name: "Converge", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("converge"), note: "number of steps unknown", examples: &[] },
    Op { id: "scan-converge", glyph: "\\~", valence: "adverb1", name: "Scan-Converging", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("scan-dynamic"), note: "result length unknown", examples: &[] },
    Op { id: "scan-while", glyph: "\\~", valence: "adverb2", name: "Scan-While", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("scan-dynamic"), note: "result length unknown", examples: &[] },
    Op { id: "scan-iterate", glyph: "\\*", valence: "adverb2", name: "Scan-Iterating", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("scan-dynamic"), note: "fixed length for a constant count, but not in the profile", examples: &[] },
    Op { id: "format", glyph: "$", valence: "monad", name: "Format", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("string"), note: "makes a string", examples: &[] },
    Op { id: "format2", glyph: "$", valence: "dyad", name: "Format2", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("string"), note: "makes a string", examples: &[] },
    Op { id: "form", glyph: ":$", valence: "dyad", name: "Form", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("string"), note: "reads a string", examples: &[] },
    Op { id: "char", glyph: ":#", valence: "monad", name: "Char", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("string"), note: "makes a character", examples: &[] },
    Op { id: "atom", glyph: "@", valence: "monad", name: "Atom", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("unsupported-op"), note: "always a constant in a typed program", examples: &[] },
    Op { id: "undefined", glyph: ":_", valence: "monad", name: "Undefined", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("unsupported-op"), note: ":undefined is not a run-time value", examples: &[] },
    Op { id: "amend-in-depth", glyph: ":-", valence: "dyad", name: "Amend-in-Depth", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("unsupported-op"), note: "not in the profile", examples: &[] },
    Op { id: "index-in-depth", glyph: ":@", valence: "dyad", name: "Index-in-Depth", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("unsupported-op"), note: "not in the profile; write (a@i)@j", examples: &[] },
    Op { id: "each-left", glyph: ":\\", valence: "adverb2", name: "Each-Left", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("unsupported-op"), note: "not in the profile; write {a f x}'b", examples: &[] },
    Op { id: "each-right", glyph: ":/", valence: "adverb2", name: "Each-Right", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("unsupported-op"), note: "not in the profile; write {x f b}'a", examples: &[] },
    Op { id: "each-index", glyph: "@'", valence: "adverb1", name: "Each-Index", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("unsupported-op"), note: "not in the profile", examples: &[] },
    Op { id: "track", glyph: "˙", valence: "monad", name: "Track", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("unsupported-op"), note: "autograd", examples: &[] },
    Op { id: "grad", glyph: "∇", valence: "monad", name: "Grad", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("unsupported-op"), note: "autograd", examples: &[] },
    Op { id: "grad2", glyph: "∇", valence: "dyad", name: "Grad", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("unsupported-op"), note: "autograd", examples: &[] },
    Op { id: "jacobian", glyph: "∂", valence: "dyad", name: "Jacobian", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("unsupported-op"), note: "autograd", examples: &[] },
    Op { id: "autograd", glyph: ":>", valence: "dyad", name: "Autograd", allowed: false, operands: &[], type_rule: None, shape_rule: None, rule: Some("unsupported-op"), note: "autograd", examples: &[] },
];

pub fn lookup(glyph: &str, valence: &str) -> Option<&'static Op> {
    OPTABLE.iter().find(|o| o.glyph == glyph && o.valence == valence)
}

pub fn rule(id: &str) -> Option<&'static Rule> {
    RULES.iter().find(|r| r.id == id)
}
