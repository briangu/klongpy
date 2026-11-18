# Golden parse trees

Snapshots of the canonical parse-tree dump (`klongpy/dump.py`, documented in
`conformance/README.md`). They pin what the front end produces so that a change to the
parser can be compared against them, and so that another implementation's parser can be
checked byte for byte. `tests/test_golden_trees.py` re-dumps every source and asserts
equality, one subtest per file.

Layout: `tests/golden/<group>/<name>.sexp`

| group | sources |
|---|---|
| `language` | every `.kg` under `tests/kgtests/language/` |
| `lib` | every `klongpy/lib/*.kg` |
| `fixtures` | `tests/golden/fixtures/*.kg`: small inputs, grouped by name prefix (below) |

Each file holds one top-level form per statement. An input that raises while parsing is
recorded as `(error ClassName)`. Fixture sources are stored without a trailing newline
unless the newline is the point of the test. Each source is parsed on a fresh
interpreter, without evaluating it.

Fixture prefixes: `s2_*` one input per row of a table of syntax forms and their results,
`s6_*` one per quirk of the parser that is kept as it is, `x_*` other forms, `brk_*`
string and char literals whose content is a bracket, `dst_*` destructuring, `named_*`
named parameters and `dict_*` the evaluated dictionary literal.

The default token parser (`klongpy/lexer.py`) must reproduce every file byte for byte.
`tests/test_parser_tokens.py` also compares the token parser with the legacy parser
(`KLONGPY_PARSER=legacy`) on every source, except for the files that hold syntax only the
token parser has:

* destructuring `[a b]::v` is `(destructure (a b) R)`;
* named parameters `f(a;b)::{...}` are `(def (sym f) (fn 2 (params a b) ...))`;
* the evaluated dictionary literal `:{;k;v}` is `(edict K V ...)`;
* the `brk_*` fixtures, which the legacy parser misreads as the bracket.

Four fixtures record syntax errors that the token parser reports and the legacy parser
accepts (`tests/test_parser_tokens.py::FIXTURES_SYNTAX_CHECKS`):
`s2_cond_ws_before_second_semi` and `s2_cond_ws_both` accept whitespace before the first
`;` of a conditional, `s2_space_before_adverb` is an adverb that does not follow its verb,
and `s6_unterminated_dict` is a dictionary literal whose entries are not pairs. The
warnings of these checks do not appear in the trees: the snapshots are taken at strict
level 0, where the tree of a malformed input is the plain one.

`(fn N ...)` prints N as the highest of x, y, z that the body mentions: `{-x}` is
`(fn 1 ...)` and `{y}` is `(fn 2 ...)`.

To regenerate after an intended change to the parser or the dump:

    .venv/bin/python tests/golden/regen.py

and review the diff. `--check` lists the files that differ. To inspect one file:
`python -m klongpy.dump FILE`.
