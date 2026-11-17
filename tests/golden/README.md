# Golden parse trees

Snapshots of the canonical parse-tree dump (`klongpy/dump.py`, documented in
`conformance/README.md`) for the current parser. They pin what
the front end produces today so that later parser work (token parser onward) can
be compared against it. `tests/test_golden_trees.py` re-dumps every source and
asserts equality, one subtest per file.

Layout: `tests/golden/<group>/<name>.sexp`

| group | sources |
|---|---|
| `language` | every `.kg` under `tests/kgtests/language/` |
| `lib` | every `klongpy/lib/*.kg` |
| `fixtures` | `tests/golden/fixtures/*.kg`: small inputs, one per row of the observed table (`s2_*`), per quirk of the behaviour-preserved table (`s6_*`), for other forms (`x_*`) and for the bracket literals (`brk_*`) |

Each file holds one top-level form per statement. An input that raises while
parsing is recorded as `(error ClassName)`. Fixture sources are stored without
a trailing newline unless the newline is the point of the test.

Each source is parsed on a fresh interpreter, without evaluating it.

The snapshots were taken on the legacy character-cursor parser and are now
checked against the default token parser (`klongpy/lexer.py`), which must
reproduce them byte for byte. `tests/test_parser_tokens.py` additionally
compares the two parsers directly on every source. The `brk_*` fixtures are
the exception: they record the token parser's result for string and char
literals whose content is a bracket, which the legacy parser
(`KLONGPY_PARSER=legacy`) still misreads as that bracket.

To regenerate after an intended change to the parser or the dump:

    .venv/bin/python tests/golden/regen.py

and review the diff. `--check` lists the files that differ. To inspect one
file: `python -m klongpy.dump FILE`.

The token parser was changed on purpose in three places that hold at
every strict level, and the legacy parser was left as it is; five fixtures
record the new result (`tests/test_parser_tokens.py::FIXTURES_SYNTAX_CHECKS`, which
lists four since the evaluated dictionary literal gave `s2_dict_empty_slot` a meaning):
`s2_cond_ws_before_second_semi` and `s2_cond_ws_both` (whitespace before the
first `;` of a conditional is accepted), `s2_space_before_adverb` (an adverb
that does not follow its verb is a syntax error when parsed), and
`s2_dict_empty_slot` and `s6_unterminated_dict` (a dictionary literal whose
entries are not pairs is a syntax error, where the reader used to fail with an
`IndexError` or a `TypeError`). The warnings of the syntax checks do not appear in
the trees: at strict level 0, where the snapshots are taken, the tree of a
malformed input is what it always was.

The arity change altered what `(fn N ...)` prints for a function literal: N is the
highest of x, y, z the body mentions (`{-x}` is `(fn 1 ...)`, `{y}` is
`(fn 2 ...)`), where it used to be a count of the names an incomplete scan
found. 17 snapshots were regenerated (16 of `language` and `lib`, and
`fixtures/x_nested_functions`); 210 lines differ and each differs in that
number only. Both parsers agree, because the arity function is shared.

Destructuring, named parameters and the evaluated dictionary literal are syntax that only the token parser has:
destructuring `[a b]::v` (`(destructure (a b) R)`), named parameters
`f(a;b)::{...}` (`(def (sym f) (fn 2 (params a b) ...))`) and the evaluated
dictionary literal `:{;k;v}` (`(edict K V ...)`). The `dst_*`, `named_*` and
`dict_*` fixtures record it, with the inputs next to it that are not that
syntax (a nested or duplicate name, `x` as a parameter, an odd number of
expressions, ...). Three existing snapshots changed, the three that held the
place of that syntax in the observed table: `s2_destructuring_slot`,
`s2_named_params_slot` and `s2_dict_empty_slot`. No other snapshot changed. A
dictionary literal nested in a list or dictionary literal is now a dictionary
value where it was an unevaluated call; both dump as `(dict ...)`, so
`x_dict_nested` is unchanged. `tests/test_parser_tokens.py` leaves these files
out of the comparison with the legacy parser (`is_new_syntax`).
