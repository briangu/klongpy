# Golden parse trees

Snapshots of the canonical parse-tree dump (`klongpy/dump.py`, design in
`tickets/design/frontend.md` section 7) for the current parser. They pin what
the front end produces today so that later parser work (ticket 030 onward) can
be compared against it. `tests/test_golden_trees.py` re-dumps every source and
asserts equality, one subtest per file.

Layout: `tests/golden/<group>/<name>.sexp`

| group | sources |
|---|---|
| `language` | every `.kg` under `tests/kgtests/language/` |
| `lib` | every `klongpy/lib/*.kg` |
| `fixtures` | `tests/golden/fixtures/*.kg`: small inputs, one per row of the observed table (`s2_*`), per quirk of the behaviour-preserved table (`s6_*`), for other forms (`x_*`) and for the bracket literals of ticket 023 (`t023_*`) |

Each file holds one top-level form per statement. An input that raises while
parsing is recorded as `(error ClassName)`. Fixture sources are stored without
a trailing newline unless the newline is the point of the test.

Each source is parsed on a fresh interpreter, without evaluating it.

The snapshots were taken on the legacy character-cursor parser and are now
checked against the default token parser (`klongpy/lexer.py`), which must
reproduce them byte for byte. `tests/test_parser_tokens.py` additionally
compares the two parsers directly on every source. The `t023_*` fixtures are
the exception: they record the token parser's result for string and char
literals whose content is a bracket, which the legacy parser
(`KLONGPY_PARSER=legacy`) still misreads as that bracket.

To regenerate after an intended change to the parser or the dump:

    .venv/bin/python tests/golden/regen.py

and review the diff. `--check` lists the files that differ. To inspect one
file: `python -m klongpy.dump FILE`.

Ticket 031 changed the token parser on purpose in three places that hold at
every strict level, and the legacy parser was left as it is; five fixtures
record the new result (`tests/test_parser_tokens.py::FIXTURES_031`):
`s2_cond_ws_before_second_semi` and `s2_cond_ws_both` (whitespace before the
first `;` of a conditional is accepted), `s2_space_before_adverb` (an adverb
that does not follow its verb is a syntax error when parsed), and
`s2_dict_empty_slot` and `s6_unterminated_dict` (a dictionary literal whose
entries are not pairs is a syntax error, where the reader used to fail with an
`IndexError` or a `TypeError`). The warnings of that ticket do not appear in
the trees: at strict level 0, where the snapshots are taken, the tree of a
malformed input is what it always was.

Ticket 026 changed what `(fn N ...)` prints for a function literal: N is the
highest of x, y, z the body mentions (`{-x}` is `(fn 1 ...)`, `{y}` is
`(fn 2 ...)`), where it used to be a count of the names an incomplete scan
found. 17 snapshots were regenerated (16 of `language` and `lib`, and
`fixtures/x_nested_functions`); 210 lines differ and each differs in that
number only. Both parsers agree, because the arity function is shared.
