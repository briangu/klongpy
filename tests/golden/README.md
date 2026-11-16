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
