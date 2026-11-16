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
| `fixtures` | `tests/golden/fixtures/*.kg`: small inputs, one per row of the observed table (`s2_*`), per quirk of the behaviour-preserved table (`s6_*`) and for other forms (`x_*`) |

Each file holds one top-level form per statement. An input that raises while
parsing is recorded as `(error ClassName)`. Fixture sources are stored without
a trailing newline unless the newline is the point of the test.

Each source is parsed on a fresh interpreter, without evaluating it.

To regenerate after an intended change to the parser or the dump:

    .venv/bin/python tests/golden/regen.py

and review the diff. `--check` lists the files that differ. To inspect one
file: `python -m klongpy.dump FILE`.
