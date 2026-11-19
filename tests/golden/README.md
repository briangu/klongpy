# Golden parse trees

Snapshots of the canonical parse-tree dump (`kgpy ast`: the native engine's `ast_text`, documented in
`conformance/README.md`). They pin what the front end produces so that a change to the
parser can be compared against them, and so that another implementation's parser can be
checked byte for byte (`python conformance/run.py --trees --cmd CMD`). `tests/test_golden_trees.py`
re-dumps every source and asserts equality, one subtest per file.

Layout: `tests/golden/<group>/<name>.sexp`

| group | sources |
|---|---|
| `language` | every `.kg` under `tests/kgtests/language/` |
| `lib` | every `klongpy/lib/*.kg` |
| `fixtures` | `tests/golden/fixtures/*.kg`: small inputs, grouped by name prefix (below) |

Each file holds one top-level form per statement. An input that raises while parsing is
recorded as `(error ClassName)`. Fixture sources are stored without a trailing newline
unless the newline is the point of the test. Each source is parsed without evaluating it.

Fixture prefixes: `s2_*` one input per row of a table of syntax forms and their results,
`s6_*` one per quirk of the syntax that is kept as it is, `x_*` other forms, `brk_*`
string and char literals whose content is a bracket, `dst_*` destructuring, `named_*`
named parameters and `dict_*` the evaluated dictionary literal.

The syntax of destructuring `[a b]::v` is `(destructure (a b) R)`, named parameters
`f(a;b)::{...}` are `(def (sym f) (fn 2 (params a b) ...))` and the evaluated dictionary
literal `:{;k;v}` is `(edict K V ...)`. Four fixtures record syntax errors that the reader
reports at strict level 1: `s2_cond_ws_before_second_semi` and `s2_cond_ws_both` accept
whitespace before the first `;` of a conditional, `s2_space_before_adverb` is an adverb that
does not follow its verb, and `s6_unterminated_dict` is a dictionary literal whose entries are
not pairs. The warnings of these checks do not appear in the trees: the snapshots are taken
at strict level 0, where the tree of a malformed input is the plain one.

`(fn N ...)` prints N as the highest of x, y, z that the body mentions: `{-x}` is
`(fn 1 ...)` and `{y}` is `(fn 2 ...)`.

To regenerate after an intended change to the reader or the dump:

    .venv/bin/python tests/golden/regen.py

and review the diff. `--check` lists the files that differ. To inspect one file:
`kgpy ast FILE`.
