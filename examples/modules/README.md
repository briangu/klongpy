# A backtest in modules

A program of twelve files: seeded synthetic prices, two trading strategies behind one interface, a portfolio, and a report. It shows how a Klong program is split into modules: files that keep their helpers and their state private and offer a short list of names.

Run it from this directory:

```
kg main.kg
```

`kgpy main.kg` prints the same.

It prints the text in `EXPECTED.txt`. Each program in `errors/` stops with the error named for it (`kg errors/readonly.kg`; the messages are in `EXPECTED-ERRORS.txt`), and `classic_load.kg` shows `.l` on a file that is a module: it binds the module to the file's name and returns it.

## The module system

A file is a module when it assigns the variable `export` at its top level. The variable holds a dictionary written as a literal: the names the module offers, each key a string. The key may differ from the name inside the file (`"seed"` for `reset` in `lib/rng.kg`).

```klong
mean(v)::{(+/v)%#v}
spread(v)::{mean(v)}
export:::{;"mean";mean;"sd";spread}
```

The file runs in a scope of its own, once. Its other top-level names are private, and when the file has finished the interface is read and made read-only. `.l` loads it, in three forms:

| Klong | Python |
|---|---|
| `.l("stats")` binds the module to `stats`, the file's stem | `import stats` |
| `.l("stats";:s)` binds it to `s` and not to `stats` | `import stats as s` |
| `.l("stats";["mean" "sd"])` binds `mean` and `sd` as plain names | `from stats import mean, sd` |

`.l` returns the module. A missing entry in the third form is an error that names it. A second argument on a file that is not a module is an error.

- `m.name` is `m@"name"`, everywhere: `m.mean(v)`, `cfg.run.days::90`, `stats.mean'windows`, `stats.scale(;2.0)`. A missing name is an error that says which dictionary lacks which key.
- A module is read-only from outside: `m.name::v` is an error. A dictionary stored inside it is an ordinary shared dictionary.
- Loading the same file again gives the same module (the cache key is the real path of the file), so a state that a module keeps is shared by every loader, for as long as some value refers to the module: its dictionary or a function taken from it. A module nothing refers to is freed, and the next load runs the file again; `m::0` releases a library. To keep a module between calls of a function, hold it in a global.
- A circular load is an error. Under `:s`, the alias must not already hold a value at strict level 0 or 1, where `:s` reads the variable; use a name not yet in use.

## Paths

A bare name (`"csv"`) is looked for in the directory of the file that contains the `.l`, then in the directories of `KLONGPATH`, then in the library that ships with Klong. `./x` and `../x` are relative to the file that contains the `.l`. `.kg` is added when the name has no extension. A local file with the name of a library file shadows it, with a warning (lint rule W023).

## Files

| File | What it shows |
|---|---|
| `main.kg` | the program: relative loads, in the three forms of `.l`; writing `cfg.run.days::90` and `cfg.strategy.params.crossover.fast::4` through `a.b`; a diamond of loads seen from the top; another spelling of a path giving the same module; a private helper that is not reachable; qualified names as the verb of an adverb, in a projection, as a value, in a dictionary, on a parameter; strategies picked by name from a registry; a classic file loaded beside the modules |
| `config.kg` | nested defaults: `defaults()` builds a fresh dictionary per call, so a program may change its own copy |
| `lib/rng.kg` | a seeded generator: private state behind exported functions, a name exposed under another name |
| `lib/stats.kg` | pure array functions with private helpers; also what the static compiler reads (`static/ma4.kg`) |
| `lib/series.kg` | a module that loads a module (`./stats`) |
| `data/feed.kg` | the names form: `.l("../lib/rng";["seed" "noise"])`, then a plain rename |
| `book/portfolio.kg` | a dictionary passed through functions, read and written with `a.b`; every function returns a new dictionary |
| `strategy/*.kg` | two strategies with one interface (`name`, `signal`, `step`, `describe`); a registry holding modules as values |
| `report/table.kg` | the text of the report; loads `csv` inside a function, with the alias form,, so the library loads when the function runs and is let go when it returns; re-exports two functions of `stats` under other names |
| `legacy.kg` | a classic file, loaded with `.l`: its names are plain globals and do not meet the modules' |
| `static/ma4.kg` | the static compiler through a module: `kg compile c static/ma4.kg` resolves `stats.ma4` at compile time |
| `errors/*.kg` | one program per error: read-only module, missing entry, private helper, a missing entry in the names form, cycle, an alias on a classic file, a file name that is not a name, a discarded `.extern`, undefined name, a name that is not a dictionary |

## What to copy into your own program

Give each file one job, put the names a caller needs in one `export` dictionary at the end, and load with a relative path. Keep state behind functions: an exported variable is its value at the moment the file finishes loading, so `m.counter` does not follow later changes; an exported function does.
