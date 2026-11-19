# A backtest in modules

A program of twelve files: seeded synthetic prices, two trading strategies behind one interface, a portfolio, and a report. It shows how a Klong program is split into modules: files that keep their helpers and their state private and offer a short list of names.

Run it from this directory:

```
klong main.kg
```

It prints the text in `EXPECTED.txt`. Each program in `errors/` stops with the error named for it (`klong errors/readonly.kg`), and `classic_load.kg` shows `.l` on a file that is a module: it binds the module to the file's name, as `name::.import("name")` does.

## The module system in four lines

- `m::.import("./lib/stats")` runs the file in a scope of its own and gives back the dictionary the file passed to `.export`. Nothing is written into your globals, and the file's top-level names are private to it.
- A file ends with `.export(:{;"name";value;...})`. The key may differ from the name inside the file (`"seed"` for `reset` in `lib/rng.kg`).
- `m.name` is `m@"name"`, everywhere: `m.mean(v)`, `cfg.run.days::90`, `stats.mean'windows`, `stats.scale(;2.0)`. A missing name is an error that says which dictionary lacks which key.
- What `.import` returns is read-only: `m.name::v` is an error. A dictionary stored inside it is an ordinary shared dictionary.

Importing the same file again gives the same module (the cache key is the real path of the file), so a state that a module keeps is shared by every importer, for as long as some value refers to the module: its dictionary or a function taken from it. A module nothing refers to is freed, and the next import runs the file again; `m::0` releases a library. To keep a module between calls of a function, hold it in a global. A circular import and a file with no `.export` are errors.

## Paths

`./x` and `../x` are relative to the file that contains the import. A name without a dot-slash (`"csv"`) is looked for in the directories of `KLONGPATH` and in the library that ships with Klong, never in the working directory. `.kg` is added when the name has no extension.

## Files

| File | What it shows |
|---|---|
| `main.kg` | the program: relative imports; writing `cfg.run.days::90` and `cfg.strategy.params.crossover.fast::4` through `a.b`; a diamond of imports seen from the top; another spelling of a path giving the same module; a private helper that is not reachable; qualified names as the verb of an adverb, in a projection, as a value, in a dictionary, on a parameter; strategies picked by name from a registry; a classic file loaded beside the modules |
| `config.kg` | nested defaults: `defaults()` builds a fresh dictionary per call, so a program may change its own copy |
| `lib/rng.kg` | a seeded generator: private state behind exported functions, a name exposed under another name |
| `lib/stats.kg` | pure array functions with private helpers; also what the static compiler reads (`static/ma4.kg`) |
| `lib/series.kg` | a module that imports a module (`./stats`) |
| `data/feed.kg` | selection and rename on import: `[reseed noise]::.import("../lib/rng";["seed" "noise"])` |
| `book/portfolio.kg` | a dictionary passed through functions, read and written with `a.b`; every function returns a new dictionary |
| `strategy/*.kg` | two strategies with one interface (`name`, `signal`, `step`, `describe`); a registry holding modules as values |
| `report/table.kg` | the text of the report; imports `csv` inside a function, so the library loads when the function runs and is let go when it returns; re-exports two functions of `stats` under other names |
| `legacy.kg` | a classic file, loaded with `.l`: its names are plain globals and do not meet the modules' |
| `static/ma4.kg` | the static compiler through a module: `klong compile c static/ma4.kg` resolves `stats.ma4` at compile time |
| `errors/*.kg` | one program per error: read-only import, missing entry, private helper, cycle, no `.export`, undefined name, a name that is not a dictionary |

## What to copy into your own program

Give each file one job, put the names a caller needs in one `.export` at the end, and import with a relative path. Keep state behind functions: an exported variable is its value at the moment of `.export`, so `m.counter` does not follow later changes; an exported function does.
