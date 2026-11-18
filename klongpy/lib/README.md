# Library code

These Klong library files (`math.kg`, `csv.kg`, `set.kg` and others) are loaded with `.l("name")`. `kgpy` adds this directory to `KLONGPATH`, so `.l("math")` works in the REPL and in `kgpy -e`. Set the `KLONGPATH` environment variable to look elsewhere, or to add directories.

The directory is inside the package so that `pip install` includes it.

## Acknowledgement

These files come from the Klong library in the [original Klong distribution](https://t3x.org/klong/index.html), with small changes for Unicode. The plotting libraries of that distribution are not included.
