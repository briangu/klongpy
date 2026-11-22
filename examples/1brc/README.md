# One Billion Row Challenge

This Klong example computes each station's minimum, mean and maximum temperature
from a UTF-8 measurements file. The input uses `station;temperature` lines with
one decimal digit per measurement. [Challenge input format](https://github.com/gunnarmorling/1brc#rules-and-limits)

From the repository root, run the program with either desktop command.

```bash
kg examples/1brc/par.kg /path/to/measurements.txt
kgpy examples/1brc/par.kg -- /path/to/measurements.txt
```

Both commands print one line in the challenge format, for example
`{Oslo=-2.0/0.3/3.0, Zürich=10.0/11.0/12.0}`. Stations follow Java's string
ordering. The program sums integer tenths and rounds the mean to the nearest
tenth, with ties toward positive infinity.

The built-in DuckDB engine scans the file in parallel and aggregates the rows
without loading the measurements into Klong arrays. Station names that start
with a byte order mark at the beginning of the file or contain a carriage return
use a line-by-line Klong fallback. The fallback retains one running aggregate
per station and uses the same SQL for rounding and formatting.
