Example for 1 Billion Row contest.

KlongPy isn't very suited to this problem because most of the problem is processing the data as it's read.  For KlongPy to shine here, the data needs to already be loaded into memory so NumPy can do its magic.

So for this example, we show a few neat things you can do with KlongPy.

1. Use Python multiprocessing to parallel read the data.  This is part of the normal fast solution.
2. Instead of processing the data as its read, we buffer the data into chunks and pass it to a KlongPy function which does vector processing on it to compute stats.
3. Results are aggregated in the primary process and reported.


To run:

```bash
$ kgpy par.kg -- measurements.txt
```


