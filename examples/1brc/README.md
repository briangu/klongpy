# One Billion Row Challenge

A One Billion Row Challenge pipeline: Python reads the file in parallel and a KlongPy function computes the statistics. This README is for someone who wants to run it and see how the work is split.

KlongPy is not well suited to this problem, because most of the work is processing the data as it is read. KlongPy is fast when the data is already in memory and NumPy can work on a whole array. The example therefore splits the job:

1. Python multiprocessing reads the file in parallel, as the fast solutions do.
2. Each process buffers its data in chunks and passes a chunk to a KlongPy function, which computes the statistics with vector operations.
3. The main process aggregates the results and prints them.

## Run

From this directory, with a `measurements.txt` in the challenge's format:

```bash
kgpy par.kg -- measurements.txt
```

`par.kg` aggregates and prints; `par.py` reads and chunks the file; `worker.kg` is the KlongPy code that each process runs on a chunk.
