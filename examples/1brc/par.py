# based on https://github.com/ifnesi/1brc/blob/main/calculateAverage.py

# time python3 calculateAverage.py
import os
import multiprocessing as mp
import numpy as np
from klongpy import KlongInterpreter

def chunks(
    file_name: str,
    max_cpu: int = 8,
) -> list:
    """Split flie into chunks"""
    cpu_count = min(max_cpu, mp.cpu_count())

    file_size = os.path.getsize(file_name)
    chunk_size = file_size // cpu_count

    start_end = list()
    with open(file_name, "r") as f:

        def is_new_line(position):
            if position == 0:
                return True
            else:
                f.seek(position - 1)
                return f.read(1) == "\n"

        def next_line(position):
            f.seek(position)
            f.readline()
            return f.tell()

        chunk_start = 0
        while chunk_start < file_size:
            chunk_end = min(file_size, chunk_start + chunk_size)

            while not is_new_line(chunk_end):
                chunk_end -= 1

            if chunk_start == chunk_end:
                chunk_end = next_line(chunk_end)

            start_end.append(
                (
                    file_name,
                    chunk_start,
                    chunk_end,
                )
            )

            chunk_start = chunk_end

    return cpu_count, start_end


def _process_file_chunk(
    file_name: str,
    chunk_start: int,
    chunk_end: int,
) -> dict:
    """Process each file chunk in a different process"""
    print(chunk_start, chunk_end)
    stations = []
    temps = []
    klong = KlongInterpreter()
    klong('.l("worker.kg")')
    fn = klong['stats']
    m = {}
    with open(file_name, "r") as f:
        f.seek(chunk_start)
        for line in f:
            chunk_start += len(line)
            if chunk_start > chunk_end:
                break
            location, measurement = line.split(";")
        #  measurement = float(measurement)
            stations.append(location)
            temps.append(measurement)
            if len(stations) > 2**19:
                r = np.asarray([np.asarray(stations), np.asarray(temps,dtype=float)],dtype=object) 
                fn(m,r)
                stations = []
                temps = []

    r = np.asarray([np.asarray(stations), np.asarray(temps,dtype=float)],dtype=object) 
    fn(m,r)
    return m 


def process(
    cpu_count: int,
    start_end: list,
) -> dict:
    """Process data file"""
    with mp.Pool(cpu_count) as p:
        # Run chunks in parallel
        chunk_results = p.starmap(
            _process_file_chunk,
            start_end,
        )

    return chunk_results 

def load(fname):
    cpu_count, start_end = chunks(fname)
    return process(cpu_count, start_end)[0]


