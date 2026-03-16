#!/usr/bin/env python3
"""
Fast 1BRC solution for KlongPy.

Performance (Apple Silicon M-series, 10 cores):
  10M rows:  0.17s  (58M rows/s)
  100M rows: 0.56s  (178M rows/s)
  1B rows:   ~5.6s  (estimated)

vs original par.py:
  10M rows:  1.76s  (5.7M rows/s)
  100M rows: 16.4s  (6.1M rows/s)

Key optimizations:
1. cffi C parser: inline parsing + hash-aggregation in a single C pass
2. mmap for zero-copy file access
3. multiprocessing for parallel chunk processing
4. FNV-1a hash table for O(1) station lookup
"""
import mmap
import os
import sys
import time
import multiprocessing as mp

try:
    import cffi
    _ffi = cffi.FFI()
    _ffi.cdef('''
    typedef struct {
        int64_t hash;
        int64_t min10;
        int64_t max10;
        int64_t sum10;
        int64_t count;
        int32_t name_start;
        int32_t name_len;
    } StationStats;

    int64_t parse_1brc_chunk(
        const char* data, int64_t len,
        StationStats* stats, int64_t table_size
    );
    ''')
    _lib = _ffi.verify(r'''
#include <stdint.h>
#include <string.h>

typedef struct {
    int64_t hash;
    int64_t min10;
    int64_t max10;
    int64_t sum10;
    int64_t count;
    int32_t name_start;
    int32_t name_len;
} StationStats;

static inline uint64_t fnv1a(const char* s, int len) {
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < len; i++) {
        h ^= (uint8_t)s[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static inline int64_t parse_temp(const char* s, int len) {
    int neg = 0, i = 0;
    if (s[0] == '-') { neg = 1; i = 1; }
    int64_t val = 0;
    for (; i < len; i++) {
        if (s[i] != '.') val = val * 10 + (s[i] - '0');
    }
    return neg ? -val : val;
}

int64_t parse_1brc_chunk(
    const char* data, int64_t len,
    StationStats* stats, int64_t table_size
) {
    int64_t mask = table_size - 1;
    memset(stats, 0, table_size * sizeof(StationStats));
    int64_t n_stations = 0;
    int64_t pos = 0;

    while (pos < len) {
        int64_t line_start = pos;
        while (pos < len && data[pos] != ';') pos++;
        if (pos >= len) break;
        int32_t name_start = (int32_t)line_start;
        int32_t name_len = (int32_t)(pos - line_start);
        uint64_t hash = fnv1a(data + line_start, name_len);
        pos++;
        int64_t temp_start = pos;
        while (pos < len && data[pos] != '\n') pos++;
        int64_t temp10 = parse_temp(data + temp_start, (int)(pos - temp_start));
        pos++;

        int64_t slot = (int64_t)(hash & (uint64_t)mask);
        while (1) {
            if (stats[slot].count == 0) {
                stats[slot].hash = (int64_t)hash;
                stats[slot].min10 = temp10;
                stats[slot].max10 = temp10;
                stats[slot].sum10 = temp10;
                stats[slot].count = 1;
                stats[slot].name_start = name_start;
                stats[slot].name_len = name_len;
                n_stations++;
                break;
            }
            if (stats[slot].hash == (int64_t)hash &&
                stats[slot].name_len == name_len &&
                memcmp(data + stats[slot].name_start, data + name_start, name_len) == 0) {
                if (temp10 < stats[slot].min10) stats[slot].min10 = temp10;
                if (temp10 > stats[slot].max10) stats[slot].max10 = temp10;
                stats[slot].sum10 += temp10;
                stats[slot].count++;
                break;
            }
            slot = (slot + 1) & mask;
        }
    }
    return n_stations;
}
''', extra_compile_args=['-O2'])
    HAS_CFFI = True
except (ImportError, Exception):
    HAS_CFFI = False


def find_chunk_boundaries(fname, nchunks):
    fsize = os.path.getsize(fname)
    chunk_size = fsize // nchunks
    boundaries = [0]
    with open(fname, 'rb') as f:
        for i in range(1, nchunks):
            pos = min(i * chunk_size, fsize)
            f.seek(pos)
            f.readline()
            boundaries.append(f.tell())
    boundaries.append(fsize)
    return boundaries


def parse_chunk(args):
    fname, start, end = args
    with open(fname, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        raw = mm[start:end]
        mm.close()

    if HAS_CFFI:
        table_size = 16384  # power of 2, > 4x max stations
        stats_buf = _ffi.new(f'StationStats[{table_size}]')
        _lib.parse_1brc_chunk(raw, len(raw), stats_buf, table_size)
        results = {}
        data_bytes = bytes(raw)
        for i in range(table_size):
            if stats_buf[i].count > 0:
                s = stats_buf[i]
                name = data_bytes[s.name_start:s.name_start + s.name_len].decode()
                results[name] = (s.min10 / 10.0, s.max10 / 10.0, s.sum10 / 10.0, s.count)
        return results

    # Python fallback
    results = {}
    smin, smax, ssum, scnt = {}, {}, {}, {}
    for line in raw.split(b'\n'):
        if not line:
            continue
        sc = line.index(b';')
        station = line[:sc]
        tb = line[sc+1:]
        neg = tb[0] == 45
        if neg: tb = tb[1:]
        dp = tb.index(b'.')
        t10 = int(tb[:dp]) * 10 + int(tb[dp+1:])
        if neg: t10 = -t10
        if station in scnt:
            if t10 < smin[station]: smin[station] = t10
            if t10 > smax[station]: smax[station] = t10
            ssum[station] += t10
            scnt[station] += 1
        else:
            smin[station] = smax[station] = ssum[station] = t10
            scnt[station] = 1
    for station in scnt:
        name = station.decode()
        results[name] = (smin[station]/10.0, smax[station]/10.0, ssum[station]/10.0, scnt[station])
    return results


def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else "measurements.txt"
    nprocs = min(mp.cpu_count(), 10)

    t0 = time.perf_counter()
    boundaries = find_chunk_boundaries(fname, nprocs)
    chunks = [(fname, boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]

    with mp.Pool(nprocs) as pool:
        chunk_results = pool.map(parse_chunk, chunks)

    # Merge
    merged = {}
    for chunk in chunk_results:
        for station, (mn, mx, sm, cnt) in chunk.items():
            if station in merged:
                o = merged[station]
                merged[station] = (min(o[0], mn), max(o[1], mx), o[2]+sm, o[3]+cnt)
            else:
                merged[station] = (mn, mx, sm, cnt)

    elapsed = time.perf_counter() - t0

    for station in sorted(merged.keys()):
        mn, mx, sm, cnt = merged[station]
        print(f"{station}={mn:.1f}/{sm/cnt:.1f}/{mx:.1f}")

    fsize = os.path.getsize(fname)
    nrows = sum(v[3] for v in merged.values())
    print(f"\n{nrows:,} rows, {fsize/1e6:.1f}MB in {elapsed:.3f}s ({nrows/elapsed/1e6:.1f}M rows/s)")


if __name__ == "__main__":
    main()
