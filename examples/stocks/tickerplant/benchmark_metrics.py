"""Independent checks and timing summaries for ticker-plant sessions."""
import math
import statistics


EPOCH = 1700000000000000000
SYMBOLS = 1024


def quantile(values, q):
    return sorted(values)[math.ceil(q * len(values)) - 1] if values else None


def timings(samples):
    return dict(samples=len(samples), p50_ms=statistics.median(samples) if samples else None,
                p99_ms=quantile(samples, .99), max_ms=max(samples, default=None))


def last_symbol_seq(end, symbol):
    return end - 1 - ((end - 1 - symbol) % SYMBOLS)


def volume_before(end):
    cycles, rest = divmod(end, 97)
    return cycles * (97 * 98 // 2) + rest * (rest + 1) // 2


def archive_checks(day, first, end):
    # Bound the distinct-count hash table even when a session has tens of millions of rows.
    for low in range(first, end, 1000000):
        high = min(low + 1000000, end)
        sql = (f"SELECT count(*),count(DISTINCT seq),count(*) FILTER (WHERE "
               f"ts IS DISTINCT FROM {EPOCH}+seq*1000+(seq%523)*20 OR sym IS DISTINCT FROM seq%{SYMBOLS} "
               "OR px IS DISTINCT FROM 10000+(seq*17)%100003 OR qty IS DISTINCT FROM 1+seq%97) "
               f"FROM trades WHERE dayno={day} AND seq>={low} AND seq<{high}")
        yield high - low, sql
