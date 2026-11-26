"""Rate and pipeline selection shared by the native and Python benchmark clients."""
import math


def on_time(result, backlog_batches=0):
    batch = result["batch"]
    rate = result["target_rows_s"]
    window = result["window_s"]
    if backlog_batches:
        counts = result["acknowledged_each_second"]
        mid = len(counts) // 2
        at_mid = counts[mid - 1] if mid else 0
        budget = backlog_batches * batch
        second_half = math.floor(rate * (window - mid) / batch) * batch
        return (result["acknowledged_by_deadline"] >= batch
                and result["backlog_at_deadline"] <= budget and result["drain_s"] <= budget / rate
                and all(math.ceil(t * rate / batch) * batch - count <= budget
                        for t, count in enumerate(counts, 1))
                and result["acknowledged_by_deadline"] - at_mid >= second_half - batch)
    whole_batches = math.floor(rate * window / batch)
    return (result["acknowledged_by_deadline"] >= max(1, whole_batches) * batch
            and result["backlog_at_deadline"] <= batch
            and result["drain_s"] <= batch / rate)


def find_rate(start, ceiling, step, confirmations, trial):
    """Return the highest confirmed grid point and the lowest observed miss above it."""
    low, high = 0, None
    candidate = start
    while trial(candidate, "probe"):
        low = candidate
        if candidate == ceiling:
            break
        candidate = min(ceiling, 2 * candidate)
    else:
        high = candidate

    while high is not None and high - low > step:
        candidate = low + ((high - low) // (2 * step)) * step
        if trial(candidate, "probe"):
            low = candidate
        else:
            high = candidate

    def confirmed(rate):
        return all(trial(rate, "confirm") for _ in range(confirmations))

    if low == 0 or confirmed(low):
        return low, high

    # A short-window pass can fail over a minute. Back off, then refine using
    # full confirmations so a noisy short probe cannot become the final answer.
    high = low
    distance = step
    candidate = max(0, high - distance)
    while candidate and not confirmed(candidate):
        high = candidate
        distance *= 2
        candidate = max(0, high - distance)
    low = candidate
    while high - low > step:
        candidate = low + ((high - low) // (2 * step)) * step
        if confirmed(candidate):
            low = candidate
        else:
            high = candidate
    return low, high


def select_shape(measure):
    """Repeat the two fastest single-batch shapes, then try overlapping requests."""
    sizes = (65536, 131072, 262144, 524288)
    baseline = {batch: measure(batch, 4, 1) for batch in sizes}
    fastest = sorted(sizes, key=baseline.get, reverse=True)[:2]
    candidates = []
    for workers in (8, 4):
        for batch in fastest if workers == 8 else reversed(fastest):
            rate = measure(batch, workers, 1)
            if workers == 4:
                rate = (rate + baseline[batch]) / 2
            candidates.append((rate, batch, workers))
    rate, batch, workers = max(candidates)
    pipelines = [(rate, batch, 1)]
    for size in sorted(set(fastest) | {sizes[0]}):
        for depth in (2, 4):
            pipelines.append((measure(size, workers, depth), size, depth))
    _, batch, depth = max(pipelines)
    if depth > 1:
        pipelines.append((measure(batch, workers, 8), batch, 8))
        _, batch, depth = max(pipelines)
    return batch, workers, depth
