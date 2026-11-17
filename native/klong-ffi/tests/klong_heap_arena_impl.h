/* The definitions of klong_heap_arena.h, included after klong_heap.h (they call kh_sbrk). */
#ifndef KLONG_HEAP_ARENA_IMPL_H
#define KLONG_HEAP_ARENA_IMPL_H

static void *mock_sbrk(long incr) { return kh_sbrk(incr); }

static void mock_init(void) {
    kh_init((char *)mock_arena, (char *)mock_arena + (mock_arena_bytes & ~7u));
    mock_free_list = NULL;
}

static void *mock_malloc(size_t n) {
    size_t need = (n + 8u + 7u) & ~7u;
    if (need < 16) need = 16;
    mock_chunk *prev = NULL, *c = mock_free_list;
    for (; c; prev = c, c = c->next)
        if (c->size >= need) break;
    if (!c) {
        /* nothing fits: grow the heap; the last free chunk, if it ends at the break, is extended */
        mock_chunk *last = prev;
        size_t have = last && (uint8_t *)last + last->size == (uint8_t *)kh_heap_end ? last->size : 0;
        void *p = mock_sbrk((long)(need - have));
        if (p == (void *)-1) return NULL;
        if (have) {
            last->size += need - have;
            c = last;
            prev = NULL;
            for (mock_chunk *q = mock_free_list; q && q != last; q = q->next) prev = q;
        } else {
            c = p;
            c->size = need;
            c->next = NULL;
            if (prev) prev->next = c; else mock_free_list = c;
        }
    }
    if (c->size - need >= 16) {
        mock_chunk *rest = (mock_chunk *)((uint8_t *)c + need);
        rest->size = c->size - need;
        rest->next = c->next;
        c->size = need;
        if (prev) prev->next = rest; else mock_free_list = rest;
    } else if (prev) {
        prev->next = c->next;
    } else {
        mock_free_list = c->next;
    }
    mock_used_bytes += c->size;
    memset((uint8_t *)c + 16, 0xAA, c->size - 16);
    mock_live++; if (mock_live > mock_live_peak) mock_live_peak = mock_live;
    if (mock_used_bytes > mock_phys_peak) mock_phys_peak = mock_used_bytes;
    return (uint8_t *)c + 8;
}

static void mock_free(void *p) {
    if (!p) return;
    mock_chunk *c = (mock_chunk *)((uint8_t *)p - 8);
    mock_used_bytes -= c->size;
    mock_live--;
    memset((uint8_t *)c + 16, 0xDD, c->size - 16);
    mock_chunk *prev = NULL, *n = mock_free_list;
    while (n && n < c) prev = n, n = n->next;
    c->next = n;
    if (prev) prev->next = c; else mock_free_list = c;
    if (n && (uint8_t *)c + c->size == (uint8_t *)n) { /* merge with the one after */
        c->size += n->size;
        c->next = n->next;
    }
    if (prev && (uint8_t *)prev + prev->size == (uint8_t *)c) { /* and the one before */
        prev->size += c->size;
        prev->next = c->next;
    }
}

static void *mock_memalign(size_t align, size_t n) {
    if (align > 8) abort(); /* the runtime asks for 8 at most; the firmware would use newlib's memalign */
    return mock_malloc(n);
}

/* the free chunks, for a report: how many and the largest (bytes, header included), the unused room between the
 * break and the reserve counted in */
static size_t mock_largest_free(size_t *count) {
    size_t big = 0, n = 0;
    for (mock_chunk *c = mock_free_list; c; c = c->next) {
        size_t sz = c->size;
        if ((uint8_t *)c + c->size == (uint8_t *)kh_heap_end) sz += (size_t)(kh_arena_stop - kh_heap_end - KH_RESERVE);
        n++, big = sz > big ? sz : big;
    }
    if (count) *count = n;
    return big;
}

#endif
