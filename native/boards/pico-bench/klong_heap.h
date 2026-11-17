/*
 * The heap of the Pico bench firmware: what klong_config.alloc, .free and .grow do, and the numbers the runtime's heap
 * limit is made from. A header of static functions, so that the firmware (main.c) and the host tests
 * (native/klong-ffi/tests/pico_heap.c and pico_node.c, which run this very source over a static arena of the Pico's
 * size) compile the one source. Nothing here is newlib's: the arena is the runtime's alone, and its state is a few
 * words that a diagnostic (`.heap`, below) can read whole.
 *
 * ONE POOL. The arena is one address-ordered list of free chunks (offset-linked, 8 bytes of header in a free chunk and
 * none in a block: the runtime tells `free` the size) with coalescing of neighbours on every free. There are no regions
 * and no break to move, so there is nothing that "cannot shrink" and no memory is held back by anything but live
 * blocks (and the cache of small blocks, below). Two ends, so that what lives long and what comes and goes do not interleave:
 *   - a block of up to KH_HIGH_MAX (8 KB) comes from the chunk that is HIGHEST in the arena, from its top end: the
 *     frames, boxed values, tables and buffers pack against the top;
 *   - a larger block (a vector, a string) from the LOWEST chunk that fits, from its bottom end; and so, whatever its size, is
 *     everything a parse makes (tokens, tree nodes, the copy of the source: the runtime says so, `htag::TREE`), because a parse
 *     is where a node's memory peaks (the management library's 52 KB on a board with 98 KB) and what it makes is thrown away
 *     but for a few blocks: if those lay among the top's long-lived blocks, the symbols and tables made at the peak would stay
 *     scattered over what the peak had reached and a vector could not pass them (the host model of the boards shows it: the
 *     client of the self-test lost 6% of its largest vector to it with this rule and the next one left out).
 *   The free space between the ends is one span that a vector can have whole.
 * What lives as long as the node (`htag::SYMBOLS`, `htag::GLOBALS`) never takes a block of the cache (below): that is wherever a
 * temporary of an earlier program was, and a name taken from there would stay there for ever. (Both rules need what the runtime
 * says before it allocates, `KLONG_ALLOC_TAG`: they are in the build with KH_TAGS, which KH_DIAG implies, the Pico bench's
 * default; without it every block is placed by its size alone.)
 *
 * THE CACHE. Small blocks (up to 128 bytes, alignment up to 8: what a call of a Klong function allocates) are, when
 * freed, kept on a list of their size class instead of being given back, up to KH_CACHE_MAX bytes in all: a call and a
 * return then take a few instructions each where a walk of the free list takes hundreds of cycles on a Cortex-M0+.
 * A cached block is memory held: it sits where it was allocated and keeps its neighbours from merging. So the cache is
 * given back (`kh_drain`) when an allocation fails and, by the firmware, whenever the runtime has nothing to do
 * (`k_wait`): a node that is idle holds nothing but live blocks.
 *
 * GROWTH. `kh_grow` resizes a block in place when the chunk after it is free (a Vec that doubles, a String that
 * grows): the runtime asks first and copies only if this says no, so a vector's growth does not need the old and the
 * new block at once. A resize that could put the block further along its own end of the pool (a small block higher, a parse's or a large
 * one lower) says no, and the runtime moves it: how a table that grew at a bad moment goes back to where its kind lives, when the
 * runtime shrinks its tables at the end of a program.
 *
 * LIMITS. The runtime counts every block as the pool holds it (heap_round 8, heap_overhead 0), so its limit is the
 * pool's size less a little for holes (`kh_limits`): the cache is not counted, and is given back before a request fails. A request of
 * 512 bytes or more is refused KH_RESERVE bytes before the limit, so that unwinding, printing the error and the next
 * prompt find room. A vector that does not fit is a `memory` error, never a NULL where nothing can be done with it.
 *
 * DIAGNOSTIC. With KH_DIAG defined (the Pico bench build does) the pool keeps a table of its live blocks of 512 bytes or
 * more (address, size, what kind of object asked: the runtime's `KLONG_ALLOC_TAG`) and a byte per 64 of the arena with
 * the kinds of the blocks that start in it, and `kh_diag_*` return the whole state as integers (see `.heap` in
 * docs/native/pico-bench.md). Without it none of that exists.
 */
#ifndef KLONG_HEAP_H
#define KLONG_HEAP_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define KH_SMALL_MAX 128u
#define KH_CLASSES (KH_SMALL_MAX / 8u)
#define KH_CACHE_MAX (4u * 1024)
#define KH_HIGH_MAX 8192u
#define KH_RESERVE (6u * 1024)
#define KH_SLACK_MIN (2u * 1024)
/* What the SDK's own malloc (stdio, USB) is given at the bottom of the heap, apart from the runtime's pool (main.c). */
#define KH_SDK_BYTES (4u * 1024)

#define KH_NIL 0xFFFFFFFFu

typedef struct kh_chunk {
    uint32_t size; /* bytes of the chunk, a multiple of 8 */
    uint32_t next; /* offset of the next free chunk (higher in the arena), or KH_NIL */
} kh_chunk;

static char *kh_base;       /* the pool: 8-aligned */
static uint32_t kh_bytes;   /* its size, a multiple of 8 */
static uint32_t kh_head;    /* offset of the lowest free chunk */
static uint32_t kh_taken, kh_taken_peak; /* bytes out of the pool: live blocks and the cache */
static void *kh_free_list[KH_CLASSES + 1];
static uint32_t kh_cached;                   /* bytes on the lists */
static uint32_t kh_drains, kh_drained_bytes; /* how often the lists were given back, and how much */
static uint32_t kh_fail_small, kh_fail_large; /* requests the pool could not serve (after giving the cache back) */
static uint32_t kh_grown, kh_grow_missed;     /* blocks resized in place; asked to, and the chunk after was not free */

#define KH_AT(o) ((kh_chunk *)(kh_base + (o)))

#ifdef KH_DIAG
#ifndef KH_TAGS
#define KH_TAGS 1
#endif
#endif
#ifdef KH_TAGS
extern volatile uint8_t KLONG_ALLOC_TAG; /* set by the runtime (klong-core, feature heap-tags) before it allocates */
#define KH_TAG_TREE 5u /* htag::TREE */
#define KH_TENURED(tag) ((tag) == 3u || (tag) == 4u) /* htag::SYMBOLS, htag::GLOBALS: they live as long as the node does */
#else
#define KH_TENURED(tag) 0
#endif
#ifdef KH_TAGS
#define KLONG_ALLOC_TAG_NOW (KLONG_ALLOC_TAG)
#else
#define KLONG_ALLOC_TAG_NOW 0
#endif

#ifdef KH_DIAG
#define KH_CELL 64u
#ifndef KH_CELLS_MAX
#define KH_CELLS_MAX 1600u /* 100 KB of arena */
#endif
#define KH_TAB_N 96u
#define KH_TAB_MIN 512u
static uint8_t kh_cell[KH_CELLS_MAX]; /* per 64 bytes of the pool: a bit per kind (1 << tag) of the blocks that started in it */
struct kh_ent {
    uint32_t off, size;
    uint8_t tag;
};
static struct kh_ent kh_tab[KH_TAB_N];
static uint32_t kh_tab_n, kh_tab_bytes, kh_tab_lost;
static void kh_diag_note(void *p, size_t n) {
    uint32_t o = (uint32_t)((char *)p - kh_base);
    uint8_t tag = KLONG_ALLOC_TAG & 7u;
    if ((o / KH_CELL) < KH_CELLS_MAX) kh_cell[o / KH_CELL] |= (uint8_t)(1u << tag);
    if (n >= KH_TAB_MIN) {
        for (uint32_t i = 0; i < KH_TAB_N; i++)
            if (kh_tab[i].size == 0) {
                kh_tab[i].off = o, kh_tab[i].size = (uint32_t)n, kh_tab[i].tag = tag;
                kh_tab_n++, kh_tab_bytes += (uint32_t)n;
                return;
            }
        kh_tab_lost++;
    }
}
static void kh_diag_forget(void *p, size_t n) {
    uint32_t o = (uint32_t)((char *)p - kh_base);
    if (n >= KH_TAB_MIN) {
        for (uint32_t i = 0; i < KH_TAB_N; i++)
            if (kh_tab[i].size && kh_tab[i].off == o) {
                kh_tab_n--, kh_tab_bytes -= kh_tab[i].size;
                kh_tab[i].size = 0;
                return;
            }
    }
}
/* the cells that lie wholly in [lo, hi) hold no block start any more */
static void kh_diag_clear(uint32_t lo, uint32_t hi) {
    for (uint32_t c = (lo + KH_CELL - 1) / KH_CELL; (c + 1) * KH_CELL <= hi && c < KH_CELLS_MAX; c++) kh_cell[c] = 0;
}
#define KH_NOTE(p, n) kh_diag_note((p), (n))
#define KH_FORGET(p, n) kh_diag_forget((p), (n))
#else
#define KH_NOTE(p, n) ((void)0)
#define KH_FORGET(p, n) ((void)0)
#endif

static inline void kh_init(char *start, char *stop) {
    uintptr_t s = ((uintptr_t)start + 7u) & ~(uintptr_t)7u;
    kh_base = (char *)s;
    kh_bytes = (uint32_t)(((uintptr_t)stop - s) & ~(uintptr_t)7u);
    kh_head = 0;
    KH_AT(0)->size = kh_bytes;
    KH_AT(0)->next = KH_NIL;
    kh_taken = kh_taken_peak = kh_cached = 0;
}

/* heap_limit and heap_reserve for klong_config, from the bytes of the pool. The cache is not taken off: it is given back when a
 * request fails, so what the pool holds is the live blocks (all counted) and what is free; the slack is for holes. */
static inline void kh_limits(size_t arena, size_t *limit, size_t *reserve) {
    size_t frag = arena / 64;
    if (frag < KH_SLACK_MIN) frag = KH_SLACK_MIN;
    *limit = arena - frag;
    *reserve = KH_RESERVE;
}

static size_t kh_round(size_t n) {
    n = (n + 7u) & ~(size_t)7u;
    return n ? n : 8u;
}

/* Carve [b, b+n) out of the chunk at offset `o` (whose predecessor in the list is `prev`): what is left of it on either side stays. */
static void kh_carve(uint32_t prev, uint32_t o, uint32_t b, uint32_t n) {
    kh_chunk *c = KH_AT(o);
    uint32_t size = c->size, nx = c->next, after = nx;
    uint32_t tail = o + size - (b + n);
    if (tail) {
        kh_chunk *t = KH_AT(b + n);
        t->size = tail;
        t->next = nx;
        after = b + n;
    }
    uint32_t link = after;
    if (b > o) {
        c->size = b - o;
        c->next = after;
        link = o;
    }
    if (prev == KH_NIL) kh_head = link; else KH_AT(prev)->next = link;
}

/* n bytes (a multiple of 8) from the pool: `high` the chunk highest in the arena, from its top; otherwise the lowest that fits,
 * from its bottom (at an address aligned to `align`). NULL when no chunk has it. */
static void *kh_pool_alloc(size_t n, size_t align, int high) {
    uint32_t prev = KH_NIL, p = kh_head, fit_prev = KH_NIL, fit = KH_NIL, lead = 0;
    for (; p != KH_NIL; prev = p, p = KH_AT(p)->next) {
        uint32_t size = KH_AT(p)->size;
        if (size < n) continue;
        uint32_t l = 0;
        if (align > 8 && !high) {
            uintptr_t a = (uintptr_t)(kh_base + p);
            l = (uint32_t)(((a + align - 1) & ~(uintptr_t)(align - 1)) - a);
            if (size < n + l) continue;
        }
        fit_prev = prev, fit = p, lead = l;
        if (!high) break;
    }
    if (fit == KH_NIL) return NULL;
    uint32_t size = KH_AT(fit)->size;
    uint32_t b = high ? fit + size - (uint32_t)n : fit + lead;
    kh_carve(fit_prev, fit, b, (uint32_t)n);
    kh_taken += (uint32_t)n;
    if (kh_taken > kh_taken_peak) kh_taken_peak = kh_taken;
    return kh_base + b;
}

/* Give n bytes back: into the list in address order, merged with the chunk after and the one before. */
static void kh_pool_free(void *ptr, size_t n) {
    uint32_t o = (uint32_t)((char *)ptr - kh_base), prev = KH_NIL, nx = kh_head;
    while (nx != KH_NIL && nx < o) prev = nx, nx = KH_AT(nx)->next;
    kh_chunk *c = KH_AT(o);
    c->size = (uint32_t)n;
    c->next = nx;
    if (nx != KH_NIL && o + (uint32_t)n == nx) {
        c->size += KH_AT(nx)->size;
        c->next = KH_AT(nx)->next;
    }
    uint32_t lo = o;
    if (prev != KH_NIL) {
        kh_chunk *pc = KH_AT(prev);
        if (prev + pc->size == o) {
            pc->size += c->size;
            pc->next = c->next;
            lo = prev;
        } else {
            pc->next = o;
        }
    } else {
        kh_head = o;
    }
    kh_taken -= (uint32_t)n;
#ifdef KH_DIAG
    kh_diag_clear(lo, KH_AT(lo)->size + lo);
#else
    (void)lo;
#endif
}

/* Give the cached blocks back to the pool (which merges them). Returns the bytes (0: there were none). */
static size_t kh_drain(void) {
    size_t freed = 0;
    for (uint32_t c = 1; c <= KH_CLASSES; c++) {
        void *p = kh_free_list[c];
        while (p) {
            void *next = *(void **)p;
            KH_FORGET(p, c * 8u);
            kh_pool_free(p, c * 8u);
            freed += c * 8u;
            p = next;
        }
        kh_free_list[c] = NULL;
    }
    kh_cached = 0;
    if (freed) {
        kh_drains++;
        kh_drained_bytes += (uint32_t)freed;
    }
    return freed;
}

static void *kh_take(size_t size, size_t align) {
    size_t n = kh_round(size);
    int high = size <= KH_HIGH_MAX && align <= 8;
#ifdef KH_TAGS
    if (KLONG_ALLOC_TAG == KH_TAG_TREE) high = 0; /* a parse (tokens, nodes, the copy of the source) is made and thrown away, or lives as long as its library: with the vectors, not among what lives for ever */
#endif
    void *p = kh_pool_alloc(n, align, high);
    if (!p && kh_drain()) p = kh_pool_alloc(n, align, high);
    if (p) {
        KH_NOTE(p, n);
    } else if (size <= KH_SMALL_MAX) {
        kh_fail_small++;
    } else {
        kh_fail_large++;
    }
    return p;
}

static void *kh_alloc(void *ctx, size_t size, size_t align) {
    (void)ctx;
    if (size <= KH_SMALL_MAX && align <= 8) {
        size_t c = ((size ? size : 1) + 7u) >> 3;
        /* What lives as long as the node (the table of symbols and the globals) never takes a cached block: those are wherever a
         * temporary of an earlier program was, and a name taken from the middle of the heap stays there for ever and pins it. It
         * comes from the pool, at the highest address that is free. */
        void *p = KH_TENURED(KLONG_ALLOC_TAG_NOW) ? NULL : kh_free_list[c];
        if (p) {
            kh_free_list[c] = *(void **)p;
            kh_cached -= (uint32_t)(c << 3);
#ifdef KH_POISON
            memset(p, 0xAA, c << 3);
#endif
            return p;
        }
        p = kh_take(c << 3, 8);
#ifdef KH_POISON
        if (p) memset(p, 0xAA, c << 3);
#endif
        return p;
    }
    void *p = kh_take(size, align);
#ifdef KH_POISON
    if (p && size <= 4096) memset(p, 0xAA, kh_round(size));
#endif
    return p;
}

static void kh_free(void *ctx, void *ptr, size_t size, size_t align) {
    (void)ctx;
    if (!ptr) return;
    if (size <= KH_SMALL_MAX && align <= 8) {
        size_t c = ((size ? size : 1) + 7u) >> 3;
        if (kh_cached + (c << 3) <= KH_CACHE_MAX) {
#ifdef KH_POISON
            memset(ptr, 0xDD, c << 3); /* the host test: a block that is read after it was freed reads 0xDDDD... */
#endif
            *(void **)ptr = kh_free_list[c];
            kh_free_list[c] = ptr;
            kh_cached += (uint32_t)(c << 3);
            return;
        }
    }
    size_t n = kh_round(size);
    KH_FORGET(ptr, n);
#ifdef KH_POISON
    if (n <= 4096) memset(ptr, 0xDD, n);
#endif
    kh_pool_free(ptr, n);
}

#ifdef KH_TAGS
#define KH_IS_TREE() (KLONG_ALLOC_TAG == KH_TAG_TREE)
#else
#define KH_IS_TREE() 0
#endif
/* Is there a chunk further along the end this block belongs to (the top for a small block, the bottom for a large one or a parse's) that
 * would hold n bytes? */
static int kh_better_place(uint32_t off, size_t o, size_t n) {
    int low = n > KH_HIGH_MAX || KH_IS_TREE();
    for (uint32_t p = kh_head; p != KH_NIL; p = KH_AT(p)->next) {
        uint32_t size = KH_AT(p)->size;
        if (size < n) continue;
        if (low) {
            if (p < off) return 1;
        } else if (p > off + (uint32_t)o) {
            return 1;
        } else if (p == off + (uint32_t)o && size >= 2 * (uint32_t)n) {
            /* a chunk right after the block counts when it holds the block twice over: the block moves to its far end and what it leaves
             * merges with what is below (the boards: a 2,720-byte table with 5,936 free right above it). A smaller one does not: moving
             * for it only shifts the block a little and, in the model, costs more than it gives */
            return 1;
        }
    }
    return 0;
}

/* Resize a block in place: shrinking gives the tail back, growing takes the chunk after the block when it is free and long enough.
 * Returns 1 when the block is now `new_size` bytes where it was (the runtime copies and frees otherwise). */
static int kh_grow(void *ctx, void *ptr, size_t old_size, size_t new_size, size_t align) {
    (void)ctx;
    if (!ptr || align > 8) return 0;
    size_t o = kh_round(old_size), n = kh_round(new_size);
    /* a block that is, or becomes, one of the cached size classes is left alone: `free` is told its size and files it by class */
    if (o <= KH_SMALL_MAX || n <= KH_SMALL_MAX) return 0;
    if (n == o) return 1;
    uint32_t off = (uint32_t)((char *)ptr - kh_base);
    if (n < o) {
        /* a block that could be put further along its own end of the pool is moved by the runtime instead (alloc, copy, free): a small block
         * higher, a parse's (a function body kept, shrunk after its parse) or a large one lower. A table that grew while memory was full
         * lies where the memory was free, and what has been freed since is better for it, and for what needs the room under it. */
        if (kh_better_place(off, o, n)) return 0;
        KH_FORGET(ptr, o);
        kh_pool_free((char *)ptr + n, o - n);
        KH_NOTE(ptr, n);
        return 1;
    }
    /* growing is the same: a block of the top end that could be put higher is moved, not extended */
    if (n <= KH_HIGH_MAX && !KH_IS_TREE() && kh_better_place(off, o, n)) {
        kh_grow_missed++;
        return 0;
    }
    uint32_t prev = KH_NIL, p = kh_head, want = off + (uint32_t)o;
    while (p != KH_NIL && p < want) prev = p, p = KH_AT(p)->next;
    if (p != want || KH_AT(p)->size < (uint32_t)(n - o)) {
        kh_grow_missed++;
        return 0;
    }
    uint32_t extra = (uint32_t)(n - o), size = KH_AT(p)->size;
    kh_chunk *c = KH_AT(p);
    uint32_t nx = c->next;
    if (size == extra) {
        if (prev == KH_NIL) kh_head = nx; else KH_AT(prev)->next = nx;
    } else {
        kh_chunk *rest = KH_AT(p + extra);
        rest->size = size - extra;
        rest->next = nx;
        if (prev == KH_NIL) kh_head = p + extra; else KH_AT(prev)->next = p + extra;
    }
    KH_FORGET(ptr, o);
    kh_grown++;
    kh_taken += extra;
    if (kh_taken > kh_taken_peak) kh_taken_peak = kh_taken;
    KH_NOTE(ptr, n);
    return 1;
}

#ifdef KH_DIAG
/* ---- the diagnostic: the pool as integers (`.heap`; docs/native/pico-bench.md) -------------------------------------------------- */

#define KH_PAGE 20u /* entries of a page of the map; blocks: 24 */

/* The summary, fields 0 to 20 (the firmware appends its own from 21). Offsets are from the base of the pool. */
#define KH_SUMMARY_N 21u
static size_t kh_diag_summary(int64_t *out, size_t cap) {
    if (cap < KH_SUMMARY_N) return 0;
    uint32_t free_bytes = 0, largest = 0, chunks = 0, first = KH_NIL, last = KH_NIL;
    for (uint32_t p = kh_head; p != KH_NIL; p = KH_AT(p)->next) {
        uint32_t s = KH_AT(p)->size;
        free_bytes += s, chunks++;
        if (s > largest) largest = s;
        if (first == KH_NIL) first = p;
        last = p;
    }
    int64_t v[KH_SUMMARY_N] = {(int64_t)(uintptr_t)kh_base, kh_bytes, kh_taken, kh_taken_peak, kh_cached, kh_drains, kh_drained_bytes, free_bytes, largest, chunks,
                               first == KH_NIL ? -1 : (int64_t)first, first == KH_NIL ? 0 : KH_AT(first)->size, last == KH_NIL ? -1 : (int64_t)last, last == KH_NIL ? 0 : KH_AT(last)->size,
                               kh_tab_n, kh_tab_bytes, kh_tab_lost, kh_fail_small, kh_fail_large, kh_grown, kh_grow_missed};
    for (uint32_t i = 0; i < KH_SUMMARY_N; i++) out[i] = v[i];
    return KH_SUMMARY_N;
}

/* A page of the map of the whole pool in address order, runs of free and of used bytes: [offset size kind kinds] with kind 0 free,
 * 1 used, and `kinds` the bits (1 << tag) of the blocks that started in a used run (so what lives there). */
static size_t kh_diag_map(uint32_t page, int64_t *out, size_t cap) {
    size_t n = 0, idx = 0, first = (size_t)page * KH_PAGE;
    uint32_t pos = 0;
    uint32_t p = kh_head;
    for (;;) {
        uint32_t run_off, run_size, kind;
        if (p != KH_NIL && p == pos) { /* a free chunk is here */
            run_off = p, run_size = KH_AT(p)->size, kind = 0;
            p = KH_AT(p)->next;
        } else {
            uint32_t stop = p == KH_NIL ? kh_bytes : p;
            if (pos >= kh_bytes) break;
            run_off = pos, run_size = stop - pos, kind = 1;
        }
        pos = run_off + run_size;
        if (idx >= first) {
            if (n + 4 > cap || n / 4 >= KH_PAGE) break;
            uint32_t kinds = 0;
            if (kind) {
                for (uint32_t c = run_off / KH_CELL; c <= (run_off + run_size - 1) / KH_CELL && c < KH_CELLS_MAX; c++) kinds |= kh_cell[c];
            }
            out[n++] = run_off, out[n++] = run_size, out[n++] = kind, out[n++] = kinds;
        }
        idx++;
    }
    return n;
}

/* A page of the live blocks of 512 bytes or more, by address: [offset size tag]. */
#define KH_BLOCKS_PAGE 24u
static size_t kh_diag_blocks(uint32_t page, int64_t *out, size_t cap) {
    uint8_t idx[KH_TAB_N];
    uint32_t n = 0;
    for (uint32_t i = 0; i < KH_TAB_N; i++) {
        if (!kh_tab[i].size) continue;
        uint32_t j = n++;
        while (j > 0 && kh_tab[idx[j - 1]].off > kh_tab[i].off) idx[j] = idx[j - 1], j--;
        idx[j] = (uint8_t)i;
    }
    size_t k = 0;
    for (uint32_t i = page * KH_BLOCKS_PAGE; i < n && i < (page + 1) * KH_BLOCKS_PAGE && k + 3 <= cap; i++) {
        out[k++] = kh_tab[idx[i]].off, out[k++] = kh_tab[idx[i]].size, out[k++] = kh_tab[idx[i]].tag;
    }
    return k;
}

/* The cached blocks of each size class (8, 16, ... 128 bytes). */
static size_t kh_diag_cache(int64_t *out, size_t cap) {
    size_t k = 0;
    for (uint32_t c = 1; c <= KH_CLASSES && k < cap; c++) {
        uint32_t cnt = 0;
        for (void *p = kh_free_list[c]; p; p = *(void **)p) cnt++;
        out[k++] = cnt;
    }
    return k;
}

/* What the pool held when the runtime's count last reached a new high-water mark (the runtime tells, `.heap(300)` starts it again): the
 * counted bytes, what was taken, the free chunks, and the live blocks of 512 bytes or more with their kinds. Taken at most once for each KB
 * the mark rises, so what is kept is the state within 1 KB of the peak. Mode 301 the summary, 302.. pages of blocks. */
#define KH_SNAP_N 48u /* the largest 48 live blocks are kept */
static struct {
    uint32_t valid, used, taken, free_bytes, largest, chunks, tab_n, tab_bytes, cached, drains;
    struct kh_ent tab[KH_SNAP_N];
} kh_snap;
static void kh_snap_take(uint32_t used) {
    uint32_t fb = 0, lg = 0, ch = 0;
    for (uint32_t p = kh_head; p != KH_NIL; p = KH_AT(p)->next) {
        uint32_t s = KH_AT(p)->size;
        fb += s, ch++;
        if (s > lg) lg = s;
    }
    kh_snap.valid = 1, kh_snap.used = used, kh_snap.taken = kh_taken, kh_snap.free_bytes = fb, kh_snap.largest = lg, kh_snap.chunks = ch;
    kh_snap.tab_n = kh_tab_n, kh_snap.tab_bytes = kh_tab_bytes, kh_snap.cached = kh_cached, kh_snap.drains = kh_drains;
    memset(kh_snap.tab, 0, sizeof kh_snap.tab);
    for (uint32_t i = 0, k = 0; i < KH_TAB_N && k < KH_SNAP_N; i++)
        if (kh_tab[i].size) kh_snap.tab[k++] = kh_tab[i];
}
static size_t kh_snap_summary(int64_t *out, size_t cap) {
    if (cap < 10) return 0;
    int64_t v[10] = {kh_snap.valid, kh_snap.used, kh_snap.taken, kh_snap.free_bytes, kh_snap.largest, kh_snap.chunks, kh_snap.tab_n, kh_snap.tab_bytes, kh_snap.cached, kh_snap.drains};
    for (int i = 0; i < 10; i++) out[i] = v[i];
    return 10;
}
static size_t kh_snap_blocks(uint32_t page, int64_t *out, size_t cap) {
    uint8_t idx[KH_SNAP_N];
    uint32_t n = 0;
    for (uint32_t i = 0; i < KH_SNAP_N; i++) {
        if (!kh_snap.tab[i].size) continue;
        uint32_t j = n++;
        while (j > 0 && kh_snap.tab[idx[j - 1]].off > kh_snap.tab[i].off) idx[j] = idx[j - 1], j--;
        idx[j] = (uint8_t)i;
    }
    size_t k = 0;
    for (uint32_t i = page * KH_BLOCKS_PAGE; i < n && i < (page + 1) * KH_BLOCKS_PAGE && k + 3 <= cap; i++)
        out[k++] = kh_snap.tab[idx[i]].off, out[k++] = kh_snap.tab[idx[i]].size, out[k++] = kh_snap.tab[idx[i]].tag;
    return k;
}

/* One entry for the whole diagnostic: mode 0 the summary (heap fields), 1..99 a page of the map (mode - 1), 100..199 a page of the
 * live blocks (mode - 100), 200 the cache, 300 (the runtime's) the high-water mark starts again, 301 and 302.. the heap at the mark. */
static size_t kh_diag(uint32_t mode, int64_t *out, size_t cap) {
    if (mode == 0) return kh_diag_summary(out, cap);
    if (mode < 100) return kh_diag_map(mode - 1, out, cap);
    if (mode < 200) return kh_diag_blocks(mode - 100, out, cap);
    if (mode == 200) return kh_diag_cache(out, cap);
    if (mode == 301) return kh_snap_summary(out, cap);
    if (mode >= 302 && mode < 400) return kh_snap_blocks(mode - 302, out, cap);
    if (mode == 1000) {
        kh_snap_take((uint32_t)cap); /* (the runtime passes the counted bytes in place of the capacity) */
        return 0;
    }
    if (mode == 1001) {
        memset(&kh_snap, 0, sizeof kh_snap);
        return 0;
    }
    return 0;
}
#endif

#endif
