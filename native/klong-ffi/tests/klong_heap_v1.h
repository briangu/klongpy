/* FROZEN: an earlier allocator of the Pico firmware, kept so that the host model of newlib's malloc
 * (klong_heap_arena.h) can be checked against what that image did on the boards: the largest vector of the client of the self-test
 * smaller than the server's, and `.l("mgmt")` refused after it. Not the firmware's: that is boards/pico-bench/klong_heap.h. Built with
 * pico_node.c -DPICO_V1. */
/*
 * The heap of the Pico bench firmware: what klong_config.alloc and .free do, the `_sbrk` under malloc, and the
 * numbers the runtime's heap limit is made from. A header of static functions, so that the firmware (main.c) and
 * the host test (native/klong-ffi/tests/pico_heap.c, which runs the same code over a model of newlib's malloc with
 * an arena of the Pico's size) compile the one source.
 *
 * Everything is malloc's. Small blocks (up to 128 bytes, alignment up to 8: what a call of a Klong function
 * allocates: argument frames, projection layers, boxed values) are rounded to 8 bytes and, when freed, kept on a
 * free list of their size class instead of being given back, up to KH_CACHE_MAX bytes in all (the runtime tells
 * `free` the size): a call and a return of a Klong function then take a few instructions each, where newlib's
 * malloc and free walk a list and merge neighbours (a few hundred cycles on a Cortex-M0+, several times a call).
 * What is more than the cache holds goes back to malloc at once. So the memory a deep recursion had for its
 * frames is malloc's again when it returns, bar the 8 KB of the cache, and `kh_drain` gives that back too when
 * malloc fails: a vector can have the whole heap. (A first version took slabs of 2 KB from malloc and kept
 * them: a block of long life in each slab kept it, and a recursion left the heap in pieces. The test shows it.)
 *
 * Nothing in the runtime may fail where it cannot return an error (a frame, a reference count: a failure is a
 * reset), so the top KH_RESERVE bytes of the arena are for small blocks only: `kh_sbrk` refuses a request of any
 * other size that would touch them. A vector can use everything below, and the runtime's own heap limit keeps
 * its large requests KH_RESERVE short of its limit (klong-rt/src/heap.rs).
 */
#ifndef KLONG_HEAP_H
#define KLONG_HEAP_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifndef KH_MALLOC
#include <malloc.h>
#define KH_MALLOC malloc
#define KH_FREE free
#define KH_MEMALIGN memalign
#endif

#define KH_SMALL_MAX 128u
#define KH_CLASSES (KH_SMALL_MAX / 8u)
#define KH_CACHE_MAX (8u * 1024)

/* The runtime counts each block as the arena holds it (`heap_round` 8: sizes rounded up to 8; `heap_overhead` 0: the blocks of the
 * region, which are nearly all of them, have no header; the few over KH_REGION_MAX are malloc's, with 8 bytes more each, which
 * the slack below covers). Not counted, and in the arena: the free blocks on the cache (up to KH_CACHE_MAX), and what first-fit
 * leaves in holes (KH_FRAG: 3% of the arena, at least 2 KB). The limit is the arena less those. A request of LARGE bytes or
 * more (klong-rt/src/heap.rs) is refused KH_RESERVE bytes before the limit, so that the smaller requests of unwinding, printing
 * the error and the next prompt find room. A vector that does not fit is a `memory` error, never a NULL where nothing can be
 * done with it. */
#define KH_RESERVE (6u * 1024)
#define KH_SLACK_MIN (2u * 1024)

/* Blocks of up to KH_REGION_MAX bytes (what is not a vector: argument frames, boxed values, the tables that grow with the
 * symbols, the buffers of a connection) are not newlib's: they live in a region at the top of the arena, which grows down from
 * the end of the arena and shrinks back when its lowest block is freed, in a first-fit list in address order with no header (the
 * runtime tells `free` the size). What is longer lived than a program (the table of symbols, of globals, a connection's buffer: they
 * grow by doubling while a library loads, and the last, grown block stays) then sits at the top, and the vectors and strings
 * and parse trees of over 8 KB, which come and go, have the bottom of the arena to themselves: a block left in the middle of it
 * by a program that has gone is what made the largest vector of a node that had loaded the management library 20 KB smaller. */
#define KH_REGION_MAX 8192u
typedef struct kh_chunk {
    struct kh_chunk *next;
    size_t size;
} kh_chunk;
#define KH_GRAIN (sizeof(kh_chunk))
static kh_chunk *kh_rfree;     /* the region's free chunks, in address order */
static char *kh_top_lo;        /* the bottom of the region */
static uint32_t kh_region_bytes; /* in use in the region */
static char *kh_arena_start, *kh_heap_end, *kh_arena_stop;
static void *kh_free_list[KH_CLASSES + 1];
static uint32_t kh_cached;                   /* bytes on the lists */
static uint32_t kh_drains, kh_drained_bytes; /* how often malloc failed and the lists were given back, and how much */

static inline void kh_init(char *start, char *stop) {
    kh_arena_start = kh_heap_end = start;
    kh_arena_stop = stop;
    kh_top_lo = stop;
}

/* _sbrk: the break may rise to KH_RESERVE bytes below the end of the arena, or to the end for a small block. */
static inline void *kh_sbrk(long incr) {
    char *prev = kh_heap_end;
    long room = kh_top_lo - kh_heap_end - (long)KH_RESERVE; /* the top KH_RESERVE bytes of the room are the region's */
    if (incr > room) return (void *)-1;
    kh_heap_end += incr;
    return prev;
}

/* heap_limit and heap_reserve for klong_config, from the bytes of the arena. */
static inline void kh_limits(size_t arena, size_t *limit, size_t *reserve) {
    size_t frag = arena / 32;
    if (frag < KH_SLACK_MIN) frag = KH_SLACK_MIN;
    *limit = arena - KH_CACHE_MAX - frag;
    *reserve = KH_RESERVE;
}

/* A block of up to KH_REGION_MAX bytes: from the region, and from malloc when the region cannot grow (what is free in
 * malloc's part below its break, after a vector has been and gone, is room for it too); and back to where it came from, which
 * its address says: the region is above every block of malloc's. */
#define KH_IN_REGION(p) ((char *)(p) >= kh_top_lo)
static void *kh_region_alloc(size_t size);
static void kh_region_free(void *ptr, size_t size);
static void *kh_block_alloc(size_t size) {
    void *p = kh_region_alloc(size);
    if (!p) p = KH_MALLOC(size < 16 ? 16 : size);
    return p;
}
static void kh_block_free(void *ptr, size_t size) {
    if (KH_IN_REGION(ptr)) kh_region_free(ptr, size);
    else KH_FREE(ptr);
}

#ifndef KH_ON_ALLOC
#define KH_ON_ALLOC(p, n)
#define KH_ON_FREE(p)
#endif

static size_t kh_round(size_t n) {
    return (n + KH_GRAIN - 1) & ~(KH_GRAIN - 1);
}

/* A block of the region: first fit, and the region grows down when nothing fits. */
static void *kh_region_alloc(size_t size) {
    size_t n = kh_round(size ? size : 1);
    /* the fit highest in the arena, and its end of it: the region packs against the top of the arena and what is free stays at the
     * bottom, where it can be given back */
    kh_chunk *prev = NULL, *fit_prev = NULL, *fit = NULL;
    for (kh_chunk *c = kh_rfree; c; prev = c, c = c->next)
        if (c->size >= n) fit = c, fit_prev = prev;
    if (fit) {
        void *p;
        if (fit->size - n >= KH_GRAIN) {
            fit->size -= n;
            p = (char *)fit + fit->size;
        } else {
            p = fit;
            if (fit_prev) fit_prev->next = fit->next; else kh_rfree = fit->next;
        }
        kh_region_bytes += (uint32_t)n;
        KH_ON_ALLOC(p, n);
        return p;
    }
    /* nothing fits: lower the bottom of the region (a free chunk at the bottom is extended instead of left in front of a new block) */
    size_t have = (kh_rfree && (char *)kh_rfree == kh_top_lo) ? kh_rfree->size : 0;
    if (have >= n) return NULL; /* unreachable: it would have fitted */
    size_t need = n - have;
    if ((size_t)(kh_top_lo - kh_heap_end) < need) return NULL;
    kh_top_lo -= need;
    if (have) kh_rfree = kh_rfree->next; /* the chunk is the head of the list and becomes the block */
    kh_region_bytes += (uint32_t)n;
    KH_ON_ALLOC(kh_top_lo, n);
    return kh_top_lo;
}

static void kh_region_free(void *ptr, size_t size) {
    size_t n = kh_round(size ? size : 1);
    kh_chunk *c = ptr, *prev = NULL, *next = kh_rfree;
    kh_region_bytes -= (uint32_t)n;
    KH_ON_FREE(ptr);
    while (next && next < c) prev = next, next = next->next;
    c->size = n;
    c->next = next;
    if (prev) prev->next = c; else kh_rfree = c;
    if (next && (char *)c + c->size == (char *)next) { /* merge with the one after */
        c->size += next->size;
        c->next = next->next;
    }
    if (prev && (char *)prev + prev->size == (char *)c) { /* and the one before */
        prev->size += c->size;
        prev->next = c->next;
        c = prev;
    }
    if ((char *)c == kh_top_lo) { /* the lowest block of the region is free: the region shrinks, the room goes back to the heap */
        kh_top_lo += c->size;
        kh_rfree = c->next;
    }
}

/* Give the cached blocks back to the region (which merges them). Returns the bytes (0: there were none). */
static size_t kh_drain(void) {
    size_t freed = 0;
    for (uint32_t c = 1; c <= KH_CLASSES; c++) {
        void *p = kh_free_list[c];
        while (p) {
            void *next = *(void **)p;
            kh_block_free(p, c * 8u);
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

static void *kh_small_alloc(size_t size) {
    size_t c = (size + 7u) >> 3;
    void *p = kh_free_list[c];
    if (p) {
        kh_free_list[c] = *(void **)p;
        kh_cached -= (uint32_t)(c << 3);
#ifdef KH_POISON
        memset(p, 0xAA, c << 3);
#endif
        return p;
    }
    p = kh_block_alloc(c << 3);
    if (!p && kh_drain()) p = kh_block_alloc(c << 3);
#ifdef KH_POISON
    if (p) memset(p, 0xAA, c << 3);
#endif
    return p;
}

static void *kh_alloc(void *ctx, size_t size, size_t align) {
    (void)ctx;
    if (size <= KH_SMALL_MAX && align <= 8) return kh_small_alloc(size ? size : 1);
    if (size <= KH_REGION_MAX && align <= 8) {
        void *r = kh_block_alloc(size);
        if (!r && kh_drain()) r = kh_block_alloc(size);
        return r;
    }
    void *p = align <= 8 ? KH_MALLOC(size) : KH_MEMALIGN(align, size);
    if (!p && kh_drain()) p = align <= 8 ? KH_MALLOC(size) : KH_MEMALIGN(align, size);
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
    if (size <= KH_REGION_MAX && align <= 8) {
        kh_block_free(ptr, size);
        return;
    }
    KH_FREE(ptr);
}

#endif
