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

/* The runtime counts each block as malloc holds it (`heap_overhead`: the 8-byte header, sizes rounded to 8, at least 16), so
 * what it counts is what its blocks take of the arena. Not counted, and in the arena: the free blocks on the cache (up to
 * KH_CACHE_MAX), and what first-fit leaves in holes (KH_FRAG: 3% of the arena, at least 2 KB). The limit is the arena less
 * those. A request of LARGE bytes or more (klong-rt/src/heap.rs) is refused KH_RESERVE bytes before the limit, so that the
 * smaller requests of unwinding, printing the error and the next prompt find room. Every block, small or large, is taken with
 * the room of an arena that cannot give out more than the limit allows: a vector that does not fit is a `memory` error, never
 * a NULL where nothing can be done with it. */
#define KH_RESERVE (6u * 1024)
#define KH_SLACK_MIN (2u * 1024)

static char *kh_arena_start, *kh_heap_end, *kh_arena_stop;
static int kh_small; /* a small block is being asked for: the reserve is open */
static void *kh_free_list[KH_CLASSES + 1];
static uint32_t kh_cached;                   /* bytes on the lists */
static uint32_t kh_drains, kh_drained_bytes; /* how often malloc failed and the lists were given back, and how much */

static inline void kh_init(char *start, char *stop) {
    kh_arena_start = kh_heap_end = start;
    kh_arena_stop = stop;
}

/* _sbrk: the break may rise to KH_RESERVE bytes below the end of the arena, or to the end for a small block. */
static inline void *kh_sbrk(long incr) {
    char *prev = kh_heap_end;
    long room = kh_arena_stop - kh_heap_end - (kh_small ? 0 : (long)KH_RESERVE);
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

/* Give the cached blocks back to malloc. Returns the bytes (0: there were none). */
static size_t kh_drain(void) {
    size_t freed = 0;
    for (uint32_t c = 1; c <= KH_CLASSES; c++) {
        void *p = kh_free_list[c];
        while (p) {
            void *next = *(void **)p;
            KH_FREE(p);
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
    kh_small = 1;
    p = KH_MALLOC(c << 3);
    if (!p && kh_drain()) p = KH_MALLOC(c << 3);
    kh_small = 0;
#ifdef KH_POISON
    if (p) memset(p, 0xAA, c << 3);
#endif
    return p;
}

static void *kh_alloc(void *ctx, size_t size, size_t align) {
    (void)ctx;
    if (size <= KH_SMALL_MAX && align <= 8) return kh_small_alloc(size ? size : 1);
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
    KH_FREE(ptr);
}

#endif
