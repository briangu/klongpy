/* A model of newlib's malloc (the nano allocator the Pico SDK links) over the arena of the image, KH_TESTED_ARENA
 * bytes: a block has an 8-byte header holding its size, sizes are multiples of 8 (at least 16), the free list is
 * kept in address order and searched first fit, a block is split when the rest is at least 16 bytes, and a freed
 * block is merged with the free neighbours. When nothing fits it asks `_sbrk` for more, as nano does (and extends
 * the last free chunk if that one ends at the break); `kh_sbrk` of klong_heap.h answers, which refuses at the bottom of the
 * slab pool. It returns NULL when that fails. Included after klong_heap.h's macros are defined and before the header. */
#ifndef KLONG_HEAP_ARENA_H
#define KLONG_HEAP_ARENA_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "klong_memory.h"

typedef struct mock_chunk {
    size_t size; /* the whole chunk, header included */
    struct mock_chunk *next; /* in a free chunk: the next free chunk; in a used chunk this word is the first of the payload */
} mock_chunk;

static _Alignas(8) uint8_t mock_arena[KH_TESTED_ARENA];
static mock_chunk *mock_free_list;
static size_t mock_used_bytes, mock_phys_peak;
static long mock_live, mock_live_peak;
static size_t mock_arena_bytes = KH_TESTED_ARENA;
static void *mock_sbrk(long incr);

static void mock_init(void);
static void *mock_malloc(size_t n);
static void mock_free(void *p);
static void *mock_memalign(size_t align, size_t n);

#endif
