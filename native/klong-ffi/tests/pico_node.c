/*
 * One node of the Pico bench as a host process: the runtime of klong-ffi with the lane and the boot self-test, the firmware's
 * own allocator (boards/pico-bench/klong_heap.h) over a static arena of a given size, the firmware's heap limits and its
 * diagnostic (`.heap`), and a lane that is two file descriptors: 3 (frames from the switch, non-blocking) and 4 (frames to
 * it). tests/test_native_pico_pair.py is the switch and the supervisor: it starts two of these (nodes 2 and 3 of the bench: the
 * higher one runs the self-test as a client of the lower one) and watches what the supervisor sees.
 *
 *   pico_node ARENA_BYTES
 *
 * Built with -DPICO_V1 it is the allocator of the image a232666c instead (klong_heap_v1.h over a model of newlib's malloc,
 * klong_heap_arena.h), which the boards showed to leave the client of the self-test with a smaller largest vector than the
 * server's and no room for the management library. A NULL is an answer in both (the firmware takes memory from `__real_malloc`);
 * what cannot fail and does ends in `fatal`: the process prints FATAL and exits, as the board reboots.
 *
 * What the host cannot show: the runtime here is the 64-bit build, whose objects are about twice the Pico's, and its library peak
 * is several times the board's; the arena is therefore chosen so that the proportions (the peak against the arena, the idle
 * use against it) are the board's, and the test asserts ratios, never sizes.
 */
#define _DARWIN_C_SOURCE
#define _DEFAULT_SOURCE
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "klong.h"

#ifdef PICO_V1
#include "klong_heap_arena.h"
#define KH_MALLOC mock_malloc
#define KH_FREE mock_free
#define KH_MEMALIGN mock_memalign
#define KH_POISON 1
#include "klong_heap_v1.h"
#include "klong_heap_arena_impl.h"
#else
#define KH_DIAG 1
#define KH_POISON 1 /* freed blocks are filled with 0xDD, new ones with 0xAA: a use after free reads garbage */
#ifndef KH_CELLS_MAX
#define KH_CELLS_MAX (KH_TESTED_ARENA / 64u + 1u)
#endif
#include "klong_heap.h"
static _Alignas(8) uint8_t arena_mem[KH_TESTED_ARENA];
#endif

#ifndef PICO_V1
/* `.heap`: the summary has the heap's fields only here (the board appends its own). */
static size_t node_diag(uint32_t mode, int64_t *out, size_t cap) {
    return kh_diag(mode, out, cap);
}
#endif

static void on_usr1(int s) {
    (void)s;
    size_t u, p, l;
    klong_heap(&u, &p, &l);
    fprintf(stderr, "COUNTED used %zu peak %zu limit %zu\n", u, p, l);
#ifndef PICO_V1
    fprintf(stderr, "POOL taken %u of %u, cached %u, free chunks:", (unsigned)kh_taken, (unsigned)kh_bytes, (unsigned)kh_cached);
    for (uint32_t c = kh_head; c != KH_NIL; c = KH_AT(c)->next) fprintf(stderr, " %u+%u", (unsigned)c, (unsigned)KH_AT(c)->size);
    fprintf(stderr, "\n");
#else
    mock_dump(stderr);
#endif
}
static void h_write(void *ctx, const uint8_t *b, size_t n) { (void)ctx, (void)b, (void)n; }
static size_t h_read(void *ctx, uint8_t *b, size_t cap) { (void)ctx, (void)b, (void)cap; return 0; }
static uint64_t h_now_ms(void *ctx) {
    (void)ctx;
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000u + (uint64_t)t.tv_nsec / 1000000u;
}
static void h_fatal(void *ctx, const char *msg, size_t len) {
    (void)ctx;
    fprintf(stderr, "FATAL %.*s\n", (int)len, msg);
    _exit(98);
}
static size_t l_read(void *ctx, uint8_t *b, size_t cap) {
    (void)ctx;
    ssize_t n = read(3, b, cap);
    return n > 0 ? (size_t)n : 0;
}
static size_t l_write(void *ctx, const uint8_t *b, size_t n) {
    (void)ctx;
    ssize_t w = write(4, b, n);
    return w > 0 ? (size_t)w : 0;
}

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    size_t arena = (size_t)strtoul(argv[1], NULL, 10);
    if (arena > KH_TESTED_ARENA) return 2;
#ifdef PICO_V1
    mock_arena_bytes = arena;
    mock_init();
#else
    kh_init((char *)arena_mem, (char *)arena_mem + arena);
    arena = kh_bytes;
#endif
    signal(SIGUSR1, on_usr1);
    fcntl(3, F_SETFL, O_NONBLOCK);
    klong_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.abi_version = KLONG_ABI_VERSION;
    cfg.size = sizeof cfg;
    cfg.alloc = kh_alloc;
    cfg.free = kh_free;
#ifndef PICO_V1
    cfg.grow = kh_grow;
#endif
    kh_limits(arena, &cfg.heap_limit, &cfg.heap_reserve);
    cfg.heap_overhead = 0;
    cfg.heap_round = 8;
    cfg.stack_bytes = 4u << 20;
    cfg.write = h_write;
    cfg.read = h_read;
    cfg.now_ms = h_now_ms;
    cfg.fatal = h_fatal;
    cfg.name = "pico-node";
    cfg.flags = KLONG_FLAG_CRLF;
    if (klong_init(&cfg) != KLONG_OK) return 3;
#ifndef PICO_V1
    klong_set_diag(node_diag);
#endif
    klong_lane_config lane;
    memset(&lane, 0, sizeof lane);
    lane.abi_version = KLONG_ABI_VERSION;
    lane.size = sizeof lane;
    lane.lane_read = l_read;
    lane.lane_write = l_write;
    lane.node = -1;
    lane.seed = (uint32_t)getpid();
    lane.flags = KLONG_LANE_SELFTEST;
    lane.baud = 4500000;
    lane.note = getenv("PICO_NOTE");
    if (klong_lane_init(&lane) != KLONG_OK) return 4;
    for (;;) {
        if (klong_step() != 1) {
#ifndef PICO_V1
            kh_drain(); /* idle: the cache goes back (the firmware's k_wait) */
#endif
            usleep(200);
        }
    }
}
